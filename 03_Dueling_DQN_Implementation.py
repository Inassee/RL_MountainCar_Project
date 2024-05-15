import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import pygame
import cv2  # Optional, for saving video frames if needed

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer_sizes):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[2], action_size)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[2], 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + (advantage - advantage.mean())

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def add(self, experience):
        max_prio = max(self.priorities, default=1.0)  # Use the maximum priority as default for new entries
        self.buffer.append(experience)
        self.priorities.append(max_prio)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = np.array(self.priorities)
            probs = prios ** self.alpha
            probs /= probs.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]
            total = len(self.buffer)
            weights = (total * probs[indices]) ** (-beta)
            weights /= weights.max()
            return samples, indices, np.array(weights, dtype=np.float32)
        else:
            return [], [], []

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_layer_sizes, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.Q = DuelingDQN(state_size, action_size, hidden_layer_sizes)
        self.Q_target = DuelingDQN(state_size, action_size, hidden_layer_sizes)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.Q(state)
        return torch.argmax(action_values).item()

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def replay(self, batch_size):
        samples, indices, weights = self.replay_buffer.sample(batch_size)
        if not samples:
            return
        
        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_q_values = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.Q_target(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = (current_q_values - expected_q_values.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

    def update_target_network(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

# Correct the main loop to use `add_experience`


# Initialize environment
env = gym.make('MountainCar-v0', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size, [128, 128, 64])
batch_size = 64
episodes = 200

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Mountain Car")

for e in tqdm(range(episodes)):
    state = env.reset()[0]
    total_reward = 0
    done = False
    frame_idx = 0

    while not done:
        pygame.event.pump()
        action = agent.act(state)  # Remove frame_idx
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        total_reward += reward
        agent.add_experience(state, action, reward, next_state, done)
        state = next_state

        # Render to pygame
        frame = env.render()
        if frame is not None and len(frame.shape) == 3:
            frame = np.transpose(frame, (1, 0, 2))  # Transpose for correct orientation
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.update()

        if len(agent.replay_buffer.buffer) > batch_size:
            agent.replay(batch_size)

        frame_idx += 1

    agent.update_target_network()
    if e % 10 == 0:
        print(f"Episode: {e}, Total reward: {total_reward}, Epsilon: {agent.epsilon}")

pygame.quit()
env.close()