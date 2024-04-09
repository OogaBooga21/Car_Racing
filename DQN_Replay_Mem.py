import numpy as np
import torch
import random
from collections import deque

class Replay_Memory:
    memory_fill_index = 0
    
    def __init__(self,capacity,stack_size,img_height,img_width):
        self.capacity = capacity
        self.memory_fill_index = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.memory = deque(maxlen=capacity)
        
    def fill_rate(self):
        return self.memory_fill_index
    
    def get_capacity(self):
        return self.capacity
    
    def store_memory(self, state, action, reward, terminal, next_state):
        state = np.stack(state, axis=0)  # Convert the list of states to a NumPy array
        next_state = np.stack(next_state, axis=0)  # Convert the list of next_states to a NumPy array
        
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        terminal = torch.tensor(terminal, dtype=torch.int64, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        self.memory.append((state, action, reward, terminal, next_state))
        
        self.memory_fill_index = min(self.memory_fill_index + 1, self.capacity)
        # print(self.memory)
        
        
    def random_memory_batch(self, batch_size):
        # weighted_indices = np.arange(len(self.memory))[::-1]  # Reverse order to give more weight to recent experiences
        # sampled_indices = random.choices(weighted_indices, k=batch_size)
        
        # batch = random.sample(self.memory, batch_size)
        # states, actions, rewards, terminals, next_states = zip(*batch)

        # states = torch.stack(states).to(self.device)
        # actions = torch.stack(actions).to(self.device)
        # rewards = torch.stack(rewards).to(self.device)
        # terminals = torch.stack(terminals).to(self.device)
        # next_states = torch.stack(next_states).to(self.device)

        # return states, actions, rewards, terminals, next_states
        memory_length = len(self.memory)
    
        weights = np.linspace(start=0.1, stop=1.0, num=memory_length) 
        weights /= weights.sum()

        sampled_indices = np.random.choice(np.arange(memory_length), size=batch_size, p=weights, replace=False)

        batch = [self.memory[idx] for idx in sampled_indices]
        states, actions, rewards, terminals, next_states = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        terminals = torch.stack(terminals).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        return states, actions, rewards, terminals, next_states