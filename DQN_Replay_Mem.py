import numpy as np
import torch
import random
from collections import deque

class Replay_Memory:
    memory_fill_index = 0
    
    def __init__(self,capacity,stack_size,img_height,img_width):
        self.capacity = capacity
        # self.memory_index = 0
        self.memory_fill_index = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.memory = deque(maxlen=capacity)
        
        
        # self.memory_states = torch.zeros((capacity, stack_size, img_height, img_width), dtype=torch.float32, device=self.device)
        # self.memory_actions = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        # self.memory_rewards = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        # self.memory_terminal = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        # self.memory_next_states = torch.zeros((capacity, stack_size, img_height, img_width), dtype=torch.float32, device=self.device)
        
        #making up the touple (s,a,r,s') + to know if we're done for the other q branch
        
    def fill_rate(self):
        return self.memory_fill_index
    
    def get_capacity(self):
        return self.capacity
    
    def store_memory(self, state, action, reward, terminal, next_state):
        # state_tensor = torch.tensor(np.stack(state, axis=0), device=self.device, dtype=torch.float32)
        # next_state_tensor = torch.tensor(np.stack(next_state, axis=0), device=self.device, dtype=torch.float32)

        # self.memory_states[self.memory_index] = state_tensor
        # self.memory_actions[self.memory_index] = action
        # self.memory_rewards[self.memory_index] = reward
        # self.memory_terminal[self.memory_index] = terminal
        # self.memory_next_states[self.memory_index] = next_state_tensor

        # self.memory_index = (self.memory_index + 1) % self.capacity
        
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
        # unique_indexes = random.sample(range(self.memory_fill_index), batch_size)                
        # random_memory_states = self.memory_states[unique_indexes]
        # random_memory_actions = self.memory_actions[unique_indexes]
        # random_memory_rewards = self.memory_rewards[unique_indexes]
        # random_memory_terminal = self.memory_terminal[unique_indexes]
        # random_memory_next_states = self.memory_next_states[unique_indexes] #cool trick with np arrays, give a list of numebrs
        
        # return random_memory_states, random_memory_actions, random_memory_rewards, random_memory_terminal, random_memory_next_states
        
        weighted_indices = np.arange(len(self.memory))[::-1]  # Reverse order to give more weight to recent experiences
        sampled_indices = random.choices(weighted_indices, k=batch_size)
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, terminals, next_states = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        terminals = torch.stack(terminals).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        return states, actions, rewards, terminals, next_states