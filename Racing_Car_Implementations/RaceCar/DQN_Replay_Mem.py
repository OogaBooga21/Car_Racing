import numpy as np
import torch
import random

class Replay_Memory:
    memory_fill_index = 0
    
    def __init__(self,capacity,stack_size,img_height,img_width):
        self.capacity = capacity
        self.memory_index = 0
        self.memory_fill_index = 0
        
        # self.memory_states = np.empty((capacity, stack_size ,img_height, img_width), dtype=np.float32)
        # self.memory_actions = np.empty(capacity, dtype=np.int64)
        # self.memory_rewards = np.zeros(capacity, dtype=np.float32)
        # self.memory_terminal = np.empty(capacity, dtype=bool)
        # self.memory_next_states = np.empty((capacity, stack_size, img_height, img_width), dtype=np.float32)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_states = torch.zeros((capacity, stack_size, img_height, img_width), dtype=torch.float32, device=self.device)
        self.memory_actions = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        self.memory_rewards = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.memory_terminal = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        self.memory_next_states = torch.zeros((capacity, stack_size, img_height, img_width), dtype=torch.float32, device=self.device)
        #making up the touple (s,a,r,s') + to know if we're done for the other q branch
        
    def fill_rate(self):
        return self.memory_fill_index
    
    def get_capacity(self):
        return self.capacity
    
    def store_memory(self, state, action, reward, terminal, next_state):
        state_tensor = torch.tensor(np.stack(state, axis=0), device=self.device, dtype=torch.float32)
        next_state_tensor = torch.tensor(np.stack(next_state, axis=0), device=self.device, dtype=torch.float32)

        self.memory_states[self.memory_index] = state_tensor
        self.memory_actions[self.memory_index] = action
        self.memory_rewards[self.memory_index] = reward
        self.memory_terminal[self.memory_index] = terminal
        self.memory_next_states[self.memory_index] = next_state_tensor

        self.memory_index = (self.memory_index + 1) % self.capacity
        self.memory_fill_index = min(self.memory_fill_index + 1, self.capacity)
        
    def random_memory_batch(self, batch_size):
        unique_indexes = random.sample(range(self.memory_fill_index), batch_size)                
        random_memory_states = self.memory_states[unique_indexes]
        random_memory_actions = self.memory_actions[unique_indexes]
        random_memory_rewards = self.memory_rewards[unique_indexes]
        random_memory_terminal = self.memory_terminal[unique_indexes]
        random_memory_next_states = self.memory_next_states[unique_indexes] #cool trick with np arrays, give a list of numebrs
        
        return random_memory_states, random_memory_actions, random_memory_rewards, random_memory_terminal, random_memory_next_states
        #CONVERT TO DEVICE IF NEEDED