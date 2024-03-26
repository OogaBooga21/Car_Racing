import numpy as np
import random

class Replay_Memory:
    memory_fill_index = 0
    
    def __init__(self,capacity,stack_size,img_height,img_width):
        
        self.capacity = capacity
        self.memory_index = 0
        self.memory_fill_index = 0
        
        self.memory_states = np.empty((capacity, stack_size ,img_height, img_width), dtype=np.float32)
        self.memory_actions = np.empty(capacity, dtype=np.int64)
        self.memory_rewards = np.zeros(capacity, dtype=np.float32)
        self.memory_terminal = np.empty(capacity, dtype=bool)
        self.memory_next_states = np.empty((capacity, stack_size, img_height, img_width), dtype=np.float32)
        #making up the touple (s,a,r,s') + to know if we're done for the other q branch
        
    def fill_rate(self):
        return self.memory_fill_index
    
    def get_capacity(self):
        return self.capacity
    
    def store_memory(self, state, action, reward, terminal, next_state):
        # print(state[0].shape,state[1].shape,state[2].shape,state[3].shape)
        self.memory_states[self.memory_index] = state
        self.memory_actions[self.memory_index] = action
        self.memory_rewards[self.memory_index] = reward
        self.memory_terminal[self.memory_index] = terminal
        self.memory_next_states[self.memory_index] = next_state
        
        self.memory_index = (self.memory_index + 1) % self.capacity # rolling memory
        
        if self.memory_fill_index < self.capacity: #cvnty way of avoiding append
            self.memory_fill_index += 1
        
    def random_memory_batch(self, batch_size):
        unique_indexes = random.sample(range(self.memory_fill_index), batch_size)
        
        random_memory_states = np.empty([], dtype=np.float32)
        random_memory_actions = np.empty([], dtype=np.int64)
        random_memory_rewards = np.empty([], dtype=object)
        random_memory_terminal = np.empty([], dtype=np.float32)
        random_memory_next_states = np.empty([], dtype=np.float32)
        
        random_memory_states = self.memory_states[unique_indexes]
        random_memory_actions = self.memory_actions[unique_indexes]
        random_memory_rewards = self.memory_rewards[unique_indexes]
        random_memory_terminal = self.memory_terminal[unique_indexes]
        random_memory_next_states = self.memory_next_states[unique_indexes] #cool trick with np arrays, give a list of numebrs
        
        return random_memory_states, random_memory_actions, random_memory_rewards, random_memory_terminal, random_memory_next_states
        #CONVERT TO DEVICE IF NEEDED