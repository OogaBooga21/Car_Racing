import torch
import numpy as np
from torch import nn

class Network(nn.Module):
    input_layer_size = 0
    lr = 0
    
    def __init__(self,env):
        super().__init__()
        self.input_layer_size = int(np.prod(env.observation_space.shape))
        self.output_layer_size = int(env.action_space.n)
        self.lr = 0.001
        
        self.net = nn.Sequential(
            nn.Linear(self.input_layer_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_layer_size)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        
    def set_lr(self,lr):
        self.lr = lr
    
    def forward(self,state_s):
        state_tensor = torch.tensor(state_s, dtype=torch.float32) 
        # print(state_tensor.shape)
        return self.net(state_tensor)