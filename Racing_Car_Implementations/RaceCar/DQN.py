import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class Network(nn.Module):
    linear_input_size = 0
    
    def __init__(self,frame_stack_size, img_height, img_width ,output_layer_size):
        super().__init__()
        self.frame_stack_size = frame_stack_size
        self.output_layer_size = output_layer_size
        self.convolutions = nn.Sequential(
            nn.Conv2d(self.frame_stack_size, out_channels=32, kernel_size=7,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        dummy_input = torch.zeros((1, self.frame_stack_size, img_height, img_width))
        dummy_output = self.convolutions(dummy_input)
        linear_input_size = dummy_output.view(1, -1).size(1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4,)
        
        self.dnn = nn.Sequential(
            nn.Linear(linear_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_layer_size)
        )      
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        extracted_features = self.convolutions(state)
        
        if(len(extracted_features.shape) == 4):
            extracted_features = extracted_features.view(extracted_features.size(0),-1)
        else:
            extracted_features = extracted_features.view(-1)
        
        action_output = self.dnn(extracted_features)
        return action_output