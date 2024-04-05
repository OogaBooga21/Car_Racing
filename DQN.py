import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Network(nn.Module):
    linear_input_size = 0
    
    def __init__(self,frame_stack_size, img_height, img_width ,output_layer_size):
        super().__init__()
        self.frame_stack_size = frame_stack_size
        self.output_layer_size = output_layer_size
        
        self.convolutions = nn.Sequential(
            nn.Conv2d(self.frame_stack_size, out_channels=4, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )
        
        dummy_input = torch.zeros((1, self.frame_stack_size, img_height, img_width))
        dummy_output = self.convolutions(dummy_input)
        linear_input_size = dummy_output.view(1, -1).size(1)
        self.lstm = nn.LSTM(linear_input_size, hidden_size = 256, batch_first = True)
        
        self.dnn = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, self.output_layer_size)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00025) 
        self.scheduler = StepLR(self.optimizer, step_size=4000, gamma=0.5)  # Adjust parameters as needed
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        extracted_features = self.convolutions(state)
        if(len(extracted_features.shape)==4):
            extracted_features = extracted_features.view(extracted_features.size(0), -1)
        else:
            extracted_features = extracted_features.view(-1)
            
        lstm_output, _ = self.lstm(extracted_features.unsqueeze(0))
        lstm_output = lstm_output.squeeze(0)

        action_output = self.dnn(lstm_output)
        return action_output