import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

class Network(nn.Module):
    linear_input_size = 0
    
    def __init__(self,frame_stack_size, img_height, img_width ,output_layer_size):
        super().__init__()
        self.output_layer_size = output_layer_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.convolutions = nn.Sequential(
            nn.Conv2d(frame_stack_size, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),  # Reduce to 12x12
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce to 6x6
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros((1,frame_stack_size, img_height, img_width))
            dummy_output = self.convolutions(dummy_input)
            linear_input_size = np.prod(dummy_output.size()[1:])
        
        self.lstm = nn.LSTM(linear_input_size, hidden_size = 512, batch_first = True)
        
        self.dnn = nn.Sequential(
            nn.Linear(linear_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128, self.output_layer_size)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        self.scheduler = StepLR(self.optimizer, step_size=4000, gamma=0.5)
        

    def forward(self, state):
        batch_size, frame_stack_size, height, width = state.size()
        #state = state.view(batch_size * frame_stack_size, 1, height,width)
        
        conv_out = self.convolutions(state)
        conv_out = conv_out.view(batch_size, -1)
        # conv_out = conv_out.view(batch_size, frame_stack_size, -1)
        
        #lstm_output, _ = self.lstm(conv_out)
        #lstm_output = lstm_output[:, -1, :]
    
        action_output = self.dnn(conv_out)
        
        return action_output