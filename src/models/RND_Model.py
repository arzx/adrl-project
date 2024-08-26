import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
class RND_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RND_Model, self).__init__()   
        flatten = state_dim[0] * state_dim[1] * state_dim[2]
        self.nn_layer = nn.Sequential(
                nn.Linear(flatten, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.ReLU(),
                nn.Linear(action_dim, 1),
              ).float().to(device)
        
    def forward(self, states):
        states = states.reshape(states.size(0), -1)
        return self.nn_layer(states)