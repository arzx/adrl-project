import torch
from torch.utils.data import Dataset
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be .* leaked semaphore objects")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class ObsMemory(Dataset):
    def __init__(self, state_dim):
        self.observations    = []

        self.mean_obs           = torch.zeros(state_dim).to(device)
        self.std_obs            = torch.zeros(state_dim).to(device)
        self.std_in_rewards     = torch.zeros(1).to(device)
        self.total_number_obs   = torch.zeros(1).to(device)
        self.total_number_rwd   = torch.zeros(1).to(device)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return np.array(self.observations[idx], dtype = np.float32)

    def get_all(self):
        return torch.FloatTensor(self.observations)

    def save_eps(self, obs):
        self.observations.append(obs)

    def save_observation_normalize_parameter(self, mean_obs, std_obs, total_number_obs):
        self.mean_obs           = mean_obs
        self.std_obs            = std_obs
        self.total_number_obs   = total_number_obs
        
    def save_rewards_normalize_parameter(self, std_in_rewards, total_number_rwd):
        self.std_in_rewards     = std_in_rewards
        self.total_number_rwd   = total_number_rwd

    def clear_memory(self):
        del self.observations[:]