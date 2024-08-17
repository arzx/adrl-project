import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, activation=nn.Tanh(), device='cpu'):
        super(ActorCritic, self).__init__()
        self.device = device
        self.flattened_input_layer = nn.Sequential(
            nn.Linear(192, n_latent_var),
            activation
        ).to(self.device)

        self.flattened_state_layer = nn.Sequential(
            nn.Linear(22016, n_latent_var),
            activation
        ).to(self.device)

        self.body = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            activation
        ).to(self.device)

        # Actor head
        self.action_layer = nn.Sequential(
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        # Critic head
        self.value_layer = nn.Sequential(
            nn.Linear(172, n_latent_var),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(self.device)

        self.value_layer2 = nn.Sequential(
            nn.Linear(512, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, 1)
        ).to(self.device)

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(self.device)
        state = state.reshape(state.size(0), -1)  # Flattening the state to [batch_size, flattened_size]

        flattened_state = self.flattened_input_layer(state)
        body_output = self.body(flattened_state)

        action_probs = self.action_layer(body_output)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action.cpu().numpy()

    def evaluate(self, state, action):
        state = state.to(self.device).reshape(state.size(0), -1)  # Flattening the state to [batch_size, flattened_size]
        flattened_state = self.flattened_state_layer(state)
        body_output = self.body(flattened_state)

        action_probs = self.action_layer(body_output)
        dist = Categorical(action_probs)

        # If action has extra dimensions, reduce it
        if action.dim() > 1:
            action = action[:, 0]  # or apply another reduction method

        # Ensure action indices are within the valid range
        action = torch.clamp(action, 0, action_probs.size(-1) - 1)

        # Debug: Print out actions and their valid range
        print(f"action: {action}")
        print(f"Valid range: 0 to {action_probs.size(-1) - 1}")

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer2(body_output)

        return action_logprobs, torch.squeeze(state_value), dist_entropy