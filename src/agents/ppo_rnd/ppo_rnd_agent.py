from envs.crafter_env import create_env
from models.Actor_Model import Actor_Model
from models.Critic_Model import Critic_Model
from models.RND_Model import RND_Model
from src.agents.ppo_rnd.memory import Memory
from src.agents.ppo_rnd.obs_memory import ObsMemory
import torch
from utils.distributions import Distributions
from torch.utils.data import DataLoader
from utils.policy_function import PolicyFunction
from torch.optim import Adam
import warnings
from utils.utils import Utils
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be .* leaked semaphore objects")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Agent():  
    def __init__(self, state_dim, action_dim, is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                 minibatch, PPO_epochs, gamma, lam, learning_rate):        
        self.policy_kl_range        = policy_kl_range 
        self.policy_params          = policy_params
        self.value_clip             = value_clip    
        self.entropy_coef           = entropy_coef
        self.vf_loss_coef           = vf_loss_coef
        self.minibatch              = minibatch       
        self.PPO_epochs             = PPO_epochs
        self.RND_epochs             = 5
        self.is_training_mode       = is_training_mode
        self.action_dim             = action_dim               

        self.actor                  = Actor_Model(state_dim, action_dim)
        self.actor_old              = Actor_Model(state_dim, action_dim)
        self.actor_optimizer        = Adam(self.actor.parameters(), lr = learning_rate)

        self.ex_critic              = Critic_Model(state_dim, action_dim)
        self.ex_critic_old          = Critic_Model(state_dim, action_dim)
        self.ex_critic_optimizer    = Adam(self.ex_critic.parameters(), lr = learning_rate)

        self.in_critic              = Critic_Model(state_dim, action_dim)
        self.in_critic_old          = Critic_Model(state_dim, action_dim)
        self.in_critic_optimizer    = Adam(self.in_critic.parameters(), lr = learning_rate)

        self.rnd_predict            = RND_Model(state_dim, action_dim)
        self.rnd_predict_optimizer  = Adam(self.rnd_predict.parameters(), lr = learning_rate)
        self.rnd_target             = RND_Model(state_dim, action_dim)

        self.memory                 = Memory()
        self.obs_memory             = ObsMemory(state_dim)

        self.policy_function        = PolicyFunction(gamma, lam)  
        self.distributions          = Distributions()
        self.utils                  = Utils()

        self.ex_advantages_coef     = 2
        self.in_advantages_coef     = 1        
        self.clip_normalization     = 5

        if is_training_mode:
          self.actor.train()
          self.ex_critic.train()
          self.in_critic.train()
        else:
          self.actor.eval()
          self.ex_critic.eval()
          self.in_critic.eval()

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)

    def save_observation(self, obs):
        self.obs_memory.save_eps(obs)

    def update_obs_normalization_param(self, obs):
        obs = np.array(obs)
        obs                 = torch.FloatTensor(obs).to(device).detach()

        mean_obs            = self.utils.count_new_mean(self.obs_memory.mean_obs, self.obs_memory.total_number_obs, obs)
        std_obs             = self.utils.count_new_std(self.obs_memory.std_obs, self.obs_memory.total_number_obs, obs)
        total_number_obs    = len(obs) + self.obs_memory.total_number_obs
        
        self.obs_memory.save_observation_normalize_parameter(mean_obs, std_obs, total_number_obs)
    
    def update_rwd_normalization_param(self, in_rewards):
        std_in_rewards      = self.utils.count_new_std(self.obs_memory.std_in_rewards, self.obs_memory.total_number_rwd, in_rewards)
        total_number_rwd    = len(in_rewards) + self.obs_memory.total_number_rwd
        
        self.obs_memory.save_rewards_normalize_parameter(std_in_rewards, total_number_rwd)    

    # Loss for RND 
    def get_rnd_loss(self, state_pred, state_target):        
        # Don't update target state value
        state_target = state_target.detach()        
        
        # Mean Squared Error Calculation between state and predict
        forward_loss = ((state_target - state_pred).pow(2) * 0.5).mean()
        return forward_loss

    # Loss for PPO  
    def get_PPO_loss(self, action_probs, ex_values, old_action_probs, old_ex_values, next_ex_values, actions, ex_rewards, dones, 
        state_preds, state_targets, in_values, old_in_values, next_in_values, std_in_rewards):
        
        # Apply softmax to logits if not already done and ensure they are probabilities
        action_probs = torch.softmax(action_probs, dim=-1)
        action_probs = torch.clamp(action_probs, min=1e-8)  # Avoid probabilities being exactly zero
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)  # Ensure they sum to 1

        # Apply the same process to old_action_probs
        old_action_probs = torch.softmax(old_action_probs, dim=-1)
        old_action_probs = torch.clamp(old_action_probs, min=1e-8)
        old_action_probs = old_action_probs / old_action_probs.sum(dim=-1, keepdim=True)
        
        # Don't use old value in backpropagation
        Old_ex_values = old_ex_values.detach()

        # Getting external general advantages estimator
        External_Advantages = self.policy_function.generalized_advantage_estimation(ex_values, ex_rewards, next_ex_values, dones)
        External_Returns = (External_Advantages + ex_values).detach()
        External_Advantages = self.utils.normalize(External_Advantages).detach()

        # Computing internal reward, then getting internal general advantages estimator
        in_rewards = (state_targets - state_preds).pow(2) * 0.5 / (std_in_rewards.mean() + 1e-8)
        Internal_Advantages = self.policy_function.generalized_advantage_estimation(in_values, in_rewards, next_in_values, dones)
        Internal_Returns = (Internal_Advantages + in_values).detach()
        Internal_Advantages = self.utils.normalize(Internal_Advantages).detach()      

        # Getting overall advantages
        Advantages = (self.ex_advantages_coef * External_Advantages + self.in_advantages_coef * Internal_Advantages).detach()

        # Finding the ratio (pi_theta / pi_theta__old)
        logprobs = self.distributions.logprob(action_probs, actions)
        Old_logprobs = self.distributions.logprob(old_action_probs, actions).detach()
        ratios = (logprobs - Old_logprobs).exp()  # ratios = exp(log(new) - log(old))

        # Finding KL Divergence                
        Kl = self.distributions.kl_divergence(old_action_probs, action_probs)

        # Combining TR-PPO with Rollback (Truly PPO)
        pg_loss = torch.where(
                (Kl >= self.policy_kl_range) & (ratios > 1),
                ratios * Advantages - self.policy_params * Kl,
                ratios * Advantages
        )
        pg_loss = pg_loss.mean()

        # Getting entropy from the action probability 
        dist_entropy = self.distributions.entropy(action_probs).mean()

        # Getting critic loss by using Clipped critic value
        ex_vpredclipped = Old_ex_values + torch.clamp(ex_values - Old_ex_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        ex_vf_losses1 = (External_Returns - ex_values).pow(2) # Mean Squared Error
        ex_vf_losses2 = (External_Returns - ex_vpredclipped).pow(2) # Mean Squared Error
        critic_ext_loss = torch.max(ex_vf_losses1, ex_vf_losses2).mean()      

        # Getting Intrinsic critic loss
        critic_int_loss = (Internal_Returns - in_values).pow(2).mean() 

        # Getting overall critic loss
        critic_loss = (critic_ext_loss + critic_int_loss) * 0.5

        # Final PPO loss
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss       
    

    def act(self, state):
        state           = torch.FloatTensor(state).unsqueeze(0).to(device).detach()
        logits = self.actor(state)
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        logits = torch.clamp(logits, min=-100, max=100)
        #normalized_logits = get_normalized_logits(logits)
        action_probs = torch.softmax(logits, dim=-1)
        # clamping probs to disable negative and near zero probs
        action_probs = torch.clamp(action_probs, min=1e-8)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            action  = self.distributions.sample(action_probs) 
        else:
            action  = torch.argmax(action_probs, 1)  
        print(f"Selected action: {action.cpu().item()}")
        return action.cpu().item()

    def compute_intrinsic_reward(self, obs, mean_obs, std_obs):
        obs             = self.utils.normalize(obs, mean_obs, std_obs)
        
        state_pred      = self.rnd_predict(obs)
        state_target    = self.rnd_target(obs)

        return (state_target - state_pred)

    # Get loss and Do backpropagation
    def training_rnd(self, obs, mean_obs, std_obs):
        obs             = self.utils.normalize(obs, mean_obs, std_obs)
        
        state_pred      = self.rnd_predict(obs)
        state_target    = self.rnd_target(obs)

        loss            = self.get_rnd_loss(state_pred, state_target)

        self.rnd_predict_optimizer.zero_grad()
        loss.backward()
        self.rnd_predict_optimizer.step()
    env = create_env()
    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states, mean_obs, std_obs, std_in_rewards):
        # Don't update rnd value
        obs             = self.utils.normalize(next_states, mean_obs, std_obs, self.clip_normalization).detach()
        state_preds     = self.rnd_predict(obs)
        state_targets   = self.rnd_target(obs)

        action_probs, ex_values, in_values                  = self.actor(states), self.ex_critic(states),  self.in_critic(states)
        old_action_probs, old_ex_values, old_in_values      = self.actor_old(states), self.ex_critic_old(states),  self.in_critic_old(states)
        next_ex_values, next_in_values                      = self.ex_critic(next_states),  self.in_critic(next_states)       
        loss            = self.get_PPO_loss(action_probs, ex_values, old_action_probs, old_ex_values, next_ex_values, actions, rewards, dones,
                            state_preds, state_targets, in_values, old_in_values, next_in_values, std_in_rewards)

        self.actor_optimizer.zero_grad()
        self.ex_critic_optimizer.zero_grad()
        self.in_critic_optimizer.zero_grad()

        loss.backward()

        self.actor_optimizer.step() 
        self.ex_critic_optimizer.step() 
        self.in_critic_optimizer.step() 

    # Update the model
    def update_rnd(self):        
        batch_size  = int(len(self.obs_memory) / self.minibatch)
        dataloader  = DataLoader(self.obs_memory, batch_size, shuffle = False, num_workers=0)        

        # Optimize policy for K epochs:
        for _ in range(self.RND_epochs):       
            for obs in dataloader:
                self.training_rnd(obs.float().to(device), self.obs_memory.mean_obs.float().to(device), self.obs_memory.std_obs.float().to(device))       

        intrinsic_rewards = self.compute_intrinsic_reward(self.obs_memory.get_all().to(device), self.obs_memory.mean_obs.to(device), self.obs_memory.std_obs.to(device))
        # only for debugging, delete later
        #print("Intrinsic Rewards for the current batch:", intrinsic_rewards.detach().cpu().numpy())

        self.update_obs_normalization_param(self.obs_memory.observations)
        self.update_rwd_normalization_param(intrinsic_rewards)

        # Clear the memory
        self.obs_memory.clear_memory()

    # Update the model
    def update_ppo(self):        
        batch_size  = int(len(self.memory) / self.minibatch)
        dataloader  = DataLoader(self.memory, batch_size, shuffle = False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in dataloader:
                self.training_ppo(states.float().to(device), actions.float().to(device), rewards.float().to(device), dones.float().to(device), next_states.float().to(device),
                    self.obs_memory.mean_obs.float().to(device), self.obs_memory.std_obs.float().to(device), self.obs_memory.std_in_rewards.float().to(device))

        # Clear the memory
        self.memory.clear_memory()

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.ex_critic_old.load_state_dict(self.ex_critic.state_dict())
        self.in_critic_old.load_state_dict(self.in_critic.state_dict())

    def save_weights(self):
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
            }, '/test/My Drive/Bipedal4/actor.tar')
        
        torch.save({
            'model_state_dict': self.ex_critic.state_dict(),
            'optimizer_state_dict': self.ex_critic_optimizer.state_dict()
            }, '/test/My Drive/Bipedal4/ex_critic.tar')

        torch.save({
            'model_state_dict': self.in_critic.state_dict(),
            'optimizer_state_dict': self.in_critic_optimizer.state_dict()
            }, '/test/My Drive/Bipedal4/in_critic.tar')
        
    def load_weights(self):
        actor_checkpoint = torch.load('/test/My Drive/Bipedal4/actor.tar')
        self.actor.load_state_dict(actor_checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])

        ex_critic_checkpoint = torch.load('/test/My Drive/Bipedal4/ex_critic.tar')
        self.ex_critic.load_state_dict(ex_critic_checkpoint['model_state_dict'])
        self.ex_critic_optimizer.load_state_dict(ex_critic_checkpoint['optimizer_state_dict'])

        in_critic_checkpoint = torch.load('/test/My Drive/Bipedal4/in_critic.tar')
        self.in_critic.load_state_dict(in_critic_checkpoint['model_state_dict'])
        self.in_critic_optimizer.load_state_dict(in_critic_checkpoint['optimizer_state_dict'])