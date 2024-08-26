from src.envs.crafter_env import create_env
from src.agents.ppo_rnd.ppo_rnd_agent import Agent
import torch
import random
import warnings
import matplotlib.pyplot as plt
import json
import os
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be .* leaked semaphore objects")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def set_seed(seed):
    """
    Set the seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot(datas):
    print('----------')

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))

def run_inits_episode(env, agent, state_dim, render, n_init_episode):
    ############################################
    env.reset()

    for _ in range(n_init_episode):
        action                  = env.action_space.sample()
        next_state, _, done, _  = env.step(action)
        agent.save_observation(next_state)

        if render:
            env.render()

        if done:
            env.reset()

    agent.update_obs_normalization_param(agent.obs_memory.observations)
    agent.obs_memory.clear_memory()

    return agent

def run_episode(env, agent, state_dim, render, training_mode, t_updates, n_update, seed):
    ############################################
    state           = env.reset()
    done            = False
    total_reward    = 0
    eps_time        = 0
    ############################################
    
    while not done:
        action                      = int(agent.act(state))
        next_state, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}") 
        eps_time        += 1 
        t_updates       += 1
        total_reward    += reward

        if training_mode:
            agent.save_eps(state.tolist(), float(action), float(reward), float(done), next_state.tolist())
            agent.save_observation(next_state)
        # test how intrinsic reward is calculated, delete later
        intrinsic_reward = agent.compute_intrinsic_reward(
                torch.FloatTensor(next_state).unsqueeze(0).to(device), 
                agent.obs_memory.mean_obs, 
                agent.obs_memory.std_obs
            )
        #print("Intrinsic Reward for the current step:", intrinsic_reward.detach().cpu().numpy())

        state   = next_state
                
        if render:
            env.render()
        
        if training_mode:
            if t_updates % n_update == 0:
                agent.update_rnd()
                t_updates = 0
        
        if done:           
            # Directory creation and JSON handling
            os.makedirs("logdir/ppo-rnd/0/", exist_ok=True)
            print(f"episodes: {eps_time}")
            
            # Load existing scores from the JSON file, if it exists
            file_path = "logdir/ppo-rnd/0/scores_ppornd.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    try:
                        existing_data = json.load(file)
                        if not isinstance(existing_data, dict):
                            existing_data = {}
                    except json.JSONDecodeError:
                        existing_data = {}
            else:
                existing_data = {}

            # Add the new scores under the current seed
            existing_data[f"{seed}"] = total_reward

            # Save the updated scores to the JSON file
            with open(file_path, "w") as file:
                json.dump(existing_data, file, indent=4)

            return total_reward, eps_time, t_updates

def main():
    ############## Hyperparameters ##############
    seeds = [42, 111, 101, 292, 300]  # List of seeds
    load_weights        = False # If you want to load the agent, set this to True
    save_weights        = False # If you want to save the agent, set this to True
    training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold    = 300 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

    render              = False # If you want to display the image, set this to True. Turn this off if you run this in Google Collab
    n_step_update       = 128 # How many steps before you update the RND. Recommended set to 128 for Discrete
    n_eps_update        = 5 # How many episode before you update the PPO. Recommended set to 5 for Discrete
    n_plot_batch        = 100000000 # How many episode you want to plot the result
    n_episode           = 1000000 # change to 1M
    n_init_episode      = 1024
    n_saved             = 10 # How many episode to run before saving the weights

    policy_kl_range     = 0.0008 # Recommended set to 0.0008 for Discrete
    policy_params       = 20 # Recommended set to 20 for Discrete
    value_clip          = 1.0 # How many value will be clipped. Recommended set to the highest or lowest possible reward
    entropy_coef        = 0.05 # How much randomness of action you will get
    vf_loss_coef        = 1.0 # Just set to 1
    minibatch           = 1 # How many batch per update. size of batch = n_update / minibatch. Recommended set to 4 for Discrete
    PPO_epochs          = 1 # How many epoch per update. Recommended set to 10 for Discrete
    
    gamma               = 0.99 # Just set to 0.99
    lam                 = 0.95 # Just set to 0.95
    learning_rate       = 0.00003 # Just set to 0.95
    ############################################# 
    env                 = create_env()
    #env                = gym.make('CartPole-v1')
    state_dim           = env.observation_space.shape
    action_dim          = env.action_space.n

    agent               = Agent(state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                            minibatch, PPO_epochs, gamma, lam, learning_rate)  
    #############################################   

    if load_weights:
        agent.load_weights()
        print('Weight Loaded')

    rewards             = []   
    batch_rewards       = []
    batch_solved_reward = []

    times               = []
    batch_times         = []

    t_updates           = 0

    #############################################

    if training_mode:
        agent = run_inits_episode(env, agent, state_dim, render, n_init_episode)

    #############################################

    for i_episode in range(1, n_episode + 1):
        total_reward, time, t_updates = run_episode(env, agent, state_dim, render, training_mode, t_updates, n_step_update, seeds)
        print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, total_reward, time))
        batch_rewards.append(int(total_reward))
        batch_times.append(time)

        if i_episode % n_eps_update == 0:
            agent.update_ppo()         

        if save_weights:
            if i_episode % n_saved == 0:
                agent.save_weights() 
                print('weights saved')

        if reward_threshold:
            if len(batch_solved_reward) == 100:            
                if np.mean(batch_solved_reward) >= reward_threshold:
                    print('You solved task after {} episode'.format(len(rewards)))
                    break

                else:
                    del batch_solved_reward[0]
                    batch_solved_reward.append(total_reward)

            else:
                batch_solved_reward.append(total_reward)

        if i_episode % n_plot_batch == 0 and i_episode != 0:
            # Plot the reward, times for every n_plot_batch
            plot(batch_rewards)
            plot(batch_times)

            for reward in batch_rewards:
                rewards.append(reward)

            for time in batch_times:
                times.append(time)

            batch_rewards   = []
            batch_times     = []

            print('========== Cummulative ==========')
            # Plot the reward, times for every episode
            plot(rewards)
            plot(times)

    print('========== Final ==========')
    # Plot the reward, times for every episode

    for reward in batch_rewards:
        rewards.append(reward)

    for time in batch_times:
        times.append(time)

    plot(rewards)
    plot(times)

if __name__ == '__main__':
    main()