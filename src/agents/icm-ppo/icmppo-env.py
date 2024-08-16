import argparse
from src.envs.crafter_env import create_env
from src.utils.logging import LossLoggingCallback
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import json
from icmppo import ICMPPO
from utils.utils import Memory

# Set up the seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_icmppo_agent(env, steps, log_dir):
    """
    Train an ICMPPO agent on the given environment for a specified number of steps.
    """
    memory = Memory()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = ICMPPO(writer=None, device=device)

    timestep = 0
    solved_reward = 1.7
    log_interval = 1000
    max_episodes = 350
    max_timesteps = 64 #1000
    update_timestep = 2048
    
    T = np.zeros(16)
    state = env.reset()
    # debug
    actions = agent.policy_old.act(np.array(state), memory)
    print(f"actions from policy_old: {actions}")
    for i_episode in range(0, max_episodes+1):
        episode_rewards = np.zeros(16)
        episode_counter = np.zeros(16)
        for i in range(max_timesteps):
            timestep += 1
            T += 1
            # Running policy_old:
            print(f"States from error:{np.array(state).shape}")
            actions = agent.policy_old.act(np.array(state), memory)
            actions = np.array(actions)
            print(f"actions: {actions}, lenght actions: {len(actions)}, index: {i}")
            state, rewards, dones, info = env.step(actions[i])
            print(f"actions[i] : {np.array(actions)[i]}")
            # Fix rewards
            dones = np.array(dones)
            print(f"dones: {dones}")
            rewards = np.array(rewards)
            print(f"rewards: {rewards}")
            if isinstance(rewards, np.ndarray) and rewards.size == 1:
                if rewards == 0 and T[0] < 1000:
                    rewards += 2
                else:
                   rewards += 2 * (rewards == 0) * (T < 1000)
            episode_counter += dones
            T[dones] = 0
            # Saving reward and is_terminal:
            memory.rewards.append(rewards)
            memory.is_terminals.append(dones)

            # Update if it's time
            if timestep % update_timestep == 0:
                agent.update(memory, timestep)
                memory.clear_memory()

            episode_rewards += rewards

        if episode_counter.sum() == 0:
            episode_counter = np.ones(16)

        # Stop training if avg_reward > solved_reward
        if episode_rewards.sum() / episode_counter.sum() > solved_reward:
            print("########## Solved! ##########")
            torch.save(agent.policy.state_dict(), './ppo.pt')
            torch.save(agent.icm.state_dict(), './icm.pt')
            break

        # Logging
        if timestep % log_interval == 0:
            print(f'Episode {i_episode} \t episode reward: {episode_rewards.sum() / episode_counter.sum()} \t')
    return agent

def render_env(env, agent, steps):
    """
    Render the environment using the trained ICMPPO model.
    """
    memory = Memory() 
    state = env.reset()
    cumulative_score = 0
    for _ in range(int(steps)):
        print("inside render_env")
        action = agent.policy_old.act(np.array(state), memory) #memory
        print(f"action: {action}")
        if isinstance(action, (list, np.ndarray)):
            action = action[0]
        state, reward, done, info = env.step(int(action))
        
        cumulative_score += reward
        env.render()
        if done:
            print(f"Episode Score: {cumulative_score}")
            cumulative_score = 0
            state = env.reset()
    env.close()

def main():
    # Argument parser to handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=float, default=1e6)
    args = parser.parse_args()

    seeds = [42]  # List of seeds

    for seed in seeds:
        print(f"\nRunning training with seed: {seed}")
        set_seed(seed)

        # Create the environment with Crafter
        env = create_env()

        # Train the ICMPPO agent
        agent = train_icmppo_agent(env, args.steps, log_dir='./logdir/icmppo')

        # Render the environment after training
        render_env(env, agent, args.steps)

if __name__ == "__main__":
    main()