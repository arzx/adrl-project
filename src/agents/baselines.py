import argparse
from stable_baselines3 import PPO
from src.envs.crafter_env import create_env
from src.utils.logging import LossLoggingCallback
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import json

def set_seed(seed):
    """
    Set the seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_ppo_agent(env, steps):
    """
    Train a PPO agent on the given environment for a specified number of steps.
    """
    callback = LossLoggingCallback(total_timesteps=steps, verbose=1)
    model = PPO('CnnPolicy', env, verbose=1)
    
    scores = []
    all_info = []
    obs = env.reset()
    cumulative_score = 0

    while True:
        model.learn(total_timesteps=steps, callback=callback)  # Train the model for the given number of steps
        obs = env.reset()  # Reset the environment after training
        
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            all_info.append(info)
            cumulative_score += reward
            #print(cumulative_score)

            if done:
                scores.append(cumulative_score)
                print(scores)
                print(len(scores))
                cumulative_score = 0  # Reset score after each episode
                obs = env.reset()

        if len(scores) >= steps:
            break  # Stop if we've collected enough steps

    #all_info = [dict((k, v.tolist() if isinstance(v, np.ndarray) else v) for k, v in i.items()) for i in all_info]
    with open(f"logdir/crafter_reward-ppo/0/scores.json", "w") as file:
        json.dump(scores, file, indent=4)
    print("------ check if states work --------")
    return model, callback.get_losses(), callback.get_rewards(), scores

def render_env(env, model, steps):
    """
    Render the environment using the trained model.
    """
    obs = env.reset()
    cumulative_score = 0
    for _ in range(int(steps)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        cumulative_score += reward
        print(_states)
        env.render()
        if done:
            print(f"Episode Score: {cumulative_score}")
            cumulative_score = 0
            #print(f"achievements: {env.achievements}")
            obs = env.reset()
    env.close()

def plot_losses(losses, seed):
    """
    Plot the average loss over training steps.
    """
    if not losses:
        print("No losses to plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Average Loss during PPO Training (Seed: {seed})')
    plt.grid(True)
    plt.ylim(min(losses) * 0.9, max(losses) * 1.1)
    plt.show()

def plot_rewards(rewards, seed):
    if not rewards:
        print("No rewards to plot.")
        return
    #plt.xscale("log")
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.title(f'Average Rewards during PPO Training (Seed: {seed})')
    plt.grid(True)
    plt.show()

def plot_scores(scores, seed):
    if not scores:
        print("No scores to plot.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title(f'Scores during PPO Training (Seed: {seed})')
    plt.grid(True)
    plt.show()


def main():
    # Argument parser to handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=float, default=1e6)
    args = parser.parse_args()

    seeds = [42]  # List of seeds

    for seed in seeds:
        print(f"\nRunning training with seed: {seed}")
        set_seed(seed)

        # Create the environment
        env = create_env()

        # Train the PPO agent
        model, losses, rewards, scores = train_ppo_agent(env, args.steps)

        # Render the environment after training
        render_env(env, model, args.steps)

        # Plot the losses, rewards, and scores
        plot_losses(losses, seed)
        plot_rewards(rewards, seed)
        plot_scores(scores, seed)

if __name__ == "__main__":
    main()