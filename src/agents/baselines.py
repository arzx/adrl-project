import argparse
import json
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.envs.crafter_env import create_env
from src.utils.custom_logging import LossLoggingCallback
from stable_baselines3 import PPO
import logging
from skimage.transform import resize

def set_seed(seed):
    """
    Set the seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logger(seed):
    log_filename = f'logdir/crafter_reward-ppo/training_log_seed_{seed}.log'
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def train_ppo_agent(env, steps, seed):
    """
    Train a PPO agent on the given environment for a specified number of steps.
    """
    callback = LossLoggingCallback(total_timesteps=steps, verbose=1)
    model = PPO('CnnPolicy', env, verbose=1)
    
    scores = []
    action_probs_log = []  # List to store action probabilities
    obs = env.reset()
    cumulative_score = 0
    episodes = 0
    
    while True:
        model.learn(total_timesteps=steps, callback=callback)  # Train the model for the given number of steps
        obs = env.reset()  # Reset the environment after training
        
        done = False
        while not done:
            # Get action and action probabilities from the model
            action, _states = model.predict(obs, deterministic=False)
            
            # Extract action probabilities from the policy network
            obs_tensor = torch.tensor(obs).float().permute(2, 0, 1).unsqueeze(0)  # [batch_size, channels, height, width]
            with torch.no_grad():
                action_distribution = model.policy.get_distribution(obs_tensor)
                action_probabilities = action_distribution.distribution.probs.cpu().numpy()
            
            # Save action probabilities for each step
            action_probs_log.append(action_probabilities.tolist())
            
            print(f"Action Probabilities: {action_probabilities}")  # Print the action probabilities
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            cumulative_score += reward
            if done:
                scores.append(cumulative_score)
                print(scores)
                cumulative_score = 0  # Reset score after each episode
                obs = env.reset()
                print("Observation shape after reset:", obs.shape)  # Debugging shape
                episodes += 1

        if episodes >= steps:
            break  # Stop if we've completed enough episodes

    # Ensure the directory exists
    os.makedirs("logdir/crafter_reward-ppo/1", exist_ok=True)
    print(f"episodes: {episodes}")

    # Load existing scores from the JSON file, if it exists
    file_path = "logdir/crafter_reward-ppo/1/scores.json"
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
    existing_data[f"{seed}"] = scores

    # Save the updated scores to the JSON file
    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4)

    # Save action probabilities to a separate JSON file
    action_probs_file = f"logdir/crafter_reward-ppo/1/action_probs_seed_{seed}.json"
    with open(action_probs_file, "w") as file:
        json.dump(action_probs_log, file, indent=4)

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
        env.render()
        if done:
            print(f"Episode Score: {cumulative_score}")
            cumulative_score = 0
            obs = env.reset()
    env.close()

def main():
    # Argument parser to handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=float, default=1e6)
    args = parser.parse_args()

    seeds = [99, 101, 200, 301, 400]  # List of seeds

    for seed in seeds:
        setup_logger(seed) 
        print(f"\nRunning training with seed: {seed}")
        set_seed(seed)

        # Create the environment with crafter
        env = create_env()

        # Train the PPO agent
        model, losses, rewards, scores = train_ppo_agent(env, args.steps, seed)

        # Render the environment after training
        render_env(env, model, args.steps)

if __name__ == "__main__":
    main()
