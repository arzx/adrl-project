import argparse
from stable_baselines3 import PPO
from src.envs.crafter_env import create_env
from src.utils.logging import LossLoggingCallback
import matplotlib.pyplot as plt

def train_ppo_agent(env, steps):
    """
    Train a PPO agent on the given environment for a specified number of steps.
    """
    callback = LossLoggingCallback(verbose=1)
    model = PPO('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=steps, callback=callback)
    return model, callback.get_losses(), callback.get_rewards()

def render_env(env, model, steps):
    """
    Render the environment using the trained model.
    """
    obs = env.reset()
    for _ in range(int(steps)):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()

def plot_losses(losses):
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
    plt.title('Average Loss during PPO Training')
    plt.grid(True)
    # Set y-axis limits based on the range of loss values
    plt.ylim(min(losses) * 0.9, max(losses) * 1.1)
    plt.show()

def plot_rewards(rewards):
    if not rewards:
        print("No rewards to plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.title('Average Rewards during PPO Training')
    plt.grid(True)
    plt.show()

def main():
    # Argument parser to handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=float, default=1e6)
    args = parser.parse_args()

    # Create the environment
    env = create_env()

    # Train the PPO agent
    model, losses, rewards = train_ppo_agent(env, args.steps)

    # Render the environment after training
    render_env(env, model, args.steps)

    # Plot the losses and rewards
    plot_losses(losses)
    plot_rewards(rewards)

if __name__ == "__main__":
    main()