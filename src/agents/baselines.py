# baselines.py
import argparse
from stable_baselines3 import PPO
from src.envs.crafter_module_env import create_env

def train_ppo_agent(env, steps):
    """
    Train a PPO agent on the given environment for a specified number of steps.
    """
    model = PPO('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=steps)
    return model

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

def main():
    # Argument parser to handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=float, default=1e6)
    args = parser.parse_args()

    # Create the environment
    env = create_env()

    # Train the PPO agent
    model = train_ppo_agent(env, args.steps)

    # Render the environment after training
    render_env(env, model, args.steps)

if __name__ == "__main__":
    main()