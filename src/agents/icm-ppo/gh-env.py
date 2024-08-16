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


render = False
solved_reward = 1.7     # stop training if avg_reward > solved_reward
log_interval = 1000     # print avg reward in the interval
max_episodes = 350      # max training episodes
max_timesteps = 1000    # max timesteps in one episode
update_timestep = 2048  # Replay buffer size, update policy every n timesteps
log_dir= './'           # Where to store tensorboard logs

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
crafter = create_env()


# Initialize log_writer, memory buffer, icmppo
memory = Memory()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = ICMPPO(writer=None, device=device)

timestep = 0
T = np.zeros(16)
state = crafter.reset()
# training loop
for i_episode in range(1, max_episodes + 1):
    episode_rewards = np.zeros(16)
    episode_counter = np.zeros(16)
    for i in range(max_timesteps):
        timestep += 1
        T += 1
        # Running policy_old:
        actions = agent.policy_old.act(np.array(state), memory)
        if isinstance(actions, (list, np.ndarray)):
            actions = actions[0]
        state, rewards, dones, info = crafter.step(int(actions))

        # Fix rewards
        dones = np.array(dones)
        rewards = np.array(rewards)
        if rewards.size == 1:
            rewards = np.full_like(T, rewards)
        rewards += 2 * (rewards == 0) * (T < 1000)
        episode_counter += dones
        T[dones] = 0
        # Saving reward and is_terminal:
        memory.rewards.append(rewards)
        memory.is_terminals.append(dones)

        # update if its time
        if timestep % update_timestep == 0:
            agent.update(memory, timestep)
            memory.clear_memory()

        episode_rewards += rewards

    if episode_counter.sum() == 0:
        episode_counter = np.ones(16)

    # stop training if avg_reward > solved_reward
    if episode_rewards.sum() / episode_counter.sum() > solved_reward:
        print("########## Solved! ##########")

    # logging
    if timestep % log_interval == 0:
        print('Episode {} \t episode reward: {} \t'.format(i_episode, episode_rewards.sum() / episode_counter.sum()))