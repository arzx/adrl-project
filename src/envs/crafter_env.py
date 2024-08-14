import argparse
import crafter
import stable_baselines3
from gym.wrappers.monitoring.video_recorder import VideoRecorder

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=float, default=1e6)
args = parser.parse_args()

def create_env():
    """
    Create and return the Crafter environment.
    """
    env = crafter.Env()
    return env
env = create_env()
model = stable_baselines3.PPO('CnnPolicy', env, verbose=1)
env = crafter.Recorder(
    env, args.outdir,
    save_stats=True,
    save_episode=False,
    save_video=True,
)
model.learn(total_timesteps=args.steps)
