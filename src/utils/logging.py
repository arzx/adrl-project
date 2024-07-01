from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

class LossLoggingCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(LossLoggingCallback, self).__init__(verbose)
        self.losses = []
        self.rewards = []
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        # Initialize the progress bar at the start of training with total timesteps
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", unit="step")

    def _on_step(self) -> bool:
        # Access the loss value from the training step
        info = self.locals.get('infos', [{}])[0]
        loss = info.get('loss')
        reward = info.get('reward', self.locals.get('rewards', [0])[0])

        # Try to access loss from the model's logger
        if loss is None:
            log_records = self.model.logger.name_to_value
            if 'train/loss' in log_records:
                loss = log_records['train/loss']

        if loss is not None:
            self.losses.append(loss)
        if reward is not None:
            self.rewards.append(reward)

        # Update the progress bar by one step
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        # Close the progress bar at the end of training
        self.pbar.close()

    def get_losses(self):
        return self.losses

    def get_rewards(self):
        return self.rewards