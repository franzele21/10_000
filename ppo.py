import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from game_env import Env10000  # or from 10000_env import Env10000 if named so

class RewardLoggingCallback(BaseCallback):
    """
    Prints episode reward and running cumulative reward to console and logs them
    to TensorBoard (via model's logger) when Monitor returns episode info.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.cumulative_reward = 0.0
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            return True
        for info in infos:
            if info is None:
                continue
            ep = info.get("episode")
            if ep is not None:
                ep_reward = float(ep["r"])
                self.episode_count += 1
                self.cumulative_reward += ep_reward
                # print to console
                print(f"[Episode {self.episode_count}] reward={ep_reward:.2f}, cumulative={self.cumulative_reward:.2f}")
                # log to tensorboard
                self.logger.record("custom/episode_reward", ep_reward)
                self.logger.record("custom/cumulative_reward", self.cumulative_reward)
        return True

def make_env():
    env = Env10000()
    env = Monitor(env)  # Monitor adds 'episode' info to infos when an episode ends
    return env

if __name__ == "__main__":
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        tensorboard_log="./tb_logs", 
    )

    callback = RewardLoggingCallback()
    model.learn(total_timesteps=200000, callback=callback)

    model.save("ppo_10000_env")
    vec_env.save("vecnorm.pkl")
    vec_env.close()