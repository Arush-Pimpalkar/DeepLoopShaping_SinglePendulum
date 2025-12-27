import time
from stable_baselines3 import SAC
from pendulum_env import MuJoCoPendulumEnv
import gymnasium as gym
import numpy as np

# # ---------- TRAIN ----------
# env = MuJoCoPendulumEnv()

# model = SAC(
#     "MlpPolicy",
#     env,
#     learning_rate=3e-4,
#     gamma=0.99,
#     verbose=1,
# )

# model.learn(total_timesteps=40_000)
# model.save("sac_pendulum")

class ActionGaussianNoiseWrapper(gym.Wrapper):
    def __init__(self, env, snr_linear=20.0):
        super().__init__(env)
        self.snr_linear = snr_linear

    def step(self, action):
        # action shape: (1,)
        signal_power = np.mean(action**2) + 1e-8

        # Noise power from SNR definition
        noise_power = signal_power / self.snr_linear
        noise_std = np.sqrt(noise_power)

        noise = np.random.normal(0.0, noise_std, size=action.shape)
        action_noisy = action + noise

        return self.env.step(action_noisy)


# ---------- WATCH ----------
env = MuJoCoPendulumEnv(render_mode="human")
env = ActionGaussianNoiseWrapper(env, snr_linear=0.5)
model = SAC.load("sac_pendulum_gaussian", env=env)

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    time.sleep(env.unwrapped.dt)
