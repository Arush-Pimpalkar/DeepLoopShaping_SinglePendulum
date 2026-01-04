from pendulum_env_simple import MuJoCoPendulumEnv
import time
import numpy as np

env = MuJoCoPendulumEnv(render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    action = np.array([0.0])  # no torque
    obs, reward, terminated, truncated, _ = env.step(action)
    time.sleep(env.dt)

env.close()
