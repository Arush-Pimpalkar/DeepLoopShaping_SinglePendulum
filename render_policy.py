import time
import numpy as np
from stable_baselines3 import SAC
from pendulum_env_simple import MuJoCoPendulumEnv

# Set initial angle (in radians)
INITIAL_ANGLE = np.pi / 2

# Load trained model and render
env = MuJoCoPendulumEnv(render_mode="human")
model = SAC.load("sac_pendulum", env=env)

def reset_with_angle(env, angle, velocity=0.0):
    """Reset environment with a specific initial angle and velocity."""
    obs, info = env.reset()
    env.data.qpos[0] = angle
    env.data.qvel[0] = velocity
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    return env._get_obs(), info

obs, _ = reset_with_angle(env, INITIAL_ANGLE)
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    if terminated or truncated:
        obs, _ = reset_with_angle(env, INITIAL_ANGLE)
    
    time.sleep(env.dt)
