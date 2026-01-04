import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces


class MuJoCoPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, xml_path="pendulum_inverted.xml", render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep
        self.render_mode = render_mode
        self.viewer = None

        # Action: torque
        self.max_torque = 10.0
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(1,),
            dtype=np.float32,
        )

        # Observation: [cosθ, sinθ, θ̇]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -8.0]),
            high=np.array([1.0, 1.0, 8.0]),
            dtype=np.float32,
        )

        # Start at a random position
        self.data.qpos[0] = np.random.uniform(-np.pi, np.pi)
        self.data.qvel[0] = np.random.uniform(-1.0, 1.0)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
    
        # Check if specific starting angle is requested
        if options and "start_angle" in options:
            self.data.qpos[0] = options["start_angle"]
            self.data.qvel[0] = options.get("start_velocity", 0.0)
        else:
            # Use seed for deterministic or random starting position
            if seed is not None:
                np.random.seed(seed)
            self.data.qpos[0] = np.random.uniform(-np.pi, np.pi)
            self.data.qvel[0] = np.random.uniform(-1.0, 1.0)
    
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}


    def step(self, action):
        action = np.clip(action, -self.max_torque, self.max_torque)
        self.data.ctrl[0] = action[0]

        mujoco.mj_step(self.model, self.data)

        theta = self.data.qpos[0]
        theta_dot = self.data.qvel[0]

        # Normalize theta to [-π, π] where 0 is upright
        theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # Reward: maximize for staying upright (theta_normalized ≈ 0)
        reward = (
            1.0
            - theta_normalized**2
            - 0.1 * theta_dot**2
            - 0.001 * action[0]**2
        )

        # Terminate if pendulum falls past horizontal (|θ| > π/2)
        terminated = abs(theta_normalized) > np.pi / 2
        truncated = False
        info = {}

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        theta = self.data.qpos[0]
        theta_dot = self.data.qvel[0]

        return np.array(
            [np.cos(theta), np.sin(theta), theta_dot],
            dtype=np.float32,
        )

    def _compute_reward(self, action):
        theta = self.data.qpos[0]
        theta_dot = self.data.qvel[0]
        torque = action[0]

        # Upright error (wrapped)
        theta_err = ((theta - np.pi + np.pi) % (2*np.pi)) - np.pi

        return -(
            theta_err**2
            + 0.1 * theta_dot**2
            + 0.001 * torque**2
        )

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
