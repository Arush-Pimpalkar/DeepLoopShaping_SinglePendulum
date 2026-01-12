"""LIGO Angular Sensing and Control (ASC) Gymnasium Environment.

This environment implements the linear SISO model for θ_CHP (Common Hard Pitch)
training as described in the paper. It uses:
- Quadruple pendulum dynamics with radiation pressure
- Seismic disturbance (force noise) and sensor noise from identified PSDs
- Frequency-domain rewards using IIR bandpass filters
- Domain randomization for robust policy training

References:
- Section S2.3.2: Lightsaber simulator description
- Section S2.4: Linear SISO model for θ_CHP training
- Section S2.5: Noise generation from inverse FFT of PSDs
- Section S3.2: Environment specifications
- Section S3.5: Frequency-domain rewards
- Table S3: Domain randomization parameters
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy import signal
from typing import Optional, Dict, Any, Tuple
import os
from pathlib import Path


class LIGOASCEnv(gym.Env):
    """LIGO Angular Sensing and Control Environment.
    
    This environment simulates the θ_CHP (Common Hard Pitch) degree of freedom
    for training RL-based controllers. The plant model includes:
    - Quadruple pendulum mechanical response
    - Radiation pressure dynamics (RHP pole)
    - Optical sensor gain
    
    The environment follows the specifications from the paper:
    - Action limits: ±3000 counts
    - Episode length: 1024 seconds (262,144 steps at 256 Hz)
    - Termination on large angular errors
    
    Attributes:
        fs_sim: Simulation sampling frequency (2048 Hz for numerical stability)
        fs_ctrl: Control policy evaluation frequency (256 Hz)
        decimation_factor: Ratio of fs_sim to fs_ctrl
        episode_duration: Episode length in seconds
        action_limit: Maximum control action in counts
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(
        self,
        fs_sim: float = 2048.0,
        fs_ctrl: float = 256.0,
        episode_duration: float = 1024.0,
        action_limit: float = 3000.0,
        termination_threshold: float = 1e-6,  # rad, ~1 µrad
        noise_data_dir: Optional[str] = None,
        plant_data_path: Optional[str] = None,
        seed: Optional[int] = None,
        enable_domain_randomization: bool = True,
        render_mode: Optional[str] = None,
    ):
        """Initialize the LIGO ASC environment.
        
        Args:
            fs_sim: Simulation sampling frequency in Hz (default 2048 Hz)
            fs_ctrl: Control evaluation frequency in Hz (default 256 Hz)
            episode_duration: Episode length in seconds (default 1024 s)
            action_limit: Maximum action magnitude in counts (default ±3000)
            termination_threshold: Angular error threshold for termination in rad
            noise_data_dir: Path to directory containing noise PSD files
            plant_data_path: Path to plant model parameters (.npz file)
            seed: Random seed for reproducibility
            enable_domain_randomization: Whether to use domain randomization
            render_mode: Rendering mode (optional)
        """
        super().__init__()
        
        self.fs_sim = fs_sim
        self.fs_ctrl = fs_ctrl
        self.decimation_factor = int(fs_sim / fs_ctrl)
        self.episode_duration = episode_duration
        self.action_limit = action_limit
        self.termination_threshold = termination_threshold
        self.enable_domain_randomization = enable_domain_randomization
        self.render_mode = render_mode
        
        # Set up paths
        self._setup_paths(noise_data_dir, plant_data_path)
        
        # Random state
        self._rng = np.random.default_rng(seed)
        
        # Episode parameters
        self.max_steps = int(episode_duration * fs_ctrl)  # 262,144 steps
        
        # Observation space: [pitch_error, pitch_error_filtered, control_output]
        # Plus history buffer for proper state representation
        obs_dim = 8  # Current error, filtered bands, control history
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: continuous control output in counts
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Initialize plant model
        self._init_plant_model()
        
        # Initialize noise generators
        self._init_noise_generators()
        
        # Initialize frequency-domain reward filters (8-30 Hz observation band)
        self._init_reward_filters()
        
        # State variables
        self._reset_state()
        
    def _setup_paths(self, noise_data_dir: Optional[str], plant_data_path: Optional[str]):
        """Set up paths to data files."""
        base_path = Path(__file__).parent.parent.parent.parent
        
        if noise_data_dir is None:
            self.noise_data_dir = base_path / "zenodo_data" / "noise_inputs"
        else:
            self.noise_data_dir = Path(noise_data_dir)
            
        if plant_data_path is None:
            self.plant_data_path = base_path / "zenodo_data" / "hdf5" / "CHARD-plant-fit-params.npz"
        else:
            self.plant_data_path = Path(plant_data_path)
            
    def _init_plant_model(self):
        """Initialize the plant model (quadruple pendulum + radiation pressure).
        
        The plant transfer function from actuation to pitch angle includes:
        - Mechanical response of quadruple pendulum
        - Radiation pressure coupling (RHP pole causing instability)
        - Optical sensor gain
        
        The model is converted to discrete-time using Modified Matched Pole-Zero
        bilinear transformation for numerical stability.
        """
        # Load plant parameters from zenodo data
        if self.plant_data_path.exists():
            params = np.load(self.plant_data_path, allow_pickle=True)
            self.plant_zeros = params['z_p3']
            self.plant_poles = params['p_p3']
            self.plant_gain = float(params['k_p3'])
        else:
            # Default CHARD pitch plant model (from paper)
            # RHP pole at ~2.4 Hz (unstable radiation pressure mode)
            self.plant_zeros = np.array([])
            self.plant_poles = np.array([
                -0.18849556 + 6.53451272j,  # Pendulum mode ~1 Hz
                -0.18849556 - 6.53451272j,
                0.40840704 + 15.07964474j,   # RHP pole ~2.4 Hz (unstable!)
                0.40840704 - 15.07964474j,
            ])
            self.plant_gain = 80.0
            
        # Store base RHP pole frequency for domain randomization
        rhp_poles = self.plant_poles[np.real(self.plant_poles) > 0]
        if len(rhp_poles) > 0:
            self.base_rhp_freq = np.abs(np.imag(rhp_poles[0])) / (2 * np.pi)
        else:
            self.base_rhp_freq = 2.4  # Default ~2.4 Hz
            
        # Convert continuous-time to discrete-time using bilinear transform
        self._discretize_plant()
        
    def _discretize_plant(self, rhp_scale: float = 1.0):
        """Discretize the plant model using bilinear transform.
        
        Args:
            rhp_scale: Scale factor for RHP pole frequency (for domain randomization)
        """
        # Apply RHP pole scaling for domain randomization
        poles = self.plant_poles.copy()
        rhp_mask = np.real(poles) > 0
        if np.any(rhp_mask):
            poles[rhp_mask] = poles[rhp_mask] * rhp_scale
            
        # Create continuous-time system
        if len(self.plant_zeros) == 0:
            sys_c = signal.ZerosPolesGain([], poles, self.plant_gain)
        else:
            sys_c = signal.ZerosPolesGain(self.plant_zeros, poles, self.plant_gain)
            
        # Convert to transfer function then to SOS for numerical stability
        sys_tf = sys_c.to_tf()
        
        # Discretize using bilinear (Tustin) transform
        sys_d = signal.cont2discrete((sys_tf.num, sys_tf.den), 1/self.fs_sim, method='bilinear')
        
        # Convert to second-order sections for low round-off noise
        try:
            self.plant_sos = signal.tf2sos(sys_d[0].flatten(), sys_d[1])
        except:
            # Fallback to direct form if SOS conversion fails
            self.plant_b = sys_d[0].flatten()
            self.plant_a = sys_d[1]
            self.plant_sos = None
            
        # Initialize filter state
        if self.plant_sos is not None:
            self.plant_state = np.zeros((self.plant_sos.shape[0], 2))
        else:
            self.plant_state = np.zeros(max(len(self.plant_b), len(self.plant_a)) - 1)
            
    def _init_noise_generators(self):
        """Initialize noise generators from PSDs.
        
        Generates noise time series using inverse FFT of identified noise PSDs:
        - Seismic disturbance (force noise on pendulum)
        - Sensor noise (readout noise)
        
        The noise is generated in batches and uses the method from Section S2.5.
        """
        self.noise_batch_duration = 64.0  # seconds
        self.noise_batch_samples = int(self.noise_batch_duration * self.fs_sim)
        
        # Load seismic noise PSD (suspension input noise)
        seismic_file = self.noise_data_dir / "ITM_SEI_LIGO_O3.csv"
        if seismic_file.exists():
            self.seismic_psd = self._load_psd(seismic_file)
        else:
            # Default pink noise spectrum
            self.seismic_psd = self._default_seismic_psd()
            
        # Load sensor noise PSD
        sensor_file = self.noise_data_dir / "SENSOR_PITCH_HARD.csv"
        if sensor_file.exists():
            self.sensor_psd = self._load_psd(sensor_file)
        else:
            # Default white noise spectrum
            self.sensor_psd = self._default_sensor_psd()
            
        # Pre-generate initial noise batch
        self._generate_noise_batch()
        
    def _load_psd(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load PSD data from CSV file.
        
        Args:
            filepath: Path to CSV file with frequency, amplitude columns
            
        Returns:
            Tuple of (frequencies, sqrt_psd_values)
        """
        # Try comma delimiter first, then space/whitespace
        try:
            data = np.genfromtxt(filepath, delimiter=',')
            if data.ndim == 1:
                # Single column or wrong delimiter, try whitespace
                data = np.genfromtxt(filepath)
        except:
            data = np.genfromtxt(filepath)
            
        if data.ndim == 1:
            raise ValueError(f"Could not parse PSD file: {filepath}")
            
        return (data[:, 0], data[:, 1])
        
    def _default_seismic_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate default seismic noise PSD (1/f^2 spectrum)."""
        freqs = np.logspace(-2, np.log10(self.fs_sim/2), 1000)
        # Typical seismic noise: ~1e-7 rad/√Hz at 1 Hz, falling as 1/f^2
        sqrt_psd = 1e-7 * (1.0 / np.maximum(freqs, 0.01))**2
        return (freqs, sqrt_psd)
        
    def _default_sensor_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate default sensor noise PSD (white noise)."""
        freqs = np.logspace(-2, np.log10(self.fs_sim/2), 1000)
        # Typical sensor noise: ~1e-10 rad/√Hz (flat)
        sqrt_psd = np.ones_like(freqs) * 1e-10
        return (freqs, sqrt_psd)
        
    def _generate_noise_batch(self, noise_scale: float = 1.0):
        """Generate a batch of noise time series from PSDs.
        
        Uses the inverse FFT method from Section S2.5:
        1. Generate white noise in frequency domain
        2. Shape spectrum using sqrt(PSD)
        3. Inverse FFT to get time domain signal
        
        Args:
            noise_scale: Multiplicative scale factor for domain randomization
        """
        n_samples = self.noise_batch_samples
        freqs = np.fft.rfftfreq(n_samples, 1/self.fs_sim)
        
        # Generate seismic noise
        seismic_sqrt_psd = np.interp(freqs, self.seismic_psd[0], self.seismic_psd[1])
        white_noise = self._rng.standard_normal(len(freqs)) + 1j * self._rng.standard_normal(len(freqs))
        colored_spectrum = white_noise * seismic_sqrt_psd * np.sqrt(self.fs_sim / 2)
        colored_spectrum[0] = 0  # DC = 0
        self.seismic_noise = np.fft.irfft(colored_spectrum, n_samples) * noise_scale
        
        # Generate sensor noise
        sensor_sqrt_psd = np.interp(freqs, self.sensor_psd[0], self.sensor_psd[1])
        white_noise = self._rng.standard_normal(len(freqs)) + 1j * self._rng.standard_normal(len(freqs))
        colored_spectrum = white_noise * sensor_sqrt_psd * np.sqrt(self.fs_sim / 2)
        colored_spectrum[0] = 0
        self.sensor_noise = np.fft.irfft(colored_spectrum, n_samples) * noise_scale
        
        self.noise_index = 0
        
    def _init_reward_filters(self):
        """Initialize IIR bandpass filters for frequency-domain rewards.
        
        From Section S3.5: The observation band is 8-30 Hz.
        We use IIR filters to compute band-limited RMS for reward calculation.
        """
        # Observation band: 8-30 Hz
        self.obs_band_low = 8.0
        self.obs_band_high = 30.0
        
        # Design bandpass filter (4th order Butterworth)
        nyq = self.fs_ctrl / 2
        low = self.obs_band_low / nyq
        high = min(self.obs_band_high / nyq, 0.99)
        
        self.reward_sos = signal.butter(4, [low, high], btype='band', output='sos')
        self.reward_filter_state = np.zeros((self.reward_sos.shape[0], 2))
        
        # Additional filter for lower frequency band (control band)
        ctrl_band_high = 8.0 / nyq
        self.ctrl_sos = signal.butter(4, ctrl_band_high, btype='low', output='sos')
        self.ctrl_filter_state = np.zeros((self.ctrl_sos.shape[0], 2))
        
    def _reset_state(self):
        """Reset all state variables."""
        self.current_step = 0
        self.pitch_angle = 0.0
        self.pitch_rate = 0.0
        self.control_output = 0.0
        self.filtered_error_obs = 0.0
        self.filtered_error_ctrl = 0.0
        
        # Reset filter states
        if hasattr(self, 'plant_sos') and self.plant_sos is not None:
            self.plant_state = np.zeros((self.plant_sos.shape[0], 2))
        elif hasattr(self, 'plant_b'):
            self.plant_state = np.zeros(max(len(self.plant_b), len(self.plant_a)) - 1)
            
        if hasattr(self, 'reward_sos'):
            self.reward_filter_state = np.zeros((self.reward_sos.shape[0], 2))
            self.ctrl_filter_state = np.zeros((self.ctrl_sos.shape[0], 2))
            
        # History buffers for observation
        self.error_history = np.zeros(4)
        self.control_history = np.zeros(4)
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for this episode
            options: Additional options (e.g., domain randomization params)
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            
        # Apply domain randomization
        if self.enable_domain_randomization:
            # Scalar noise multiplier: uniform [1.0, 5.0]
            noise_scale = self._rng.uniform(1.0, 5.0)
            # RHP pole frequency: log-uniform [0.8, 1.2]
            rhp_scale = np.exp(self._rng.uniform(np.log(0.8), np.log(1.2)))
            
            self._discretize_plant(rhp_scale)
            self._generate_noise_batch(noise_scale)
        else:
            noise_scale = 1.0
            rhp_scale = 1.0
            self._generate_noise_batch(noise_scale)
            
        self._reset_state()
        
        obs = self._get_observation()
        info = {
            "noise_scale": noise_scale if self.enable_domain_randomization else 1.0,
            "rhp_scale": rhp_scale if self.enable_domain_randomization else 1.0,
        }
        
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one control step.
        
        The control is evaluated at fs_ctrl (256 Hz), but the plant simulation
        runs at fs_sim (2048 Hz) for numerical stability.
        
        Args:
            action: Control action in [-1, 1], scaled to ±action_limit counts
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action from [-1, 1] to actual counts
        control = float(action[0]) * self.action_limit
        
        # Simulate plant at high frequency
        pitch_samples = []
        for _ in range(self.decimation_factor):
            # Get noise samples
            if self.noise_index >= self.noise_batch_samples:
                self._generate_noise_batch()
            seismic = self.seismic_noise[self.noise_index]
            sensor = self.sensor_noise[self.noise_index]
            self.noise_index += 1
            
            # Plant dynamics: control -> actuation force -> pitch angle
            # The plant includes the mechanical response and radiation pressure
            plant_input = control + seismic  # Control + disturbance
            
            if self.plant_sos is not None:
                pitch_out, self.plant_state = signal.sosfilt(
                    self.plant_sos, [plant_input], zi=self.plant_state
                )
                self.pitch_angle = pitch_out[0]
            else:
                pitch_out, self.plant_state = signal.lfilter(
                    self.plant_b, self.plant_a, [plant_input], zi=self.plant_state
                )
                self.pitch_angle = pitch_out[0]
                
            # Add sensor noise to measurement
            measurement = self.pitch_angle + sensor
            pitch_samples.append(measurement)
            
        # Decimate to control frequency (take last sample)
        pitch_error = pitch_samples[-1]
        
        # Update filtered signals for reward computation
        filtered_obs, self.reward_filter_state = signal.sosfilt(
            self.reward_sos, [pitch_error], zi=self.reward_filter_state
        )
        self.filtered_error_obs = filtered_obs[0]
        
        filtered_ctrl, self.ctrl_filter_state = signal.sosfilt(
            self.ctrl_sos, [pitch_error], zi=self.ctrl_filter_state
        )
        self.filtered_error_ctrl = filtered_ctrl[0]
        
        # Update history
        self.error_history = np.roll(self.error_history, 1)
        self.error_history[0] = pitch_error
        self.control_history = np.roll(self.control_history, 1)
        self.control_history[0] = control / self.action_limit
        
        self.control_output = control
        self.current_step += 1
        
        # Compute reward using frequency-domain scoring
        reward = self._compute_reward(pitch_error, control)
        
        # Check termination conditions
        terminated = abs(self.pitch_angle) > self.termination_threshold
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_observation()
        info = {
            "pitch_angle": self.pitch_angle,
            "pitch_error": pitch_error,
            "control_output": control,
            "filtered_error_obs": self.filtered_error_obs,
            "step": self.current_step,
        }
        
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector.
        
        Returns:
            Observation array containing error and control signals
        """
        obs = np.array([
            self.error_history[0] * 1e6,  # Current error (scaled to µrad)
            self.error_history[1] * 1e6,  # Previous error
            self.filtered_error_obs * 1e6,  # Observation band filtered error
            self.filtered_error_ctrl * 1e6,  # Control band filtered error
            self.control_history[0],  # Current control (normalized)
            self.control_history[1],  # Previous control
            self.control_history[2],  # Control t-2
            self.control_history[3],  # Control t-3
        ], dtype=np.float32)
        
        return obs
        
    def _compute_reward(self, pitch_error: float, control: float) -> float:
        """Compute reward using frequency-domain scoring.
        
        From Section S3.5: Uses sigmoid scoring function with multiplicative
        composition over frequency bands.
        
        Args:
            pitch_error: Current pitch error in radians
            control: Control action in counts
            
        Returns:
            Reward value
        """
        # Observation band (8-30 Hz) - primary science requirement
        obs_rms = abs(self.filtered_error_obs)
        obs_target = 1e-9  # Target: 1 nrad RMS
        obs_score = self._sigmoid_score(obs_rms, obs_target, 10.0)
        
        # Control effort penalty (prevent excessive actuation)
        ctrl_penalty = (control / self.action_limit) ** 2
        
        # Stability reward (encourage small errors)
        stability_score = np.exp(-abs(pitch_error) / 1e-8)
        
        # Multiplicative composition
        reward = obs_score * stability_score * (1.0 - 0.1 * ctrl_penalty)
        
        return float(reward)
        
    def _sigmoid_score(self, value: float, target: float, steepness: float) -> float:
        """Sigmoid scoring function.
        
        Args:
            value: Current value
            target: Target value (score = 0.5 at target)
            steepness: Steepness of sigmoid transition
            
        Returns:
            Score in [0, 1]
        """
        ratio = value / target
        return 1.0 / (1.0 + ratio ** steepness)
        
    def render(self):
        """Render the environment (optional visualization)."""
        if self.render_mode == "human":
            print(f"Step {self.current_step}: pitch={self.pitch_angle*1e9:.2f} nrad, "
                  f"control={self.control_output:.1f} counts")
                  
    def close(self):
        """Clean up resources."""
        pass


class LIGOASCEnvSimple(LIGOASCEnv):
    """Simplified LIGO ASC environment for faster training.
    
    Uses a simpler plant model with just the essential dynamics:
    - Single RHP pole for radiation pressure instability
    - Second-order pendulum response
    """
    
    def __init__(self, **kwargs):
        # Override to use simplified model
        kwargs['plant_data_path'] = None  # Use default simple model
        super().__init__(**kwargs)
        
    def _init_plant_model(self):
        """Initialize simplified plant model."""
        # Simple 2nd order system with RHP pole
        # Represents pendulum mode + radiation pressure instability
        self.plant_zeros = np.array([])
        self.plant_poles = np.array([
            -0.3 + 2.7j,   # Pendulum mode ~0.43 Hz, damped
            -0.3 - 2.7j,
            0.4 + 15.0j,   # RHP pole ~2.4 Hz (unstable)
            0.4 - 15.0j,
        ])
        self.plant_gain = 50.0
        self.base_rhp_freq = 2.4
        
        self._discretize_plant()


# Register environments with Gymnasium
def register_ligo_envs():
    """Register LIGO ASC environments with Gymnasium."""
    from gymnasium.envs.registration import register
    
    try:
        register(
            id='LIGO-ASC-v0',
            entry_point='rl_arush.envs.ligo_asc_env:LIGOASCEnv',
            max_episode_steps=262144,
        )
    except:
        pass  # Already registered
        
    try:
        register(
            id='LIGO-ASC-Simple-v0',
            entry_point='rl_arush.envs.ligo_asc_env:LIGOASCEnvSimple',
            max_episode_steps=262144,
        )
    except:
        pass


if __name__ == "__main__":
    # Quick test
    env = LIGOASCEnv(enable_domain_randomization=False)
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 10 == 0:
            print(f"Step {i}: reward={reward:.4f}, pitch={info['pitch_angle']*1e9:.2f} nrad")
        if terminated:
            print(f"Terminated at step {i}")
            break
