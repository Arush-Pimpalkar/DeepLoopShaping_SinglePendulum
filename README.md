# Overview

 - Implementing simple pendulum using MuJoCo.
 - Added gaussian noise at action using ``` python
 ActionGaussianNoiseWrapper 
 ```
 - Tried various methods to attenuate noise at f > 10 hz

## Methods tried: 

### Simple SAC (Baseline):
 - Standard Soft Actor-Critic algorithm without any additional penalty terms
 - Trains directly on a noisy environment with Gaussian action noise (SNR = 10)
 - Uses default SAC hyperparameters (learning rate 3e-4, gamma 0.99)
 - **Reward Function:** `R = R_base` (no modifications)

### HighFrequencyPenaltyWrapper:
 - Uses a 2nd-order Butterworth high-pass filter to detect high-frequency action components above 10Hz
 - Applies a quadratic penalty to the filtered high-frequency content
 - Designed to directly penalize frequencies above the desired control bandwidth
 - **Reward Function:** `R = R_base - λ_hf × (u_hf)²` where `u_hf` is high-frequency component, `λ_hf = 0.1`

### ActionRatePenaltyWrapper: 
 - Penalizes rapid changes in consecutive actions without using any filtering
 - Simple and computationally efficient approach to reduce high-frequency content
 - Tracks previous action and penalizes the squared difference with current action
 - **Reward Function:** `R = R_base - λ_rate × (u_t - u_{t-1})²` where `λ_rate = 0.1`
 
![Action Rate Penalty Comparison](plots/action_rate_penalty_vs_simple_sac.png)

### CombinedPenaltyWrapper:
 - Combines both high-frequency filtering and action rate penalties for better noise attenuation
 - Uses a low-pass filter (5Hz cutoff) to extract high-frequency components via residual method
 - Applies both frequency-based and temporal-based penalties simultaneously
 - **Reward Function:** `R = R_base - λ_hf × (u - u_lpf)² - λ_rate × (u_t - u_{t-1})²` where `λ_hf = 0.3`, `λ_rate = 0.1`
 
![Combined Penalty Comparison](plots/combined_penalty_vs_simple_sac.png)

### DeepMind Style (Improved DM):
 - Uses exponential moving average to track high-frequency energy over time
 - Applies a sigmoid function to create smooth, differentiable penalty transitions
 - Incorporates energy decay and bias parameters for more sophisticated frequency management
 - **Reward Function:** `R = R_base - λ_penalty × sigmoid_score` where `HF_energy_t = α × HF_energy_{t-1} + (1-α) × (u_hf)²` and `sigmoid_score = 1 / (1 + exp(-scale × (HF_energy_t - bias)))` with `λ_penalty = 0.5`, `scale = 50.0`, `bias = 0.005`, `α = 0.9`
 
![DeepMind Style Comparison](plots/deepmind_style_vs_simple_sac.png)

### Deep Loop Shaping (DLS) Style:
 - Penalizes high-frequency content in the physical state (angular velocity) rather than just actions
 - Uses multiplicative reward shaping instead of additive penalties
 - Focuses on system response frequencies above 10Hz using high-pass filtering of state variables
 - **Reward Function:** `R = R_base × hf_score` where `θ̇_hf = high_pass_filter(angular_velocity, cutoff=10Hz)`, `HF_energy_t = α × HF_energy_{t-1} + (1-α) × (θ̇_hf)²`, and `hf_score = 1 / (1 + exp(scale × (HF_energy_t - threshold)))` with `scale = 40.0`, `threshold = 0.01`, `α = 0.95`
 
![Deep Loop Shaping Comparison](plots/dls_style_vs_simple_sac.png)

## Results

The **Combined Penalty**, **Improved DM**, and **DLS** methods are most effective at reducing high-frequency energy content while maintaining control performance. The DLS method is unique in penalizing state frequencies rather than action frequencies, providing a more direct approach to controlling system response characteristics.

### Overall Comparison
![All Methods Comparison](plots/all_methods_comparison.png)