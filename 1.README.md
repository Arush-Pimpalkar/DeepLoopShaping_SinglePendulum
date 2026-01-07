# Overview

 - Implementing simple pendulum using MuJoCo.
 - Added gaussian noise at action using ActionGaussianNoiseWrapper  
 - Tried various methods to attenuate noise at f > 10 hz
 - Compared with a simple SAC trained without noise based rewards 
 - Notebook: [simple_gaussian.ipynb](https://github.com/Arush-Pimpalkar/RL_Arush/blob/main/simple_gaussian.ipynb
)
### Simple SAC (Baseline):
 - Standard Soft Actor-Critic algorithm without any additional penalty terms
 - Trains directly on a noisy environment with Gaussian action noise (SNR = 10)
 - Uses default SAC hyperparameters (learning rate 3e-4, gamma 0.99)
 - **Reward Function:** `R = R_base` (no modifications)

![Base Reward Function](https://raw.githubusercontent.com/Arush-Pimpalkar/RL_Arush/main/plots/base_reward_function.png)

## Methods tried: 

### DeepMind Style:
 - Applies a sigmoid function for energy available over 10 hz. 
 - Using with the base reward function showed better results.
 - **Reward Function:** `R = R_base - (λ_penalty × sigmoid_score)`
 
![DeepMind Style Comparison](https://raw.githubusercontent.com/Arush-Pimpalkar/RL_Arush/main/plots/deepmind_style_vs_simple_sac.png)

### CombinedPenaltyWrapper:
 - Combines both high-frequency filtering and action rate penalties for better noise attenuation
 - Uses a low-pass filter (5Hz cutoff) to extract high-frequency components.
 
![Combined Penalty Comparison](https://raw.githubusercontent.com/Arush-Pimpalkar/RL_Arush/main/plots/combined_penalty_vs_simple_sac.png)


### ActionRatePenaltyWrapper: 
 - Penalizes rapid changes in consecutive actions without using any filtering
 - Tracks previous action and penalizes the squared difference with current action
 - **Reward Function:** `R = R_base - λ_rate × (u_t - u_{t-1})²`
 
![Action Rate Penalty Comparison](https://raw.githubusercontent.com/Arush-Pimpalkar/RL_Arush/main/plots/action_rate_penalty_vs_simple_sac.png)


## My interpretations: 
 - Incorporating a sigmoid based penalty on high-frequency state energy (>10 Hz) alongside the base reward effectively attenuates noise without destabilizing learning. (Like the DeepMind Work)
 - Combining the sigmoid high-frequency score with an explicit high-frequency action penalty does not show better attenuation. Extended training may show similar result.
 - Penalizing high-frequency or rapid action changes alone (without a frequency-aware shaping term) did not improve attenuation.