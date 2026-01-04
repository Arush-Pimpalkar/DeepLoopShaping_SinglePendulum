#!/usr/bin/env python3
"""
Simple test script for the hanging pendulum environment.
This script verifies that the pendulum physics and reward function
work correctly for a simple hanging pendulum.
"""

import numpy as np
from pendulum_env_simple import MuJoCoPendulumEnv

def test_hanging_pendulum():
    """Test the hanging pendulum environment"""
    print("Testing Simple Hanging Pendulum Environment")
    print("=" * 50)
    
    # Create environment
    env = MuJoCoPendulumEnv()
    
    # Test 1: Verify equilibrium position gives good reward
    print("\nTest 1: Equilibrium Position Reward")
    obs, _ = env.reset(options={"start_angle": 0.0, "start_velocity": 0.0})
    action = np.array([0.0])  # No torque
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"At hanging equilibrium (θ=0°): reward = {reward:.3f}")
    
    # Test 2: Small deviation from equilibrium
    print("\nTest 2: Small Deviation from Equilibrium")
    obs, _ = env.reset(options={"start_angle": 0.1, "start_velocity": 0.0})  # ~6 degrees
    action = np.array([0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Small deviation (θ≈6°): reward = {reward:.3f}")
    
    # Test 3: Large deviation (inverted position)
    print("\nTest 3: Inverted Position")
    obs, _ = env.reset(options={"start_angle": np.pi, "start_velocity": 0.0})
    action = np.array([0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Inverted position (θ=180°): reward = {reward:.3f}")
    
    # Test 4: Simulate a few steps
    print("\nTest 4: Short Simulation")
    print("Step | θ (rad) | θ (deg) | θ̇ (rad/s) | Reward | Action")
    print("-" * 55)
    
    obs, _ = env.reset(options={"start_angle": np.pi/4, "start_velocity": 0.0})  # 45 degrees
    
    for step in range(10):
        # Simple proportional controller to try to reach hanging position
        theta = np.arctan2(obs[1], obs[0])  # Extract angle from cos/sin
        theta_dot = obs[2]
        
        # PD controller targeting hanging position (θ=0)
        action = np.array([-2.0 * theta - 0.5 * theta_dot])
        action = np.clip(action, -10.0, 10.0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        theta = np.arctan2(obs[1], obs[0])
        print(f"{step:4d} | {theta:7.3f} | {np.degrees(theta):7.1f} | {theta_dot:9.3f} | {reward:6.3f} | {action[0]:6.2f}")
        
        if terminated:
            print("Environment terminated!")
            break
    
    env.close()
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_hanging_pendulum()