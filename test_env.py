"""
Quick Test Script for TradingEnv

This script performs a basic sanity check of the TradingEnv.
Run this to verify that everything is set up correctly.

Usage:
    python test_env.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from envs.trading_env import TradingEnv


def main():
    print("=" * 60)
    print("🧪 AlphaTraderLab - Environment Test")
    print("=" * 60)
    print()
    
    # Step 1: Download sample data
    print("📡 Step 1: Downloading sample BTC-USD data...")
    ticker = "BTC-USD"
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    print(f"✅ Downloaded {len(df)} days of data")
    print()
    
    # Step 2: Create environment
    print("🎮 Step 2: Creating TradingEnv...")
    env = TradingEnv(
        df=df,
        window_size=30,
        initial_balance=10000.0,
        transaction_cost=0.001
    )
    print("✅ Environment created successfully")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Action space: {env.action_space}")
    print()
    
    # Step 3: Test reset
    print("🔄 Step 3: Testing reset()...")
    observation, info = env.reset(seed=42)
    print(f"✅ Reset successful")
    print(f"   - Observation shape: {observation.shape}")
    print(f"   - Initial equity: ${env.equity:.2f}")
    print()
    
    # Step 4: Test step
    print("👟 Step 4: Testing step() with random actions...")
    total_reward = 0
    num_steps = 10
    
    for i in range(num_steps):
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        action_names = ['FLAT', 'LONG', 'SHORT']
        print(f"   Step {i+1}: Action={action_names[action]}, "
              f"Reward={reward:.4f}, Equity=${info['equity']:.2f}")
        
        if done:
            print("   Episode ended early")
            break
    
    print()
    print(f"✅ All steps completed successfully")
    print(f"   - Total reward: {total_reward:.4f}")
    print(f"   - Final equity: ${info['equity']:.2f}")
    print()
    
    # Step 5: Summary
    print("=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)
    print()
    print("✅ Your TradingEnv is working correctly!")
    print("👉 Next step: Open the Jupyter notebook and try the full demo")
    print()


if __name__ == "__main__":
    main()
