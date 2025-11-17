"""
Train PPO Agent on TradingEnv

This script trains a PPO (Proximal Policy Optimization) agent on historical
trading data and evaluates it against baselines.

Usage:
    python train_ppo.py [--timesteps 100000] [--test]
"""

import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# Import environment and utilities
from envs.trading_env import TradingEnv
from utils.evaluation import (
    evaluate_agent,
    evaluate_random_agent,
    evaluate_buy_and_hold,
    compare_agents
)

# Import Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback


def load_and_split_data(ticker="BTC-USD", start_date="2018-01-01", train_ratio=0.7):
    """
    Download historical data and split into train/test sets.
    
    Parameters:
    -----------
    ticker : str
        Stock/crypto ticker symbol.
    start_date : str
        Start date for data download (YYYY-MM-DD).
    train_ratio : float
        Ratio of data to use for training (0.0 to 1.0).
    
    Returns:
    --------
    train_df, test_df : pandas.DataFrame
        Training and testing dataframes.
    """
    print(f"\n📊 Downloading {ticker} data from {start_date}...")
    
    # Download data
    end_date = datetime.now().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    
    if len(df) == 0:
        raise ValueError(f"No data downloaded for {ticker}")
    
    print(f"✅ Downloaded {len(df)} days of data")
    print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Split into train and test
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"\n📈 Data split:")
    print(f"   Training:   {len(train_df)} days ({train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')})")
    print(f"   Testing:    {len(test_df)} days ({test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')})")
    
    return train_df, test_df


def make_env(df, window_size=30, initial_balance=10000.0, use_indicators=True):
    """
    Create a TradingEnv and wrap it for Stable-Baselines3.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV data for the environment.
    window_size : int
        Number of past candles in observation.
    initial_balance : float
        Starting portfolio value.
    use_indicators : bool
        Whether to include technical indicators in observations.
    
    Returns:
    --------
    env : DummyVecEnv
        Vectorized environment for SB3.
    """
    def _init():
        return TradingEnv(
            df=df,
            window_size=window_size,
            initial_balance=initial_balance,
            transaction_cost=0.001,
            use_indicators=use_indicators
        )
    
    return DummyVecEnv([_init])


def train_ppo_agent(train_env, total_timesteps=100_000, save_path="models/ppo_btc.zip"):
    """
    Train a PPO agent on the training environment.
    
    Parameters:
    -----------
    train_env : VecEnv
        Vectorized training environment.
    total_timesteps : int
        Total number of timesteps to train for.
    save_path : str
        Path to save the trained model.
    
    Returns:
    --------
    model : PPO
        Trained PPO model.
    """
    print(f"\n🤖 Training PPO agent for {total_timesteps:,} timesteps...")
    
    # Create PPO model with reasonable hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,      # Standard learning rate
        n_steps=2048,             # Steps per update
        batch_size=64,            # Minibatch size
        n_epochs=10,              # Epochs per update
        gamma=0.99,               # Discount factor
        gae_lambda=0.95,          # GAE parameter
        clip_range=0.2,           # PPO clipping
        ent_coef=0.01,            # Entropy coefficient (exploration)
        verbose=1                 # Print training progress
        # Note: tensorboard_log removed to avoid dependency issues
    )
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n✅ Model saved to: {save_path}")
    
    return model


def evaluate_all_agents(test_df, model_path="models/ppo_btc.zip", window_size=30, use_indicators=True):
    """
    Evaluate PPO, Random, and Buy & Hold agents on test data.
    
    Parameters:
    -----------
    test_df : pandas.DataFrame
        Test data.
    model_path : str
        Path to trained PPO model.
    window_size : int
        Window size for environment.
    use_indicators : bool
        Whether to use technical indicators.
    
    Returns:
    --------
    results : dict
        Dictionary with results for each agent.
    """
    print(f"\n📊 Evaluating agents on test data...")
    
    # Load trained model
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"✅ Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Create test environment (non-vectorized for evaluation)
    test_env = TradingEnv(
        df=test_df,
        window_size=window_size,
        initial_balance=10000.0,
        transaction_cost=0.001,
        use_indicators=use_indicators
    )
    
    # Evaluate PPO agent
    print("\n🤖 Evaluating PPO agent...")
    ppo_results = evaluate_agent(test_env, model=model, n_episodes=1, deterministic=True)
    print(f"   Final equity: ${ppo_results['final_equity']:,.2f}")
    print(f"   Total return: {ppo_results['total_return']:.2f}%")
    print(f"   Max drawdown: {ppo_results['max_drawdown']:.2f}%")
    print(f"   Sharpe ratio: {ppo_results['sharpe_ratio']:.2f}")
    
    # Evaluate Random agent
    test_env.reset()
    print("\n🎲 Evaluating Random agent...")
    random_results = evaluate_random_agent(test_env, n_episodes=1)
    print(f"   Final equity: ${random_results['final_equity']:,.2f}")
    print(f"   Total return: {random_results['total_return']:.2f}%")
    print(f"   Max drawdown: {random_results['max_drawdown']:.2f}%")
    print(f"   Sharpe ratio: {random_results['sharpe_ratio']:.2f}")
    
    # Evaluate Buy & Hold
    test_env.reset()
    print("\n📈 Evaluating Buy & Hold strategy...")
    bh_results = evaluate_buy_and_hold(test_env, n_episodes=1)
    print(f"   Final equity: ${bh_results['final_equity']:,.2f}")
    print(f"   Total return: {bh_results['total_return']:.2f}%")
    print(f"   Max drawdown: {bh_results['max_drawdown']:.2f}%")
    print(f"   Sharpe ratio: {bh_results['sharpe_ratio']:.2f}")
    
    return {
        'PPO': ppo_results,
        'Random': random_results,
        'Buy & Hold': bh_results
    }


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Train PPO agent on TradingEnv')
    parser.add_argument('--timesteps', type=int, default=100_000,
                        help='Total training timesteps (default: 100,000)')
    parser.add_argument('--test', action='store_true',
                        help='Only run evaluation (skip training)')
    parser.add_argument('--ticker', type=str, default='BTC-USD',
                        help='Ticker symbol (default: BTC-USD)')
    parser.add_argument('--window-size', type=int, default=30,
                        help='Observation window size (default: 30)')
    parser.add_argument('--use-indicators', action='store_true', default=True,
                        help='Use technical indicators (default: True)')
    parser.add_argument('--no-indicators', action='store_true',
                        help='Disable technical indicators')
    
    args = parser.parse_args()
    
    # Handle indicator flag
    use_indicators = args.use_indicators and not args.no_indicators
    
    print("=" * 60)
    print("🚀 AlphaTraderLab - PPO Training Pipeline")
    print("=" * 60)
    
    # Load and split data
    train_df, test_df = load_and_split_data(
        ticker=args.ticker,
        start_date="2018-01-01",
        train_ratio=0.7
    )
    
    model_path = f"models/ppo_{args.ticker.replace('-', '_').lower()}.zip"
    
    print(f"\n🔧 Configuration:")
    print(f"   Use indicators: {use_indicators}")
    print(f"   Window size: {args.window_size}")
    
    # Training phase
    if not args.test:
        # Create training environment
        train_env = make_env(train_df, window_size=args.window_size, use_indicators=use_indicators)
        
        # Train PPO agent
        model = train_ppo_agent(
            train_env=train_env,
            total_timesteps=args.timesteps,
            save_path=model_path
        )
    else:
        print("\n⏭️  Skipping training (--test flag set)")
    
    # Evaluation phase
    print("\n" + "=" * 60)
    print("📊 Evaluation Phase")
    print("=" * 60)
    
    results = evaluate_all_agents(
        test_df=test_df,
        model_path=model_path,
        window_size=args.window_size,
        use_indicators=use_indicators
    )
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("📊 Agent Comparison")
    print("=" * 60)
    comparison_df = compare_agents(results)
    print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✅ Training and evaluation complete!")
    print("=" * 60)
    print(f"\n💡 Tips:")
    print(f"   - Model saved at: {model_path}")
    print(f"   - Run with --test to skip training and only evaluate")
    print(f"   - Run with --timesteps 200000 for longer training")
    print(f"   - Check notebooks/AlphaTraderLab_PPO_v1.ipynb for visualizations")


if __name__ == "__main__":
    main()
