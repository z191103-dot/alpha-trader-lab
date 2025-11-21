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


def make_env(df, window_size=30, initial_balance=10000.0, use_indicators=True,
             transaction_cost_pct=0.0, switch_penalty=0.0):
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
    transaction_cost_pct : float
        Transaction cost percentage per trade (Step 3.1).
    switch_penalty : float
        Penalty for switching position direction (Step 3.1).
    
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
            transaction_cost=0.001,  # Legacy parameter
            transaction_cost_pct=transaction_cost_pct,
            switch_penalty=switch_penalty,
            use_indicators=use_indicators
        )
    
    return DummyVecEnv([_init])


def train_ppo_agent(train_env, total_timesteps=100_000, save_path="models/ppo_btc.zip",
                    learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
                    ent_coef=0.01, clip_range=0.2):
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
    learning_rate : float
        Learning rate for PPO (Step 3.1).
    gamma : float
        Discount factor (Step 3.1).
    n_steps : int
        Steps per update (Step 3.1).
    batch_size : int
        Minibatch size (Step 3.1).
    ent_coef : float
        Entropy coefficient for exploration (Step 3.1).
    clip_range : float
        PPO clipping parameter (Step 3.1).
    
    Returns:
    --------
    model : PPO
        Trained PPO model.
    """
    print(f"\n🤖 Training PPO agent for {total_timesteps:,} timesteps...")
    
    # Create PPO model with configurable hyperparameters (Step 3.1)
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,              # Keep fixed
        gamma=gamma,
        gae_lambda=0.95,          # Keep fixed
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1
    )
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n✅ Model saved to: {save_path}")
    
    return model


def evaluate_all_agents(test_df, model_path="models/ppo_btc.zip", window_size=30, use_indicators=True,
                        transaction_cost_pct=0.0, switch_penalty=0.0):
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
    transaction_cost_pct : float
        Transaction cost percentage (Step 3.1).
    switch_penalty : float
        Position switch penalty (Step 3.1).
    
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
        transaction_cost_pct=transaction_cost_pct,
        switch_penalty=switch_penalty,
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


def run_experiment_for_ticker(
    ticker,
    timesteps=100_000,
    window_size=30,
    use_indicators=True,
    transaction_cost_pct=0.0,
    switch_penalty=0.0,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    clip_range=0.2,
    output_dir="results",
    skip_training=False,
    verbose=True
):
    """
    Run complete PPO training and evaluation experiment for a single ticker.
    
    This function is designed to be called from multi-asset runners (Step 3.2)
    or used standalone. It handles data loading, training, evaluation, and
    result saving with per-asset file naming.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol (e.g., "BTC-USD", "SPY").
    timesteps : int
        Total training timesteps.
    window_size : int
        Observation window size.
    use_indicators : bool
        Whether to use technical indicators.
    transaction_cost_pct : float
        Transaction cost percentage.
    switch_penalty : float
        Position switch penalty.
    learning_rate : float
        PPO learning rate.
    gamma : float
        PPO discount factor.
    n_steps : int
        PPO steps per update.
    batch_size : int
        PPO minibatch size.
    ent_coef : float
        PPO entropy coefficient.
    clip_range : float
        PPO clipping parameter.
    output_dir : str
        Directory for saving results (default: "results").
    skip_training : bool
        If True, skip training and only evaluate (requires existing model).
    verbose : bool
        If True, print detailed progress.
    
    Returns:
    --------
    results : dict
        Dictionary with keys 'PPO', 'Random', 'Buy & Hold', each containing
        evaluation metrics.
    """
    from config.assets import normalize_ticker_to_slug
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎯 Running experiment for {ticker}")
        print(f"{'='*60}")
    
    # Load and split data
    try:
        train_df, test_df = load_and_split_data(
            ticker=ticker,
            start_date="2018-01-01",
            train_ratio=0.7
        )
    except Exception as e:
        print(f"❌ Error loading data for {ticker}: {e}")
        return None
    
    # Create asset slug for file naming
    asset_slug = normalize_ticker_to_slug(ticker)
    model_path = f"models/ppo_{asset_slug}.zip"
    
    if verbose:
        print(f"\n🔧 Configuration:")
        print(f"   Ticker: {ticker}")
        print(f"   Asset slug: {asset_slug}")
        print(f"   Use indicators: {use_indicators}")
        print(f"   Window size: {window_size}")
        print(f"   Transaction cost: {transaction_cost_pct:.4f}")
        print(f"   Switch penalty: {switch_penalty:.4f}")
    
    # Training phase
    if not skip_training:
        train_env = make_env(
            df=train_df,
            window_size=window_size,
            use_indicators=use_indicators,
            transaction_cost_pct=transaction_cost_pct,
            switch_penalty=switch_penalty
        )
        
        model = train_ppo_agent(
            train_env=train_env,
            total_timesteps=timesteps,
            save_path=model_path,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            clip_range=clip_range
        )
    else:
        if verbose:
            print(f"\n⏭️  Skipping training (using existing model)")
    
    # Evaluation phase
    if verbose:
        print(f"\n{'='*60}")
        print(f"📊 Evaluation Phase - {ticker}")
        print(f"{'='*60}")
    
    results = evaluate_all_agents(
        test_df=test_df,
        model_path=model_path,
        window_size=window_size,
        use_indicators=use_indicators,
        transaction_cost_pct=transaction_cost_pct,
        switch_penalty=switch_penalty
    )
    
    # Save per-asset results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PPO history
    ppo_history_path = f"{output_dir}/ppo_results_{asset_slug}.csv"
    results['PPO']['history'].to_csv(ppo_history_path, index=False)
    if verbose:
        print(f"\n💾 Saved PPO history to: {ppo_history_path}")
    
    # Save agent comparison
    comparison_df = compare_agents(results)
    comparison_path = f"{output_dir}/agent_comparison_{asset_slug}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    if verbose:
        print(f"💾 Saved comparison to: {comparison_path}")
        print(f"\n📊 {ticker} Results:")
        print(comparison_df.to_string(index=False))
    
    return results


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Train PPO agent on TradingEnv (Step 3.1)')
    
    # Environment parameters
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
    
    # Step 3.1: Trading costs
    parser.add_argument('--transaction-cost', type=float, default=0.0,
                        help='Transaction cost percentage per trade (default: 0.0)')
    parser.add_argument('--switch-penalty', type=float, default=0.0,
                        help='Penalty for switching position direction (default: 0.0)')
    
    # Step 3.1: PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='PPO learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='PPO discount factor (default: 0.99)')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='PPO steps per update (default: 2048)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='PPO minibatch size (default: 64)')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='PPO entropy coefficient (default: 0.01)')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='PPO clipping parameter (default: 0.2)')
    
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
    
    # Print comprehensive configuration (Step 3.1)
    print(f"\n🔧 Configuration:")
    print(f"   Environment:")
    print(f"     - Use indicators: {use_indicators}")
    print(f"     - Window size: {args.window_size}")
    print(f"     - Transaction cost: {args.transaction_cost:.4f}")
    print(f"     - Switch penalty: {args.switch_penalty:.4f}")
    print(f"   PPO Hyperparameters:")
    print(f"     - Learning rate: {args.learning_rate}")
    print(f"     - Gamma: {args.gamma}")
    print(f"     - N-steps: {args.n_steps}")
    print(f"     - Batch size: {args.batch_size}")
    print(f"     - Entropy coef: {args.ent_coef}")
    print(f"     - Clip range: {args.clip_range}")
    
    # Training phase
    if not args.test:
        # Create training environment
        train_env = make_env(
            df=train_df, 
            window_size=args.window_size, 
            use_indicators=use_indicators,
            transaction_cost_pct=args.transaction_cost,
            switch_penalty=args.switch_penalty
        )
        
        # Train PPO agent
        model = train_ppo_agent(
            train_env=train_env,
            total_timesteps=args.timesteps,
            save_path=model_path,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range
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
        use_indicators=use_indicators,
        transaction_cost_pct=args.transaction_cost,
        switch_penalty=args.switch_penalty
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
