"""
Run Multi-Asset PPO Experiments (Step 3.2)

This script runs the same single-agent PPO training and evaluation
across multiple assets (tickers) and generates a multi-asset comparison table.

Usage:
    python run_multi_assets.py --tickers BTC-USD,SPY,QQQ --timesteps 20000
"""

import os
import argparse
from config.assets import DEFAULT_ASSETS, normalize_ticker_to_slug
from train_ppo import run_experiment_for_ticker


def run_multi_asset_experiments(
    tickers,
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
    skip_training=False
):
    """
    Run PPO experiments for multiple assets sequentially.
    
    Parameters:
    -----------
    tickers : list of str
        List of ticker symbols to process.
    timesteps : int
        Training timesteps per asset.
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
        Directory for saving results.
    skip_training : bool
        If True, skip training (evaluate existing models only).
    
    Returns:
    --------
    all_results : dict
        Dictionary mapping ticker -> results dict.
    """
    print("="*80)
    print("🚀 AlphaTraderLab - Multi-Asset PPO Experiments (Step 3.2)")
    print("="*80)
    print(f"\n📋 Assets to process: {', '.join(tickers)}")
    print(f"📊 Total: {len(tickers)} assets")
    print(f"⏱️  Timesteps per asset: {timesteps:,}")
    
    all_results = {}
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n\n{'='*80}")
        print(f"📈 Asset {i}/{len(tickers)}: {ticker}")
        print(f"{'='*80}")
        
        try:
            results = run_experiment_for_ticker(
                ticker=ticker,
                timesteps=timesteps,
                window_size=window_size,
                use_indicators=use_indicators,
                transaction_cost_pct=transaction_cost_pct,
                switch_penalty=switch_penalty,
                learning_rate=learning_rate,
                gamma=gamma,
                n_steps=n_steps,
                batch_size=batch_size,
                ent_coef=ent_coef,
                clip_range=clip_range,
                output_dir=output_dir,
                skip_training=skip_training,
                verbose=True
            )
            
            if results is not None:
                all_results[ticker] = results
                print(f"\n✅ {ticker} completed successfully")
            else:
                print(f"\n❌ {ticker} failed (skipping)")
        
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            print(f"   Continuing with next asset...")
            continue
    
    print(f"\n\n{'='*80}")
    print(f"✅ Multi-Asset Experiments Complete")
    print(f"{'='*80}")
    print(f"✅ Successfully processed: {len(all_results)}/{len(tickers)} assets")
    print(f"💾 Per-asset results saved in: {output_dir}/")
    
    return all_results


def main():
    """Main entry point for multi-asset experiments."""
    parser = argparse.ArgumentParser(
        description='Run multi-asset PPO experiments (Step 3.2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on default asset universe
  python run_multi_assets.py --timesteps 20000

  # Run on custom tickers
  python run_multi_assets.py --tickers BTC-USD,SPY,QQQ --timesteps 50000

  # With trading costs
  python run_multi_assets.py --tickers BTC-USD,SPY --timesteps 20000 \\
    --transaction-cost 0.001 --switch-penalty 0.0005
        """
    )
    
    # Asset selection
    parser.add_argument('--tickers', type=str, default=None,
                        help=f'Comma-separated list of tickers (default: {",".join(DEFAULT_ASSETS)})')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=100_000,
                        help='Training timesteps per asset (default: 100,000)')
    parser.add_argument('--window-size', type=int, default=30,
                        help='Observation window size (default: 30)')
    parser.add_argument('--use-indicators', action='store_true', default=True,
                        help='Use technical indicators (default: True)')
    parser.add_argument('--no-indicators', action='store_true',
                        help='Disable technical indicators')
    
    # Trading costs (Step 3.1)
    parser.add_argument('--transaction-cost', type=float, default=0.0,
                        help='Transaction cost percentage per trade (default: 0.0)')
    parser.add_argument('--switch-penalty', type=float, default=0.0,
                        help='Penalty for switching position direction (default: 0.0)')
    
    # PPO hyperparameters (Step 3.1)
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
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory for saving results (default: results)')
    parser.add_argument('--test', action='store_true',
                        help='Skip training, only evaluate existing models')
    
    args = parser.parse_args()
    
    # Parse tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    else:
        tickers = DEFAULT_ASSETS
    
    # Handle indicator flag
    use_indicators = args.use_indicators and not args.no_indicators
    
    # Run experiments
    all_results = run_multi_asset_experiments(
        tickers=tickers,
        timesteps=args.timesteps,
        window_size=args.window_size,
        use_indicators=use_indicators,
        transaction_cost_pct=args.transaction_cost,
        switch_penalty=args.switch_penalty,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        output_dir=args.output_dir,
        skip_training=args.test
    )
    
    # Generate multi-asset comparison table
    if all_results:
        print(f"\n{'='*80}")
        print("📊 Generating Multi-Asset Comparison Table...")
        print(f"{'='*80}")
        
        try:
            from utils.multi_asset_summary import generate_multi_asset_summary
            
            summary_path = generate_multi_asset_summary(
                results_dir=args.output_dir,
                tickers=tickers
            )
            
            print(f"\n✅ Multi-asset comparison saved to: {summary_path}")
            
        except Exception as e:
            print(f"\n⚠️  Warning: Could not generate multi-asset summary: {e}")
    
    print(f"\n{'='*80}")
    print("🎉 All done!")
    print(f"{'='*80}")
    print(f"\n💡 Next steps:")
    print(f"   - Check per-asset results in {args.output_dir}/agent_comparison_*.csv")
    print(f"   - Review multi-asset summary in {args.output_dir}/multi_asset_comparison.csv")


if __name__ == "__main__":
    main()
