"""
PPO Results Analysis Pipeline (Step 3.3)

This module provides automated analysis of PPO agent performance from CSV outputs.
It computes key metrics, compares agents, and generates summary tables.

Usage:
    python analysis/analyze_ppo_results.py
    python analysis/analyze_ppo_results.py --results-dir results --assets BTC-USD,SPY,QQQ
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_trades_from_position(df: pd.DataFrame) -> Tuple[int, List[float]]:
    """
    Count trades and compute per-trade returns from position changes.
    
    A trade is defined as:
    - Entry: position changes from 0 to non-zero (1 for LONG, 2 for SHORT)
    - Exit: position changes back to 0 or flips direction
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: 'step', 'equity', 'position'
    
    Returns:
    --------
    num_trades : int
        Total number of completed trades
    trade_returns : list of float
        List of returns (%) for each completed trade
    """
    if len(df) == 0 or 'position' not in df.columns or 'equity' not in df.columns:
        return 0, []
    
    positions = df['position'].values
    equity = df['equity'].values
    
    trades = []
    entry_equity = None
    in_trade = False
    
    for i in range(len(positions)):
        current_pos = positions[i]
        prev_pos = positions[i-1] if i > 0 else 0
        
        # Trade entry: 0 → 1 or 0 → 2
        if not in_trade and current_pos != 0 and prev_pos == 0:
            entry_equity = equity[i]
            in_trade = True
        
        # Trade exit: position changes (includes flips like 1→2 or 2→1)
        elif in_trade and current_pos != prev_pos:
            exit_equity = equity[i-1] if i > 0 else equity[i]
            if entry_equity is not None and entry_equity > 0:
                trade_return = ((exit_equity - entry_equity) / entry_equity) * 100
                trades.append(trade_return)
            
            # If switching to another position (not flat), start new trade
            if current_pos != 0:
                entry_equity = equity[i]
                in_trade = True
            else:
                in_trade = False
                entry_equity = None
    
    # Handle last trade if still open at end
    if in_trade and entry_equity is not None and len(equity) > 0:
        exit_equity = equity[-1]
        if entry_equity > 0:
            trade_return = ((exit_equity - entry_equity) / entry_equity) * 100
            trades.append(trade_return)
    
    return len(trades), trades


def compute_max_drawdown(equity_series: pd.Series) -> float:
    """
    Compute maximum drawdown from equity curve.
    
    Parameters:
    -----------
    equity_series : pd.Series
        Series of equity values over time
    
    Returns:
    --------
    max_dd : float
        Maximum drawdown as a percentage (negative value)
    """
    if len(equity_series) == 0:
        return 0.0
    
    # Compute running maximum
    running_max = equity_series.expanding().max()
    
    # Compute drawdown at each point
    drawdown = (equity_series - running_max) / running_max * 100
    
    # Return the most negative drawdown
    return drawdown.min()


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio from returns series.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of period returns (e.g., daily returns)
    risk_free_rate : float
        Risk-free rate (default: 0.0)
    
    Returns:
    --------
    sharpe : float
        Sharpe ratio (annualized)
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    
    if std_return == 0 or np.isnan(std_return):
        return 0.0
    
    # Annualize assuming ~252 trading days per year
    sharpe = (mean_return / std_return) * np.sqrt(252)
    
    return sharpe


def analyze_ppo_results(csv_path: str) -> Dict[str, float]:
    """
    Analyze a single PPO results CSV and compute performance metrics.
    
    Parameters:
    -----------
    csv_path : str
        Path to PPO results CSV file
    
    Returns:
    --------
    metrics : dict
        Dictionary of computed metrics:
        - total_return: Cumulative return (%)
        - avg_daily_return: Mean daily return (%)
        - volatility: Std of daily returns
        - sharpe_ratio: Risk-adjusted return
        - max_drawdown: Maximum peak-to-trough decline (%)
        - num_trades: Total number of trades
        - win_rate: Percentage of winning trades (%)
        - avg_trade_return: Average trade return (%)
        - avg_win: Average winning trade (%)
        - avg_loss: Average losing trade (%)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['step', 'equity', 'position']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV missing required columns. Found: {df.columns.tolist()}")
    
    # Initialize metrics
    metrics = {}
    
    # 1. Total return (first to last equity)
    if len(df) > 0:
        initial_equity = df['equity'].iloc[0]
        final_equity = df['equity'].iloc[-1]
        if initial_equity > 0:
            metrics['total_return'] = ((final_equity - initial_equity) / initial_equity) * 100
        else:
            metrics['total_return'] = 0.0
        metrics['initial_equity'] = initial_equity
        metrics['final_equity'] = final_equity
    else:
        metrics['total_return'] = 0.0
        metrics['initial_equity'] = 0.0
        metrics['final_equity'] = 0.0
    
    # 2. Daily returns (compute from equity if not present)
    if 'reward' in df.columns:
        # Use rewards as proxy for returns
        daily_returns = df['reward']
    else:
        # Compute from equity changes
        daily_returns = df['equity'].pct_change().fillna(0) * 100
    
    metrics['avg_daily_return'] = daily_returns.mean()
    metrics['volatility'] = daily_returns.std()
    
    # 3. Sharpe ratio
    metrics['sharpe_ratio'] = compute_sharpe_ratio(daily_returns)
    
    # 4. Maximum drawdown
    metrics['max_drawdown'] = compute_max_drawdown(df['equity'])
    
    # 5. Trade analysis
    num_trades, trade_returns = compute_trades_from_position(df)
    metrics['num_trades'] = num_trades
    
    if len(trade_returns) > 0:
        metrics['avg_trade_return'] = np.mean(trade_returns)
        
        # Win rate and average win/loss
        winning_trades = [t for t in trade_returns if t > 0]
        losing_trades = [t for t in trade_returns if t < 0]
        
        metrics['win_rate'] = (len(winning_trades) / len(trade_returns)) * 100 if len(trade_returns) > 0 else 0.0
        metrics['avg_win'] = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
        metrics['avg_loss'] = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
    else:
        metrics['avg_trade_return'] = 0.0
        metrics['win_rate'] = 0.0
        metrics['avg_win'] = 0.0
        metrics['avg_loss'] = 0.0
    
    return metrics


def analyze_multiple_assets(
    assets: List[str],
    results_dir: str = "results",
    file_pattern: str = "ppo_results_{asset}.csv"
) -> pd.DataFrame:
    """
    Analyze PPO results for multiple assets and create summary table.
    
    Parameters:
    -----------
    assets : list of str
        List of asset symbols (e.g., ['BTC-USD', 'SPY', 'QQQ'])
    results_dir : str
        Directory containing result CSV files
    file_pattern : str
        Filename pattern with {asset} placeholder
    
    Returns:
    --------
    summary_df : pd.DataFrame
        DataFrame with one row per asset, columns for each metric
    """
    from config.assets import normalize_ticker_to_slug
    
    results = []
    
    for asset in assets:
        asset_slug = normalize_ticker_to_slug(asset)
        csv_path = os.path.join(results_dir, file_pattern.format(asset=asset_slug))
        
        if not os.path.exists(csv_path):
            print(f"⚠️  Warning: {csv_path} not found, skipping {asset}")
            continue
        
        print(f"📊 Analyzing {asset}...")
        
        try:
            metrics = analyze_ppo_results(csv_path)
            metrics['asset'] = asset
            results.append(metrics)
        except Exception as e:
            print(f"❌ Error analyzing {asset}: {e}")
            continue
    
    if not results:
        raise ValueError("No valid results found for any asset")
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(results)
    
    # Reorder columns
    cols_order = [
        'asset', 'total_return', 'avg_daily_return', 'volatility', 
        'sharpe_ratio', 'max_drawdown', 'num_trades', 'win_rate',
        'avg_trade_return', 'avg_win', 'avg_loss', 
        'initial_equity', 'final_equity'
    ]
    
    # Only include columns that exist
    cols_order = [col for col in cols_order if col in summary_df.columns]
    summary_df = summary_df[cols_order]
    
    return summary_df


def compare_agents_from_csv(asset: str, results_dir: str = "results", file_pattern: str = "agent_comparison_{asset}.csv") -> Optional[pd.DataFrame]:
    """
    Load agent comparison CSV if it exists.
    
    Parameters:
    -----------
    asset : str
        Asset symbol
    results_dir : str
        Directory containing result files
    file_pattern : str
        Filename pattern with {asset} placeholder
    
    Returns:
    --------
    comparison_df : pd.DataFrame or None
        Agent comparison data if file exists
    """
    from config.assets import normalize_ticker_to_slug
    
    asset_slug = normalize_ticker_to_slug(asset)
    comparison_path = os.path.join(results_dir, file_pattern.format(asset=asset_slug))
    
    if not os.path.exists(comparison_path):
        return None
    
    return pd.read_csv(comparison_path)


def print_summary_table(summary_df: pd.DataFrame):
    """
    Print formatted summary table to console.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary DataFrame from analyze_multiple_assets()
    """
    print("\n" + "="*100)
    print("📊 PPO AGENT PERFORMANCE SUMMARY")
    print("="*100)
    
    # Format numeric columns for display
    display_df = summary_df.copy()
    
    format_specs = {
        'total_return': '{:.2f}%',
        'avg_daily_return': '{:.4f}%',
        'volatility': '{:.4f}',
        'sharpe_ratio': '{:.2f}',
        'max_drawdown': '{:.2f}%',
        'num_trades': '{:.0f}',
        'win_rate': '{:.1f}%',
        'avg_trade_return': '{:.2f}%',
        'avg_win': '{:.2f}%',
        'avg_loss': '{:.2f}%',
        'initial_equity': '${:,.2f}',
        'final_equity': '${:,.2f}'
    }
    
    for col, fmt in format_specs.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: fmt.format(x))
    
    print(display_df.to_string(index=False))
    print("="*100 + "\n")


def print_agent_comparisons(assets: List[str], results_dir: str = "results", file_pattern: str = "agent_comparison_{asset}.csv"):
    """
    Print agent comparison tables for each asset.
    
    Parameters:
    -----------
    assets : list of str
        List of asset symbols
    results_dir : str
        Directory containing result files
    file_pattern : str
        Filename pattern with {asset} placeholder
    """
    print("\n" + "="*100)
    print("📊 AGENT COMPARISONS (PPO vs Buy & Hold vs Random)")
    print("="*100)
    
    for asset in assets:
        comparison_df = compare_agents_from_csv(asset, results_dir, file_pattern)
        
        if comparison_df is None:
            print(f"\n⚠️  No comparison data found for {asset}")
            continue
        
        print(f"\n{'─'*100}")
        print(f"Asset: {asset}")
        print(f"{'─'*100}")
        print(comparison_df.to_string(index=False))
    
    print("\n" + "="*100 + "\n")


def main():
    """Main entry point for PPO results analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze PPO agent performance from CSV outputs (Step 3.3)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing result CSV files (default: results)'
    )
    
    parser.add_argument(
        '--assets',
        type=str,
        default='BTC-USD,SPY,QQQ',
        help='Comma-separated list of assets to analyze (default: BTC-USD,SPY,QQQ)'
    )
    
    parser.add_argument(
        '--show-comparisons',
        action='store_true',
        default=True,
        help='Show agent comparison tables (default: True)'
    )
    
    parser.add_argument(
        '--suffix',
        type=str,
        default='',
        help='Suffix for result files (e.g., "_v2") (default: none)'
    )
    
    args = parser.parse_args()
    
    # Parse assets
    assets = [a.strip() for a in args.assets.split(',')]
    
    print("\n" + "="*100)
    print("🚀 AlphaTraderLab - PPO Results Analysis (Step 3.3)")
    print("="*100)
    print(f"\n📁 Results directory: {args.results_dir}")
    print(f"📊 Assets to analyze: {', '.join(assets)}\n")
    
    # Analyze PPO results for each asset
    try:
        # Update file pattern with suffix
        file_pattern = f"ppo_results_{{asset}}{args.suffix}.csv"
        summary_df = analyze_multiple_assets(assets, results_dir=args.results_dir, file_pattern=file_pattern)
        print_summary_table(summary_df)
        
        # Print agent comparisons if available
        if args.show_comparisons:
            comp_pattern = f"agent_comparison_{{asset}}{args.suffix}.csv"
            print_agent_comparisons(assets, results_dir=args.results_dir, file_pattern=comp_pattern)
        
        # Print key insights
        print("\n" + "="*100)
        print("💡 KEY INSIGHTS")
        print("="*100)
        
        for _, row in summary_df.iterrows():
            asset = row['asset']
            total_return = row['total_return']
            sharpe = row['sharpe_ratio']
            num_trades = row['num_trades']
            win_rate = row['win_rate']
            
            print(f"\n{asset}:")
            print(f"  • Total Return: {total_return:.2f}%")
            print(f"  • Sharpe Ratio: {sharpe:.2f}")
            print(f"  • Number of Trades: {int(num_trades)}")
            print(f"  • Win Rate: {win_rate:.1f}%")
            
            # Interpretation
            if total_return > 50:
                print(f"  ✅ Strong positive returns")
            elif total_return > 0:
                print(f"  ⚠️  Modest positive returns")
            else:
                print(f"  ❌ Negative returns - needs improvement")
            
            if sharpe > 1.0:
                print(f"  ✅ Good risk-adjusted performance")
            elif sharpe > 0:
                print(f"  ⚠️  Weak risk-adjusted performance")
            else:
                print(f"  ❌ Poor risk-adjusted performance")
        
        print("\n" + "="*100 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
