"""
Plotting utilities for PPO results analysis (Step 3.3)

This module provides visualization tools for equity curves, drawdowns,
and comparative performance across agents.

Usage:
    python analysis/plots.py --assets BTC-USD,SPY,QQQ
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use non-interactive backend if no display available
matplotlib.use('Agg')


def plot_equity_curve(
    csv_path: str,
    asset_name: str,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot equity curve from PPO results CSV.
    
    Parameters:
    -----------
    csv_path : str
        Path to PPO results CSV
    asset_name : str
        Asset name for title
    save_path : str, optional
        Path to save figure (if None, doesn't save)
    show : bool
        Whether to display plot (default: False for headless)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if 'equity' not in df.columns or 'step' not in df.columns:
        raise ValueError("CSV must contain 'equity' and 'step' columns")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot equity curve
    ax.plot(df['step'], df['equity'], linewidth=2, label='PPO Agent', color='blue')
    
    # Add horizontal line at initial equity
    initial_equity = df['equity'].iloc[0]
    ax.axhline(y=initial_equity, color='gray', linestyle='--', alpha=0.5, label='Initial Equity')
    
    # Formatting
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Equity ($)', fontsize=12)
    ax.set_title(f'PPO Agent Equity Curve - {asset_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_multi_agent_comparison(
    asset_name: str,
    results_dir: str = "results",
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot equity curves comparing PPO vs other agents (if data available).
    
    Parameters:
    -----------
    asset_name : str
        Asset symbol (e.g., 'BTC-USD')
    results_dir : str
        Directory containing result files
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
    """
    from config.assets import normalize_ticker_to_slug
    
    asset_slug = normalize_ticker_to_slug(asset_name)
    ppo_path = os.path.join(results_dir, f"ppo_results_{asset_slug}.csv")
    
    if not os.path.exists(ppo_path):
        raise FileNotFoundError(f"PPO results not found: {ppo_path}")
    
    # Load PPO data
    ppo_df = pd.read_csv(ppo_path)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot PPO equity
    ax.plot(ppo_df['step'], ppo_df['equity'], linewidth=2.5, label='PPO Agent', color='blue')
    
    # Try to load other agent data (if exists)
    # Note: We'd need separate CSVs for random/buy&hold equity curves
    # For now, just plot PPO with reference lines
    
    initial_equity = ppo_df['equity'].iloc[0]
    final_equity = ppo_df['equity'].iloc[-1]
    
    # Add reference lines
    ax.axhline(y=initial_equity, color='gray', linestyle='--', alpha=0.5, label='Initial Equity')
    
    # Add markers for key points
    ax.scatter([0], [initial_equity], color='green', s=100, zorder=5, label='Start')
    ax.scatter([ppo_df['step'].iloc[-1]], [final_equity], color='red', s=100, zorder=5, label='End')
    
    # Formatting
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Equity ($)', fontsize=12)
    ax.set_title(f'Agent Performance Comparison - {asset_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_drawdown_curve(
    csv_path: str,
    asset_name: str,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot drawdown curve from equity data.
    
    Parameters:
    -----------
    csv_path : str
        Path to PPO results CSV
    asset_name : str
        Asset name for title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if 'equity' not in df.columns:
        raise ValueError("CSV must contain 'equity' column")
    
    # Compute drawdown
    equity = df['equity']
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot equity
    ax1.plot(df['step'], equity, linewidth=2, color='blue', label='Equity')
    ax1.plot(df['step'], running_max, linewidth=1.5, color='green', linestyle='--', alpha=0.7, label='Peak Equity')
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.set_title(f'Equity and Drawdown - {asset_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot drawdown
    ax2.fill_between(df['step'], drawdown, 0, where=(drawdown < 0), color='red', alpha=0.5, label='Drawdown')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_all_plots(
    assets: List[str],
    results_dir: str = "results",
    output_dir: str = "analysis/plots",
    show: bool = False
):
    """
    Create all plots for specified assets.
    
    Parameters:
    -----------
    assets : list of str
        List of asset symbols
    results_dir : str
        Directory containing result CSVs
    output_dir : str
        Directory to save plots
    show : bool
        Whether to display plots
    """
    from config.assets import normalize_ticker_to_slug
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Generating plots for {len(assets)} assets...")
    print(f"   Output directory: {output_dir}\n")
    
    for asset in assets:
        asset_slug = normalize_ticker_to_slug(asset)
        csv_path = os.path.join(results_dir, f"ppo_results_{asset_slug}.csv")
        
        if not os.path.exists(csv_path):
            print(f"⚠️  Skipping {asset} - CSV not found")
            continue
        
        print(f"📈 Creating plots for {asset}...")
        
        try:
            # 1. Equity curve
            equity_save_path = os.path.join(output_dir, f"equity_curve_{asset_slug}.png")
            plot_equity_curve(csv_path, asset, save_path=equity_save_path, show=show)
            
            # 2. Multi-agent comparison (if data available)
            comparison_save_path = os.path.join(output_dir, f"agent_comparison_{asset_slug}.png")
            plot_multi_agent_comparison(asset, results_dir, save_path=comparison_save_path, show=show)
            
            # 3. Drawdown curve
            drawdown_save_path = os.path.join(output_dir, f"drawdown_{asset_slug}.png")
            plot_drawdown_curve(csv_path, asset, save_path=drawdown_save_path, show=show)
            
            print(f"   ✅ Generated 3 plots for {asset}")
            
        except Exception as e:
            print(f"   ❌ Error creating plots for {asset}: {e}")
    
    print(f"\n✅ All plots saved to {output_dir}/\n")


def main():
    """Main entry point for plotting."""
    parser = argparse.ArgumentParser(
        description="Generate plots for PPO agent performance (Step 3.3)"
    )
    
    parser.add_argument(
        '--assets',
        type=str,
        default='BTC-USD,SPY,QQQ',
        help='Comma-separated list of assets (default: BTC-USD,SPY,QQQ)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing result CSVs (default: results)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis/plots',
        help='Directory to save plots (default: analysis/plots)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots (default: False, saves only)'
    )
    
    args = parser.parse_args()
    
    # Parse assets
    assets = [a.strip() for a in args.assets.split(',')]
    
    print("\n" + "="*80)
    print("📊 AlphaTraderLab - PPO Results Plotting (Step 3.3)")
    print("="*80)
    
    create_all_plots(
        assets=assets,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        show=args.show
    )


if __name__ == "__main__":
    main()
