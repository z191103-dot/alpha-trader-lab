"""
Multi-Asset Summary Utility (Step 3.2)

This module aggregates per-asset results into a multi-asset comparison table
and provides summary statistics across assets.
"""

import os
import pandas as pd
from typing import List


def generate_multi_asset_summary(results_dir: str, tickers: List[str]) -> str:
    """
    Generate multi-asset comparison table from per-asset results.
    
    This function scans per-asset agent_comparison CSV files and aggregates
    them into a single comparison table showing performance across all assets
    and agents (PPO, Buy & Hold, Random).
    
    Parameters:
    -----------
    results_dir : str
        Directory containing per-asset agent_comparison_*.csv files.
    tickers : list of str
        List of tickers to process.
    
    Returns:
    --------
    summary_path : str
        Path to the generated multi_asset_comparison.csv file.
    """
    from config.assets import normalize_ticker_to_slug
    
    all_data = []
    
    for ticker in tickers:
        asset_slug = normalize_ticker_to_slug(ticker)
        comparison_file = f"{results_dir}/agent_comparison_{asset_slug}.csv"
        
        if not os.path.exists(comparison_file):
            print(f"⚠️  Warning: {comparison_file} not found, skipping {ticker}")
            continue
        
        # Read per-asset comparison
        df = pd.read_csv(comparison_file)
        
        # Add asset column
        df['Asset'] = ticker
        
        # Reorder columns: Asset, Agent, metrics...
        cols = ['Asset'] + [col for col in df.columns if col != 'Asset']
        df = df[cols]
        
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No valid per-asset results found!")
    
    # Concatenate all results
    multi_asset_df = pd.concat(all_data, ignore_index=True)
    
    # Clean up column names (remove formatting artifacts if any)
    # The comparison CSV has formatted strings like "$10,000.00" and "50.00%"
    # We need to extract numeric values for proper sorting and analysis
    
    def clean_currency(val):
        """Extract numeric value from currency string like '$10,000.00'"""
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            return float(val.replace('$', '').replace(',', ''))
        return float(val)
    
    def clean_percentage(val):
        """Extract numeric value from percentage string like '50.00%'"""
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            return float(val.replace('%', ''))
        return float(val)
    
    def clean_number(val):
        """Extract numeric value from formatted number"""
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            return float(val.replace(',', ''))
        return float(val)
    
    # Create numeric columns for analysis
    if 'Final Equity' in multi_asset_df.columns:
        multi_asset_df['Final Equity (Numeric)'] = multi_asset_df['Final Equity'].apply(clean_currency)
    
    if 'Total Return (%)' in multi_asset_df.columns:
        multi_asset_df['Total Return (Numeric)'] = multi_asset_df['Total Return (%)'].apply(clean_percentage)
    
    if 'Max Drawdown (%)' in multi_asset_df.columns:
        multi_asset_df['Max Drawdown (Numeric)'] = multi_asset_df['Max Drawdown (%)'].apply(clean_percentage)
    
    if 'Sharpe Ratio' in multi_asset_df.columns:
        multi_asset_df['Sharpe Ratio (Numeric)'] = multi_asset_df['Sharpe Ratio'].apply(clean_number)
    
    if 'Volatility' in multi_asset_df.columns:
        multi_asset_df['Volatility (Numeric)'] = multi_asset_df['Volatility'].apply(clean_number)
    
    if 'Win Rate (%)' in multi_asset_df.columns:
        multi_asset_df['Win Rate (Numeric)'] = multi_asset_df['Win Rate (%)'].apply(clean_percentage)
    
    if 'Trades' in multi_asset_df.columns:
        multi_asset_df['Trades (Numeric)'] = multi_asset_df['Trades'].apply(clean_number)
    
    # Save multi-asset comparison
    summary_path = f"{results_dir}/multi_asset_comparison.csv"
    multi_asset_df.to_csv(summary_path, index=False)
    
    print(f"\n📊 Multi-Asset Comparison Table Generated")
    print(f"{'='*80}")
    print(multi_asset_df.to_string(index=False))
    
    # Generate summary insights
    print(f"\n{'='*80}")
    print("📈 Summary Insights")
    print(f"{'='*80}")
    
    # PPO performance across assets
    ppo_data = multi_asset_df[multi_asset_df['Agent'] == 'PPO'].copy()
    if not ppo_data.empty and 'Sharpe Ratio (Numeric)' in ppo_data.columns:
        ppo_sorted = ppo_data.sort_values('Sharpe Ratio (Numeric)', ascending=False)
        print(f"\n🏆 PPO Sharpe Ratio Ranking:")
        for idx, row in ppo_sorted.iterrows():
            print(f"   {row['Asset']:10s} - Sharpe: {row['Sharpe Ratio (Numeric)']:6.2f}, "
                  f"Return: {row['Total Return (Numeric)']:7.2f}%, "
                  f"Trades: {int(row.get('Trades (Numeric)', 0))}")
    
    # Compare PPO vs Buy & Hold
    print(f"\n📊 PPO vs Buy & Hold (by Total Return):")
    for asset in tickers:
        asset_data = multi_asset_df[multi_asset_df['Asset'] == asset]
        ppo_row = asset_data[asset_data['Agent'] == 'PPO']
        bh_row = asset_data[asset_data['Agent'] == 'Buy & Hold']
        
        if not ppo_row.empty and not bh_row.empty:
            ppo_return = ppo_row['Total Return (Numeric)'].values[0]
            bh_return = bh_row['Total Return (Numeric)'].values[0]
            diff = ppo_return - bh_return
            winner = "PPO" if diff > 0 else "Buy & Hold"
            symbol = "🟢" if diff > 0 else "🔴"
            print(f"   {symbol} {asset:10s} - PPO: {ppo_return:7.2f}%, "
                  f"B&H: {bh_return:7.2f}%, Diff: {diff:+7.2f}% ({winner} wins)")
    
    # Random baseline comparison
    print(f"\n🎲 PPO vs Random (by Total Return):")
    for asset in tickers:
        asset_data = multi_asset_df[multi_asset_df['Asset'] == asset]
        ppo_row = asset_data[asset_data['Agent'] == 'PPO']
        random_row = asset_data[asset_data['Agent'] == 'Random']
        
        if not ppo_row.empty and not random_row.empty:
            ppo_return = ppo_row['Total Return (Numeric)'].values[0]
            random_return = random_row['Total Return (Numeric)'].values[0]
            diff = ppo_return - random_return
            print(f"   ✅ {asset:10s} - PPO: {ppo_return:7.2f}%, "
                  f"Random: {random_return:7.2f}%, Diff: {diff:+7.2f}%")
    
    print(f"\n{'='*80}")
    
    return summary_path


def print_summary_statistics(summary_path: str):
    """
    Print aggregate statistics from multi-asset comparison.
    
    Parameters:
    -----------
    summary_path : str
        Path to multi_asset_comparison.csv.
    """
    df = pd.read_csv(summary_path)
    
    print(f"\n{'='*80}")
    print("📊 Aggregate Statistics")
    print(f"{'='*80}")
    
    # Group by agent
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        print(f"\n{agent}:")
        
        if 'Total Return (Numeric)' in agent_data.columns:
            mean_return = agent_data['Total Return (Numeric)'].mean()
            print(f"   Mean Return: {mean_return:.2f}%")
        
        if 'Sharpe Ratio (Numeric)' in agent_data.columns:
            mean_sharpe = agent_data['Sharpe Ratio (Numeric)'].mean()
            print(f"   Mean Sharpe: {mean_sharpe:.2f}")
        
        if 'Max Drawdown (Numeric)' in agent_data.columns:
            mean_dd = agent_data['Max Drawdown (Numeric)'].mean()
            print(f"   Mean Max Drawdown: {mean_dd:.2f}%")
        
        if 'Trades (Numeric)' in agent_data.columns:
            mean_trades = agent_data['Trades (Numeric)'].mean()
            print(f"   Mean Trades: {mean_trades:.1f}")
