"""
Evaluation Utilities for AlphaTraderLab

This module provides functions to evaluate different trading agents:
- Trained RL agents (e.g., PPO)
- Random agents (baseline)
- Buy & Hold strategy (benchmark)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


def _calculate_risk_metrics(history_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate risk metrics from equity history.
    
    Parameters:
    -----------
    history_df : pandas.DataFrame
        DataFrame with 'equity' and 'reward' columns.
    
    Returns:
    --------
    metrics : dict
        Dictionary containing:
        - max_drawdown: Maximum peak-to-trough decline (%)
        - volatility: Standard deviation of returns
        - sharpe_ratio: Risk-adjusted return metric
    """
    equity = history_df['equity'].values
    
    # 1. Maximum Drawdown (%)
    # Find the running maximum (peak)
    running_max = np.maximum.accumulate(equity)
    # Calculate drawdown at each point
    drawdown = (equity - running_max) / running_max * 100
    max_drawdown = np.min(drawdown)  # Most negative value
    
    # 2. Volatility (std of returns)
    # Calculate simple returns (percentage change)
    returns = np.diff(equity) / equity[:-1]
    volatility = np.std(returns) if len(returns) > 1 else 0.0
    
    # 3. Sharpe Ratio
    # Annualized Sharpe = (mean_return / std_return) * sqrt(periods_per_year)
    # Assuming daily data, periods_per_year = 252
    if len(returns) > 1 and volatility > 0:
        mean_return = np.mean(returns)
        sharpe_ratio = (mean_return / volatility) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    
    return {
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio
    }


def _calculate_trade_stats(history_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate trade-level statistics.
    
    A trade is defined as a period from entering a position (0 -> 1/2)
    to exiting it (back to 0, or switching to opposite position).
    
    Parameters:
    -----------
    history_df : pandas.DataFrame
        DataFrame with 'position' and 'equity' columns.
    
    Returns:
    --------
    stats : dict
        Dictionary containing:
        - win_rate: Percentage of profitable trades
        - avg_win: Average return of winning trades (%)
        - avg_loss: Average return of losing trades (%)
        - profit_factor: Ratio of total wins to total losses
    """
    positions = history_df['position'].values
    equity = history_df['equity'].values
    
    # Detect trade boundaries
    # A trade starts when position changes from 0 to non-zero
    # A trade ends when position goes back to 0 or changes to different non-zero
    
    trades = []
    entry_equity = None
    in_trade = False
    
    for i in range(len(positions)):
        current_pos = positions[i]
        
        if not in_trade and current_pos != 0:
            # Entering a trade
            entry_equity = equity[i]
            in_trade = True
        elif in_trade and (current_pos == 0 or (i > 0 and positions[i] != positions[i-1])):
            # Exiting a trade (flat or position change)
            exit_equity = equity[i-1] if i > 0 else equity[i]
            if entry_equity is not None and entry_equity > 0:
                trade_return = ((exit_equity - entry_equity) / entry_equity) * 100
                trades.append(trade_return)
            
            # If switching positions, start new trade
            if current_pos != 0:
                entry_equity = equity[i]
            else:
                in_trade = False
                entry_equity = None
    
    # Close any open trade at the end
    if in_trade and entry_equity is not None:
        exit_equity = equity[-1]
        if entry_equity > 0:
            trade_return = ((exit_equity - entry_equity) / entry_equity) * 100
            trades.append(trade_return)
    
    # Calculate statistics
    if len(trades) == 0:
        return {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
    
    winning_trades = [t for t in trades if t > 0]
    losing_trades = [t for t in trades if t < 0]
    
    win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0.0
    avg_win = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = np.mean(losing_trades) if losing_trades else 0.0
    
    # Profit factor: total gains / total losses
    total_gains = sum(winning_trades) if winning_trades else 0.0
    total_losses = abs(sum(losing_trades)) if losing_trades else 0.0
    profit_factor = total_gains / total_losses if total_losses > 0 else 0.0
    
    return {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }


def evaluate_agent(env, model=None, n_episodes=1, deterministic=True, agent_type="trained"):
    """
    Evaluate an agent on the given environment.
    
    Parameters:
    -----------
    env : TradingEnv or VecEnv
        The environment to evaluate on.
    model : stable_baselines3 model, optional
        The trained model to evaluate. If None, uses random actions.
    n_episodes : int, default=1
        Number of episodes to run.
    deterministic : bool, default=True
        Whether to use deterministic actions (for trained agents).
    agent_type : str
        Type of agent: "trained", "random", or "buy_and_hold"
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'history': pandas.DataFrame with step-by-step data
        - 'final_equity': final portfolio value
        - 'total_return': return as a percentage
        - 'total_reward': sum of all rewards
        - 'n_trades': number of position changes
    """
    all_histories = []
    
    for episode in range(n_episodes):
        # Reset environment
        obs, info = env.reset()
        done = False
        
        # Storage for this episode
        history = {
            'step': [],
            'equity': [],
            'balance': [],
            'position': [],
            'action': [],
            'reward': []
        }
        
        step_count = 0
        
        while not done:
            # Choose action based on agent type
            if model is not None:
                # Trained agent
                action, _states = model.predict(obs, deterministic=deterministic)
                # Handle both single and vectorized envs
                if isinstance(action, np.ndarray):
                    action = action.item()
            elif agent_type == "buy_and_hold":
                # Buy and hold: go LONG at start, stay LONG
                action = 1  # LONG
            else:
                # Random agent
                action = env.action_space.sample()
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            
            # Record data
            history['step'].append(step_count)
            history['equity'].append(info.get('equity', env.equity if hasattr(env, 'equity') else 0))
            history['balance'].append(info.get('balance', env.balance if hasattr(env, 'balance') else 0))
            history['position'].append(info.get('position', env.current_position if hasattr(env, 'current_position') else 0))
            history['action'].append(action)
            history['reward'].append(reward)
            
            step_count += 1
            
            # Safety check
            if step_count > 10000:
                print(f"Warning: Episode {episode} exceeded 10000 steps. Breaking.")
                break
        
        all_histories.append(pd.DataFrame(history))
    
    # Combine all episodes (for n_episodes > 1, we'll use the first one)
    df = all_histories[0] if all_histories else pd.DataFrame()
    
    if len(df) == 0:
        return {
            'history': df,
            'final_equity': 0,
            'total_return': 0,
            'total_reward': 0,
            'n_trades': 0
        }
    
    # Calculate metrics
    initial_equity = df['equity'].iloc[0]
    final_equity = df['equity'].iloc[-1]
    total_return = ((final_equity - initial_equity) / initial_equity) * 100
    total_reward = df['reward'].sum()
    
    # Count position changes (trades)
    n_trades = (df['position'].diff() != 0).sum()
    
    # Calculate risk metrics
    risk_metrics = _calculate_risk_metrics(df)
    
    # Calculate trade-level statistics
    trade_stats = _calculate_trade_stats(df)
    
    return {
        'history': df,
        'final_equity': final_equity,
        'total_return': total_return,
        'total_reward': total_reward,
        'n_trades': n_trades,
        'initial_equity': initial_equity,
        **risk_metrics,
        **trade_stats
    }


def evaluate_random_agent(env, n_episodes=1):
    """
    Evaluate a random agent (baseline).
    
    The random agent chooses actions uniformly at random from the action space.
    This serves as a sanity check - any trained agent should beat this.
    
    Parameters:
    -----------
    env : TradingEnv
        The environment to evaluate on.
    n_episodes : int, default=1
        Number of episodes to run.
    
    Returns:
    --------
    results : dict
        Evaluation results (see evaluate_agent for details).
    """
    return evaluate_agent(env, model=None, n_episodes=n_episodes, agent_type="random")


def evaluate_buy_and_hold(env, n_episodes=1):
    """
    Evaluate a Buy & Hold strategy (benchmark).
    
    The Buy & Hold agent goes LONG at the start and holds until the end.
    This is a common baseline for trading strategies.
    
    Parameters:
    -----------
    env : TradingEnv
        The environment to evaluate on.
    n_episodes : int, default=1
        Number of episodes to run.
    
    Returns:
    --------
    results : dict
        Evaluation results (see evaluate_agent for details).
    """
    return evaluate_agent(env, model=None, n_episodes=n_episodes, agent_type="buy_and_hold")


def compare_agents(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a comparison table of different agents.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping agent names to their evaluation results.
        Example: {'PPO': ppo_results, 'Random': random_results, ...}
    
    Returns:
    --------
    comparison_df : pandas.DataFrame
        Comparison table with metrics for each agent.
    """
    comparison_data = []
    
    for agent_name, results in results_dict.items():
        comparison_data.append({
            'Agent': agent_name,
            'Final Equity': f"${results['final_equity']:,.2f}",
            'Total Return (%)': f"{results['total_return']:.2f}%",
            'Max Drawdown (%)': f"{results.get('max_drawdown', 0):.2f}%",
            'Sharpe Ratio': f"{results.get('sharpe_ratio', 0):.2f}",
            'Volatility': f"{results.get('volatility', 0):.4f}",
            'Win Rate (%)': f"{results.get('win_rate', 0):.1f}%",
            'Avg Win (%)': f"{results.get('avg_win', 0):.2f}%",
            'Avg Loss (%)': f"{results.get('avg_loss', 0):.2f}%",
            'Trades': results['n_trades']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by total return (descending)
    if len(df) > 0:
        # Extract numeric values for sorting
        df['_return_numeric'] = df['Total Return (%)'].str.replace('%', '').astype(float)
        df = df.sort_values('_return_numeric', ascending=False)
        df = df.drop('_return_numeric', axis=1)
    
    return df


def calculate_sharpe_ratio(rewards, risk_free_rate=0.0):
    """
    Calculate a simplified Sharpe ratio from rewards.
    
    Parameters:
    -----------
    rewards : array-like
        Array of rewards from episodes.
    risk_free_rate : float, default=0.0
        Risk-free rate (usually 0 for simplicity).
    
    Returns:
    --------
    sharpe : float
        Sharpe ratio (returns NaN if std is 0).
    """
    rewards = np.array(rewards)
    if len(rewards) == 0:
        return np.nan
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    if std_reward == 0:
        return np.nan
    
    sharpe = (mean_reward - risk_free_rate) / std_reward
    return sharpe
