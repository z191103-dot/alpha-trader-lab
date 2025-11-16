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
    
    return {
        'history': df,
        'final_equity': final_equity,
        'total_return': total_return,
        'total_reward': total_reward,
        'n_trades': n_trades,
        'initial_equity': initial_equity
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
            'Initial Equity': f"${results.get('initial_equity', 0):,.2f}",
            'Final Equity': f"${results['final_equity']:,.2f}",
            'Total Return (%)': f"{results['total_return']:.2f}%",
            'Total Reward': f"{results['total_reward']:.4f}",
            'Number of Trades': results['n_trades']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by final equity (descending)
    if len(df) > 0:
        # Extract numeric values for sorting
        df['_final_equity_numeric'] = df['Final Equity'].str.replace('$', '').str.replace(',', '').astype(float)
        df = df.sort_values('_final_equity_numeric', ascending=False)
        df = df.drop('_final_equity_numeric', axis=1)
    
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
