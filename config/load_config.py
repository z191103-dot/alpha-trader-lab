"""
Configuration loader for PPO experiments.

Loads YAML configuration files and provides helper functions to extract
hyperparameters for Stable-Baselines3 PPO and TradingEnv.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_asset_symbol(config: Dict[str, Any]) -> str:
    """Extract asset symbol from config."""
    return config['asset']['symbol']


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training configuration (timesteps, seed, etc.)."""
    return config['training']


def get_ppo_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract PPO hyperparameters from config.
    
    Returns dictionary suitable for passing to PPO constructor:
    PPO(policy, env, **ppo_kwargs)
    """
    ppo_config = config['ppo']
    return {
        'learning_rate': ppo_config['learning_rate'],
        'gamma': ppo_config['gamma'],
        'n_steps': ppo_config['n_steps'],
        'batch_size': ppo_config['batch_size'],
        'ent_coef': ppo_config['ent_coef'],
        'clip_range': ppo_config['clip_range'],
        'verbose': 1
    }


def get_env_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract environment configuration from config.
    
    Returns dictionary suitable for passing to TradingEnv constructor.
    """
    env_config = config['env']
    return {
        'window_size': env_config['window_size'],
        'initial_balance': env_config['initial_balance'],
        'use_indicators': env_config['use_indicators'],
        'transaction_cost_pct': env_config['transaction_cost'],
        'switch_penalty': env_config['switch_penalty']
    }


def print_config_summary(config: Dict[str, Any]):
    """Print a human-readable summary of the configuration."""
    asset = config['asset']['symbol']
    training = config['training']
    ppo = config['ppo']
    env = config['env']
    
    print("\n" + "="*60)
    print(f"Configuration Summary: {asset}")
    print("="*60)
    print(f"\n[Asset]")
    print(f"  Symbol: {asset}")
    
    print(f"\n[Training]")
    print(f"  Total Timesteps: {training['total_timesteps']:,}")
    print(f"  Seed: {training['seed']}")
    
    print(f"\n[PPO Hyperparameters]")
    print(f"  Learning Rate: {ppo['learning_rate']}")
    print(f"  Gamma: {ppo['gamma']}")
    print(f"  N-Steps: {ppo['n_steps']}")
    print(f"  Batch Size: {ppo['batch_size']}")
    print(f"  Entropy Coef: {ppo['ent_coef']}")
    print(f"  Clip Range: {ppo['clip_range']}")
    
    print(f"\n[Environment]")
    print(f"  Window Size: {env['window_size']}")
    print(f"  Initial Balance: ${env['initial_balance']:,}")
    print(f"  Use Indicators: {env['use_indicators']}")
    print(f"  Transaction Cost: {env['transaction_cost']*100:.2f}%")
    print(f"  Switch Penalty: {env['switch_penalty']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the config loader
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_yaml_config(config_path)
        print_config_summary(config)
    else:
        print("Usage: python load_config.py <config_path>")
        print("\nExample:")
        print("  python config/load_config.py config/ppo_btc_usd.yaml")
