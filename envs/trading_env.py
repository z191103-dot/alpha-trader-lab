"""
TradingEnv - A Gymnasium-Compatible Trading Environment

This is a simple trading environment for reinforcement learning agents.
It simulates trading a single asset (like BTC-USD) with three possible actions:
- FLAT (no position)
- LONG (betting the price will go up)
- SHORT (betting the price will go down)

The environment is designed to be easy to understand and extend.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    A simple trading environment for RL agents.
    
    This environment follows the Gymnasium API (step, reset, render, etc.)
    and allows an agent to learn trading strategies on historical market data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Historical OHLCV data with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        Index should be datetime.
    
    window_size : int, default=30
        Number of past candles to include in each observation.
        The agent will see the last N candles to make decisions.
    
    initial_balance : float, default=10000.0
        Starting portfolio value in USD (or your currency).
    
    transaction_cost : float, default=0.001
        Trading fee as a fraction (0.001 = 0.1% per trade).
        Applied when changing position (e.g., FLAT → LONG).
    """
    
    # Define metadata for the environment (required by Gymnasium)
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df, window_size=30, initial_balance=10000.0, transaction_cost=0.001):
        super(TradingEnv, self).__init__()
        
        # Store the market data
        self.df = df.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Validate that we have enough data
        assert len(self.df) > window_size, \
            f"Data length ({len(self.df)}) must be greater than window_size ({window_size})"
        
        # Extract OHLCV columns (we'll normalize them for the agent)
        self.prices = self.df[['Open', 'High', 'Low', 'Close']].values
        self.volumes = self.df['Volume'].values
        
        # Normalize prices relative to the first close price in the dataset
        # This helps the neural network learn better
        self.normalized_prices = self.prices / self.prices[0, 3]  # Divide by first close
        
        # Normalize volumes (simple min-max scaling)
        volume_min = self.volumes.min()
        volume_max = self.volumes.max()
        if volume_max > volume_min:
            self.normalized_volumes = (self.volumes - volume_min) / (volume_max - volume_min)
        else:
            self.normalized_volumes = np.zeros_like(self.volumes)
        
        # Define action space: 0 = FLAT, 1 = LONG, 2 = SHORT
        self.action_space = spaces.Discrete(3)
        
        # Define observation space:
        # The agent sees:
        #   - window_size candles, each with 5 features (OHLCV normalized)
        #   - current position (0=FLAT, 1=LONG, 2=SHORT)
        #   - current equity / initial_balance (portfolio performance)
        # Total shape: (window_size, 5) + 2 scalars = (window_size * 5 + 2,)
        obs_shape = (window_size * 5 + 2,)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=obs_shape, 
            dtype=np.float32
        )
        
        # Episode variables (will be set in reset())
        self.current_step = None
        self.current_position = None  # 0=FLAT, 1=LONG, 2=SHORT
        self.entry_price = None       # Price at which we entered the current position
        self.balance = None           # Cash balance
        self.equity = None            # Total portfolio value (balance + unrealized P&L)
        self.episode_start = None
        self.episode_end = None
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Returns:
        --------
        observation : np.ndarray
            The initial observation for the agent.
        info : dict
            Additional information (empty for now).
        """
        # Set random seed if provided (for reproducibility)
        super().reset(seed=seed)
        
        # Choose a random starting point in the data
        # We need at least window_size candles before, and some candles after
        min_episode_length = 50  # Minimum number of steps we want in an episode
        available_length = len(self.df) - self.window_size
        
        if available_length < min_episode_length:
            # Dataset is too small, just use what we have
            self.episode_start = self.window_size
            self.episode_end = len(self.df) - 1
        else:
            # Choose a random starting point with enough room for an episode
            max_episode_length = 500
            episode_length = min(max_episode_length, available_length)
            
            # Calculate valid range for episode start
            max_start_index = len(self.df) - episode_length - 1
            if max_start_index > self.window_size:
                self.episode_start = self.np_random.integers(self.window_size, max_start_index)
            else:
                self.episode_start = self.window_size
            
            self.episode_end = min(self.episode_start + episode_length, len(self.df) - 1)
        
        # Start at the beginning of the episode
        self.current_step = self.episode_start
        
        # Initialize portfolio
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.current_position = 0  # Start FLAT (no position)
        self.entry_price = 0.0
        
        # Return initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Execute one time step in the environment.
        
        Parameters:
        -----------
        action : int
            The action to take (0=FLAT, 1=LONG, 2=SHORT).
        
        Returns:
        --------
        observation : np.ndarray
            The new observation after taking the action.
        reward : float
            The reward for this step.
        done : bool
            Whether the episode has ended.
        truncated : bool
            Whether the episode was truncated (time limit).
        info : dict
            Additional information.
        """
        # Store the equity before taking action
        equity_before = self.equity
        
        # Get current and next price
        current_price = self.prices[self.current_step, 3]  # Close price
        
        # Check if we're changing position (this incurs transaction cost)
        position_changed = (action != self.current_position)
        
        # If changing position, close the old position and open the new one
        if position_changed:
            # Close old position if any
            if self.current_position != 0:  # We had a position
                pnl = self._calculate_pnl(current_price)
                self.balance += pnl
            
            # Apply transaction cost
            self.balance -= self.balance * self.transaction_cost
            
            # Open new position
            self.current_position = action
            self.entry_price = current_price
        
        # Move to the next time step
        self.current_step += 1
        
        # Calculate current equity (balance + unrealized P&L)
        if self.current_step < len(self.prices):
            next_price = self.prices[self.current_step, 3]
            unrealized_pnl = self._calculate_pnl(next_price)
            self.equity = self.balance + unrealized_pnl
        else:
            self.equity = self.balance
        
        # Calculate reward as the change in equity, normalized by initial balance
        # This makes the reward scale-invariant
        reward = (self.equity - equity_before) / self.initial_balance
        
        # Check if episode is done
        done = (self.current_step >= self.episode_end) or (self.equity <= 0)
        truncated = False  # We don't truncate episodes in this simple version
        
        # Get new observation
        observation = self._get_observation()
        
        # Additional info for debugging/logging
        info = {
            'equity': self.equity,
            'balance': self.balance,
            'position': self.current_position,
            'step': self.current_step
        }
        
        return observation, reward, done, truncated, info
    
    def _get_observation(self):
        """
        Construct the observation for the current state.
        
        The observation includes:
        - Last window_size candles (OHLCV, normalized)
        - Current position (0, 1, or 2)
        - Current equity ratio (equity / initial_balance)
        
        Returns:
        --------
        observation : np.ndarray
            Flattened observation vector.
        """
        # Get the window of past candles
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        # Extract OHLCV data for the window
        window_prices = self.normalized_prices[start_idx:end_idx]  # Shape: (window_size, 4)
        window_volumes = self.normalized_volumes[start_idx:end_idx]  # Shape: (window_size,)
        
        # Combine OHLCV into one array
        window_ohlcv = np.column_stack([
            window_prices,
            window_volumes.reshape(-1, 1)
        ])  # Shape: (window_size, 5)
        
        # Flatten the window
        flat_window = window_ohlcv.flatten()  # Shape: (window_size * 5,)
        
        # Add current position and equity ratio
        position_encoded = float(self.current_position)
        equity_ratio = self.equity / self.initial_balance
        
        # Concatenate everything
        observation = np.concatenate([
            flat_window,
            [position_encoded, equity_ratio]
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_pnl(self, current_price):
        """
        Calculate the profit/loss for the current position.
        
        Parameters:
        -----------
        current_price : float
            The current market price.
        
        Returns:
        --------
        pnl : float
            Profit or loss in USD.
        """
        if self.current_position == 0:  # FLAT
            return 0.0
        
        # Calculate price change
        price_change = current_price - self.entry_price
        
        if self.current_position == 1:  # LONG
            # We profit when price goes up
            pnl = (price_change / self.entry_price) * self.balance
        elif self.current_position == 2:  # SHORT
            # We profit when price goes down
            pnl = -(price_change / self.entry_price) * self.balance
        else:
            pnl = 0.0
        
        return pnl
    
    def render(self, mode='human'):
        """
        Render the environment (optional, for visualization).
        
        For now, we just print the current state.
        """
        if mode == 'human':
            print(f"Step: {self.current_step}, Equity: ${self.equity:.2f}, Position: {self.current_position}")
    
    def close(self):
        """
        Clean up resources (if any).
        """
        pass
