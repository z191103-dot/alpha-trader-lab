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
    
    use_indicators : bool, default=True
        Whether to include technical indicators in the observation.
        If True, adds: SMA_20, SMA_50, EMA_20, RSI_14, log_return, volatility_20.
        If False, uses only OHLCV (backward compatible with Step 1).
    """
    
    # Define metadata for the environment (required by Gymnasium)
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df, window_size=30, initial_balance=10000.0, transaction_cost=0.001, use_indicators=True):
        super(TradingEnv, self).__init__()
        
        # Store the market data
        self.df = df.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.use_indicators = use_indicators
        
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
        
        # Compute technical indicators if enabled
        if self.use_indicators:
            self._compute_indicators()
        else:
            self.indicators = None
            self.min_start_index = self.window_size
        
        # Define action space: 0 = FLAT, 1 = LONG, 2 = SHORT
        self.action_space = spaces.Discrete(3)
        
        # Define observation space:
        # The agent sees:
        #   - window_size candles, each with K features:
        #     * Without indicators (use_indicators=False): 5 features (OHLCV)
        #     * With indicators (use_indicators=True): 11 features (OHLCV + 6 indicators)
        #   - current position (0=FLAT, 1=LONG, 2=SHORT)
        #   - current equity / initial_balance (portfolio performance)
        #
        # Feature order per candle (when use_indicators=True):
        #   [0] Open (normalized)
        #   [1] High (normalized)
        #   [2] Low (normalized)
        #   [3] Close (normalized)
        #   [4] Volume (normalized)
        #   [5] SMA_20 (normalized)
        #   [6] SMA_50 (normalized)
        #   [7] EMA_20 (normalized)
        #   [8] RSI_14 (scaled 0-1)
        #   [9] Log Return (standardized)
        #   [10] Volatility_20 (standardized)
        
        features_per_candle = 11 if self.use_indicators else 5
        obs_shape = (window_size * features_per_candle + 2,)
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
    
    def _compute_indicators(self):
        """
        Compute technical indicators and add them to the environment.
        
        Indicators computed:
        - SMA_20, SMA_50: Simple Moving Averages
        - EMA_20: Exponential Moving Average
        - RSI_14: Relative Strength Index (14-period)
        - log_return: Daily log returns
        - volatility_20: 20-day rolling volatility of log returns
        
        All indicators are normalized/scaled to be neural network friendly.
        """
        # Extract close prices as 1D array
        if isinstance(self.df['Close'], pd.Series):
            close_prices = self.df['Close'].values.flatten()
        else:
            close_prices = np.array(self.df['Close']).flatten()
        
        # 1. Simple Moving Averages (SMA)
        sma_20 = pd.Series(close_prices).rolling(window=20, min_periods=20).mean().values
        sma_50 = pd.Series(close_prices).rolling(window=50, min_periods=50).mean().values
        
        # 2. Exponential Moving Average (EMA)
        ema_20 = pd.Series(close_prices).ewm(span=20, min_periods=20, adjust=False).mean().values
        
        # 3. RSI (Relative Strength Index)
        rsi_14 = self._compute_rsi(close_prices, period=14)
        
        # 4. Log Returns
        log_returns = np.log(close_prices[1:] / close_prices[:-1])
        log_returns = np.concatenate([[0], log_returns])  # Prepend 0 for first value
        
        # 5. Volatility (rolling std of log returns)
        volatility_20 = pd.Series(log_returns).rolling(window=20, min_periods=20).std().values
        
        # Normalize indicators
        # Moving averages: divide by first close (same as prices)
        first_close = close_prices[0]
        sma_20_norm = sma_20 / first_close
        sma_50_norm = sma_50 / first_close
        ema_20_norm = ema_20 / first_close
        
        # RSI: already in 0-100 range, scale to 0-1
        rsi_scaled = rsi_14 / 100.0
        
        # Log returns: standardize (mean=0, std=1)
        log_returns_mean = np.nanmean(log_returns)
        log_returns_std = np.nanstd(log_returns)
        if log_returns_std > 0:
            log_returns_norm = (log_returns - log_returns_mean) / log_returns_std
        else:
            log_returns_norm = log_returns - log_returns_mean
        
        # Volatility: standardize
        volatility_mean = np.nanmean(volatility_20)
        volatility_std = np.nanstd(volatility_20)
        if volatility_std > 0:
            volatility_norm = (volatility_20 - volatility_mean) / volatility_std
        else:
            volatility_norm = volatility_20 - volatility_mean
        
        # Store normalized indicators
        self.indicators = {
            'sma_20': sma_20_norm,
            'sma_50': sma_50_norm,
            'ema_20': ema_20_norm,
            'rsi_14': rsi_scaled,
            'log_return': log_returns_norm,
            'volatility_20': volatility_norm
        }
        
        # Determine minimum valid starting index
        # We need all indicators to be non-NaN from this point forward
        # SMA_50 has the longest warmup period (50 candles)
        self.min_start_index = max(50, self.window_size)
    
    def _compute_rsi(self, prices, period=14):
        """
        Compute Relative Strength Index (RSI).
        
        Parameters:
        -----------
        prices : np.ndarray
            Price series (typically close prices).
        period : int
            RSI period (typically 14).
        
        Returns:
        --------
        rsi : np.ndarray
            RSI values (0-100 range, NaN for warmup period).
        """
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses using exponential moving average
        avg_gains = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_losses = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        # Calculate RS and RSI
        rs = np.where(avg_losses != 0, avg_gains / avg_losses, 0)
        rsi = 100 - (100 / (1 + rs))
        
        # Prepend NaN for the first value (no delta)
        rsi = np.concatenate([[np.nan], rsi])
        
        return rsi
        
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
        # If using indicators, also need to skip the warmup period
        min_start = self.min_start_index if hasattr(self, 'min_start_index') else self.window_size
        
        min_episode_length = 50  # Minimum number of steps we want in an episode
        available_length = len(self.df) - min_start
        
        if available_length < min_episode_length:
            # Dataset is too small, just use what we have
            self.episode_start = min_start
            self.episode_end = len(self.df) - 1
        else:
            # Choose a random starting point with enough room for an episode
            max_episode_length = 500
            episode_length = min(max_episode_length, available_length)
            
            # Calculate valid range for episode start
            max_start_index = len(self.df) - episode_length - 1
            if max_start_index > min_start:
                self.episode_start = self.np_random.integers(min_start, max_start_index)
            else:
                self.episode_start = min_start
            
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
        - Last window_size candles with features:
          * Without indicators: OHLCV (5 features per candle)
          * With indicators: OHLCV + 6 indicators (11 features per candle)
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
        
        if self.use_indicators:
            # Add technical indicators to the window
            # Extract indicators for the window
            window_sma20 = self.indicators['sma_20'][start_idx:end_idx].reshape(-1, 1)
            window_sma50 = self.indicators['sma_50'][start_idx:end_idx].reshape(-1, 1)
            window_ema20 = self.indicators['ema_20'][start_idx:end_idx].reshape(-1, 1)
            window_rsi = self.indicators['rsi_14'][start_idx:end_idx].reshape(-1, 1)
            window_return = self.indicators['log_return'][start_idx:end_idx].reshape(-1, 1)
            window_vol = self.indicators['volatility_20'][start_idx:end_idx].reshape(-1, 1)
            
            # Replace NaN with 0 (shouldn't happen if min_start_index is set correctly, but safety check)
            window_sma20 = np.nan_to_num(window_sma20, nan=0.0)
            window_sma50 = np.nan_to_num(window_sma50, nan=0.0)
            window_ema20 = np.nan_to_num(window_ema20, nan=0.0)
            window_rsi = np.nan_to_num(window_rsi, nan=0.5)  # Default RSI to 50 (neutral)
            window_return = np.nan_to_num(window_return, nan=0.0)
            window_vol = np.nan_to_num(window_vol, nan=0.0)
            
            # Concatenate all features
            # Order: [Open, High, Low, Close, Volume, SMA20, SMA50, EMA20, RSI14, LogReturn, Vol20]
            window_features = np.column_stack([
                window_ohlcv,
                window_sma20,
                window_sma50,
                window_ema20,
                window_rsi,
                window_return,
                window_vol
            ])  # Shape: (window_size, 11)
        else:
            window_features = window_ohlcv  # Shape: (window_size, 5)
        
        # Flatten the window
        flat_window = window_features.flatten()
        
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
