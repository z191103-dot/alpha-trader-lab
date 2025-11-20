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
        NOTE: This is the legacy parameter. For Step 3.1, use transaction_cost_pct.
    
    transaction_cost_pct : float, default=0.0
        Transaction cost percentage applied per trade (0.001 = 0.1%).
        This is deducted from equity when position changes.
        Default 0.0 for backward compatibility.
    
    switch_penalty : float, default=0.0
        Penalty applied to reward when position flips direction (LONG↔SHORT).
        Helps discourage excessive back-and-forth trading.
        Default 0.0 for backward compatibility.
    
    use_indicators : bool, default=True
        Whether to include technical indicators in the observation.
        If True, adds: SMA_20, SMA_50, EMA_20, RSI_14, log_return, volatility_20,
                       MACD, Bollinger Bands, candlestick features (Step 3.1).
        If False, uses only OHLCV (backward compatible with Step 1).
    """
    
    # Define metadata for the environment (required by Gymnasium)
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df, window_size=30, initial_balance=10000.0, transaction_cost=0.001, 
                 transaction_cost_pct=0.0, switch_penalty=0.0, use_indicators=True):
        super(TradingEnv, self).__init__()
        
        # Store the market data
        self.df = df.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost  # Legacy parameter
        self.transaction_cost_pct = transaction_cost_pct  # Step 3.1 parameter
        self.switch_penalty = switch_penalty  # Step 3.1 parameter
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
        # Feature order per candle (when use_indicators=True - Step 3.1):
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
        #   [11] MACD Line (normalized)
        #   [12] MACD Signal (normalized)
        #   [13] MACD Histogram (standardized)
        #   [14] BB %B (position in bands, 0-1)
        #   [15] BB Width (normalized)
        #   [16] ATR_14 (normalized)
        #   [17] Body % (body/range)
        #   [18] Upper Wick % (upper_wick/range)
        #   [19] Lower Wick % (lower_wick/range)
        #   [20] Is Bullish (0 or 1)
        #   [21] Is Bearish (0 or 1)
        #   [22] Is Doji (0 or 1)
        
        features_per_candle = 23 if self.use_indicators else 5
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
        self.previous_position = None # Previous position (for switch penalty)
        self.entry_price = None       # Price at which we entered the current position
        self.balance = None           # Cash balance
        self.equity = None            # Total portfolio value (balance + unrealized P&L)
        self.episode_start = None
        self.episode_end = None
    
    def _compute_indicators(self):
        """
        Compute technical indicators and add them to the environment.
        
        Step 3 Indicators:
        - SMA_20, SMA_50: Simple Moving Averages
        - EMA_20: Exponential Moving Average
        - RSI_14: Relative Strength Index (14-period)
        - log_return: Daily log returns
        - volatility_20: 20-day rolling volatility of log returns
        
        Step 3.1 Additions:
        - MACD (12, 26, 9): MACD line, signal line, histogram
        - Bollinger Bands (20): %B (position in bands), band width
        - ATR_14: Average True Range (volatility measure)
        - Candlestick features: body %, wick %, pattern flags
        
        All indicators are normalized/scaled to be neural network friendly.
        """
        # Extract OHLC prices as 1D arrays
        if isinstance(self.df['Close'], pd.Series):
            close_prices = self.df['Close'].values.flatten()
            open_prices = self.df['Open'].values.flatten()
            high_prices = self.df['High'].values.flatten()
            low_prices = self.df['Low'].values.flatten()
        else:
            close_prices = np.array(self.df['Close']).flatten()
            open_prices = np.array(self.df['Open']).flatten()
            high_prices = np.array(self.df['High']).flatten()
            low_prices = np.array(self.df['Low']).flatten()
        
        # ========== Step 3 Indicators ==========
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
        
        # ========== Step 3.1 New Indicators ==========
        # 6. MACD (Moving Average Convergence Divergence)
        macd_result = self._compute_macd(close_prices, fast=12, slow=26, signal=9)
        macd_line = macd_result['macd']
        macd_signal = macd_result['signal']
        macd_hist = macd_result['histogram']
        
        # 7. Bollinger Bands
        bb_result = self._compute_bollinger_bands(close_prices, period=20, std_dev=2)
        bb_percent_b = bb_result['percent_b']
        bb_width = bb_result['width']
        
        # 8. ATR (Average True Range)
        atr_14 = self._compute_atr(high_prices, low_prices, close_prices, period=14)
        
        # 9. Candlestick Features
        candle_features = self._compute_candlestick_features(open_prices, high_prices, low_prices, close_prices)
        body_pct = candle_features['body_pct']
        upper_wick_pct = candle_features['upper_wick_pct']
        lower_wick_pct = candle_features['lower_wick_pct']
        is_bullish = candle_features['is_bullish']
        is_bearish = candle_features['is_bearish']
        is_doji = candle_features['is_doji']
        
        # ========== Normalize Indicators ==========
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
        
        # MACD: normalize by first close (price-relative)
        macd_line_norm = macd_line / first_close
        macd_signal_norm = macd_signal / first_close
        
        # MACD Histogram: standardize
        macd_hist_mean = np.nanmean(macd_hist)
        macd_hist_std = np.nanstd(macd_hist)
        if macd_hist_std > 0:
            macd_hist_norm = (macd_hist - macd_hist_mean) / macd_hist_std
        else:
            macd_hist_norm = macd_hist - macd_hist_mean
        
        # Bollinger %B: already 0-1 range (no normalization needed)
        # BB Width: normalize by price
        bb_width_norm = bb_width / first_close
        
        # ATR: normalize by price (relative to first close)
        atr_norm = atr_14 / first_close
        
        # Candlestick features: already normalized (percentages or binary)
        # No additional normalization needed
        
        # Store normalized indicators
        self.indicators = {
            # Step 3 indicators
            'sma_20': sma_20_norm,
            'sma_50': sma_50_norm,
            'ema_20': ema_20_norm,
            'rsi_14': rsi_scaled,
            'log_return': log_returns_norm,
            'volatility_20': volatility_norm,
            # Step 3.1 indicators
            'macd_line': macd_line_norm,
            'macd_signal': macd_signal_norm,
            'macd_hist': macd_hist_norm,
            'bb_percent_b': bb_percent_b,
            'bb_width': bb_width_norm,
            'atr_14': atr_norm,
            'body_pct': body_pct,
            'upper_wick_pct': upper_wick_pct,
            'lower_wick_pct': lower_wick_pct,
            'is_bullish': is_bullish,
            'is_bearish': is_bearish,
            'is_doji': is_doji
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
    
    def _compute_macd(self, prices, fast=12, slow=26, signal=9):
        """
        Compute MACD (Moving Average Convergence Divergence).
        
        Parameters:
        -----------
        prices : np.ndarray
            Price series (typically close prices).
        fast : int
            Fast EMA period (default 12).
        slow : int
            Slow EMA period (default 26).
        signal : int
            Signal line EMA period (default 9).
        
        Returns:
        --------
        dict with 'macd', 'signal', 'histogram' arrays.
        """
        # Compute fast and slow EMAs
        ema_fast = pd.Series(prices).ewm(span=fast, min_periods=fast, adjust=False).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow, min_periods=slow, adjust=False).mean().values
        
        # MACD line = fast EMA - slow EMA
        macd_line = ema_fast - ema_slow
        
        # Signal line = EMA of MACD line
        macd_signal = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
        
        # Histogram = MACD - Signal
        macd_hist = macd_line - macd_signal
        
        return {
            'macd': macd_line,
            'signal': macd_signal,
            'histogram': macd_hist
        }
    
    def _compute_bollinger_bands(self, prices, period=20, std_dev=2):
        """
        Compute Bollinger Bands and related metrics.
        
        Parameters:
        -----------
        prices : np.ndarray
            Price series (typically close prices).
        period : int
            Period for moving average (default 20).
        std_dev : float
            Number of standard deviations for bands (default 2).
        
        Returns:
        --------
        dict with 'percent_b' (position in bands, 0-1) and 'width' (band width).
        """
        # Compute middle band (SMA)
        middle_band = pd.Series(prices).rolling(window=period, min_periods=period).mean().values
        
        # Compute standard deviation
        rolling_std = pd.Series(prices).rolling(window=period, min_periods=period).std().values
        
        # Compute upper and lower bands
        upper_band = middle_band + (std_dev * rolling_std)
        lower_band = middle_band - (std_dev * rolling_std)
        
        # Compute %B (position within bands, 0 = at lower band, 1 = at upper band)
        band_range = upper_band - lower_band
        percent_b = np.where(band_range > 0, (prices - lower_band) / band_range, 0.5)
        
        # Clip to [0, 1] range (can go outside bands during high volatility)
        percent_b = np.clip(percent_b, 0.0, 1.0)
        
        # Compute band width (normalized by middle band)
        width = np.where(middle_band > 0, band_range, 0)
        
        return {
            'percent_b': percent_b,
            'width': width
        }
    
    def _compute_atr(self, high, low, close, period=14):
        """
        Compute Average True Range (ATR).
        
        Parameters:
        -----------
        high : np.ndarray
            High prices.
        low : np.ndarray
            Low prices.
        close : np.ndarray
            Close prices.
        period : int
            ATR period (default 14).
        
        Returns:
        --------
        atr : np.ndarray
            ATR values.
        """
        # Compute True Range
        # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        prev_close = np.concatenate([[close[0]], close[:-1]])
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR = EMA of True Range
        atr = pd.Series(true_range).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        return atr
    
    def _compute_candlestick_features(self, open_p, high, low, close):
        """
        Compute candlestick-derived features.
        
        Parameters:
        -----------
        open_p : np.ndarray
            Open prices.
        high : np.ndarray
            High prices.
        low : np.ndarray
            Low prices.
        close : np.ndarray
            Close prices.
        
        Returns:
        --------
        dict with candlestick features:
        - body_pct: Body size as fraction of range
        - upper_wick_pct: Upper wick as fraction of range
        - lower_wick_pct: Lower wick as fraction of range
        - is_bullish: Binary flag for bullish candles
        - is_bearish: Binary flag for bearish candles
        - is_doji: Binary flag for doji candles
        """
        # Compute range (high - low), avoid division by zero
        candle_range = high - low
        candle_range = np.where(candle_range > 0, candle_range, 1e-8)
        
        # Compute body (close - open)
        body = close - open_p
        body_abs = np.abs(body)
        
        # Body as percentage of range
        body_pct = body_abs / candle_range
        
        # Upper wick: distance from top of body to high
        upper_shadow = high - np.maximum(close, open_p)
        upper_wick_pct = upper_shadow / candle_range
        
        # Lower wick: distance from bottom of body to low
        lower_shadow = np.minimum(close, open_p) - low
        lower_wick_pct = lower_shadow / candle_range
        
        # Bullish: close > open and body is significant (>30% of range)
        is_bullish = ((body > 0) & (body_pct > 0.3)).astype(np.float32)
        
        # Bearish: close < open and body is significant (>30% of range)
        is_bearish = ((body < 0) & (body_pct > 0.3)).astype(np.float32)
        
        # Doji: small body (<10% of range)
        is_doji = (body_pct < 0.1).astype(np.float32)
        
        return {
            'body_pct': body_pct,
            'upper_wick_pct': upper_wick_pct,
            'lower_wick_pct': lower_wick_pct,
            'is_bullish': is_bullish,
            'is_bearish': is_bearish,
            'is_doji': is_doji
        }
        
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
        self.previous_position = 0  # For tracking position switches
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
        
        # Track if this is a position direction flip (LONG↔SHORT)
        is_direction_flip = False
        if position_changed:
            # Check if switching from LONG to SHORT or SHORT to LONG
            if (self.current_position == 1 and action == 2) or (self.current_position == 2 and action == 1):
                is_direction_flip = True
        
        # If changing position, close the old position and open the new one
        if position_changed:
            # Close old position if any
            if self.current_position != 0:  # We had a position
                pnl = self._calculate_pnl(current_price)
                self.balance += pnl
            
            # Apply legacy transaction cost (backward compatibility)
            if self.transaction_cost > 0:
                self.balance -= self.balance * self.transaction_cost
            
            # Apply Step 3.1 transaction cost
            if self.transaction_cost_pct > 0:
                cost = self.equity * self.transaction_cost_pct
                self.balance -= cost
            
            # Update position tracking
            self.previous_position = self.current_position
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
        
        # Apply switch penalty if direction flipped (LONG↔SHORT)
        if is_direction_flip and self.switch_penalty > 0:
            reward -= self.switch_penalty
        
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
            # Extract Step 3 indicators
            window_sma20 = self.indicators['sma_20'][start_idx:end_idx].reshape(-1, 1)
            window_sma50 = self.indicators['sma_50'][start_idx:end_idx].reshape(-1, 1)
            window_ema20 = self.indicators['ema_20'][start_idx:end_idx].reshape(-1, 1)
            window_rsi = self.indicators['rsi_14'][start_idx:end_idx].reshape(-1, 1)
            window_return = self.indicators['log_return'][start_idx:end_idx].reshape(-1, 1)
            window_vol = self.indicators['volatility_20'][start_idx:end_idx].reshape(-1, 1)
            
            # Extract Step 3.1 indicators
            window_macd_line = self.indicators['macd_line'][start_idx:end_idx].reshape(-1, 1)
            window_macd_signal = self.indicators['macd_signal'][start_idx:end_idx].reshape(-1, 1)
            window_macd_hist = self.indicators['macd_hist'][start_idx:end_idx].reshape(-1, 1)
            window_bb_pct_b = self.indicators['bb_percent_b'][start_idx:end_idx].reshape(-1, 1)
            window_bb_width = self.indicators['bb_width'][start_idx:end_idx].reshape(-1, 1)
            window_atr = self.indicators['atr_14'][start_idx:end_idx].reshape(-1, 1)
            window_body_pct = self.indicators['body_pct'][start_idx:end_idx].reshape(-1, 1)
            window_upper_wick = self.indicators['upper_wick_pct'][start_idx:end_idx].reshape(-1, 1)
            window_lower_wick = self.indicators['lower_wick_pct'][start_idx:end_idx].reshape(-1, 1)
            window_is_bullish = self.indicators['is_bullish'][start_idx:end_idx].reshape(-1, 1)
            window_is_bearish = self.indicators['is_bearish'][start_idx:end_idx].reshape(-1, 1)
            window_is_doji = self.indicators['is_doji'][start_idx:end_idx].reshape(-1, 1)
            
            # Replace NaN with appropriate defaults (safety check)
            window_sma20 = np.nan_to_num(window_sma20, nan=0.0)
            window_sma50 = np.nan_to_num(window_sma50, nan=0.0)
            window_ema20 = np.nan_to_num(window_ema20, nan=0.0)
            window_rsi = np.nan_to_num(window_rsi, nan=0.5)  # Default RSI to 50 (neutral)
            window_return = np.nan_to_num(window_return, nan=0.0)
            window_vol = np.nan_to_num(window_vol, nan=0.0)
            window_macd_line = np.nan_to_num(window_macd_line, nan=0.0)
            window_macd_signal = np.nan_to_num(window_macd_signal, nan=0.0)
            window_macd_hist = np.nan_to_num(window_macd_hist, nan=0.0)
            window_bb_pct_b = np.nan_to_num(window_bb_pct_b, nan=0.5)  # Default to middle of bands
            window_bb_width = np.nan_to_num(window_bb_width, nan=0.0)
            window_atr = np.nan_to_num(window_atr, nan=0.0)
            window_body_pct = np.nan_to_num(window_body_pct, nan=0.0)
            window_upper_wick = np.nan_to_num(window_upper_wick, nan=0.0)
            window_lower_wick = np.nan_to_num(window_lower_wick, nan=0.0)
            window_is_bullish = np.nan_to_num(window_is_bullish, nan=0.0)
            window_is_bearish = np.nan_to_num(window_is_bearish, nan=0.0)
            window_is_doji = np.nan_to_num(window_is_doji, nan=0.0)
            
            # Concatenate all features
            # Order: [OHLCV, Step 3 indicators, Step 3.1 indicators]
            window_features = np.column_stack([
                window_ohlcv,
                window_sma20,
                window_sma50,
                window_ema20,
                window_rsi,
                window_return,
                window_vol,
                window_macd_line,
                window_macd_signal,
                window_macd_hist,
                window_bb_pct_b,
                window_bb_width,
                window_atr,
                window_body_pct,
                window_upper_wick,
                window_lower_wick,
                window_is_bullish,
                window_is_bearish,
                window_is_doji
            ])  # Shape: (window_size, 23)
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
