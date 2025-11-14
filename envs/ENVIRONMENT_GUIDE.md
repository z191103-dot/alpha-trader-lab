# 📘 TradingEnv - Developer Guide

This guide explains how the `TradingEnv` class works and how to use it effectively.

---

## 🎯 What is TradingEnv?

`TradingEnv` is a custom Gymnasium environment that simulates trading a single financial asset (like Bitcoin) with three possible positions:
- **FLAT** (0): No position, holding cash
- **LONG** (1): Betting the price will go up
- **SHORT** (2): Betting the price will go down

---

## 🔧 Basic Usage

```python
import pandas as pd
from envs.trading_env import TradingEnv

# Assume df is a pandas DataFrame with OHLCV data
env = TradingEnv(
    df=df,                    # Your historical data
    window_size=30,           # Agent sees last 30 candles
    initial_balance=10000.0,  # Starting with $10,000
    transaction_cost=0.001    # 0.1% fee per trade
)

# Reset to start a new episode
observation, info = env.reset()

# Take an action
action = 1  # Go LONG
observation, reward, done, truncated, info = env.step(action)
```

---

## 📊 Input Data Format

Your DataFrame must have these columns:
- `Open`: Opening price
- `High`: Highest price in the period
- `Low`: Lowest price in the period
- `Close`: Closing price
- `Volume`: Trading volume

The index should be a datetime index.

**Example:**
```
                 Open     High      Low    Close      Volume
Date                                                         
2023-01-01  16547.00  16625.0  16463.0  16625.00  15302000000
2023-01-02  16625.00  16820.0  16547.0  16688.00  11787000000
```

---

## 🎮 Action Space

```python
env.action_space  # Discrete(3)
```

| Action | Name  | Meaning                           |
|--------|-------|-----------------------------------|
| 0      | FLAT  | Close any position, hold cash    |
| 1      | LONG  | Buy/hold, profit when price ↑     |
| 2      | SHORT | Sell/short, profit when price ↓   |

---

## 👁️ Observation Space

```python
env.observation_space  # Box(shape=(window_size * 5 + 2,))
```

The observation is a flat vector containing:

1. **Market Data (window_size × 5 values)**:
   - Last N candles of OHLCV data, normalized
   - For window_size=30: this is 150 values

2. **Portfolio State (2 values)**:
   - Current position (0, 1, or 2)
   - Equity ratio (current_equity / initial_balance)

**Example for window_size=30:**
```
Observation shape: (152,)
├─ [0:150]   → 30 candles × 5 features (OHLCV)
├─ [150]     → Current position
└─ [151]     → Equity ratio
```

---

## 💰 Reward Function

The reward at each step is:

```
reward = (equity_after - equity_before) / initial_balance
```

**Components:**
- **Equity**: Total portfolio value = cash balance + unrealized P&L
- **Unrealized P&L**: Profit/loss from current open position
- **Transaction Cost**: Deducted when changing positions

**Example:**
```
Initial balance: $10,000
Equity before action: $10,500
Equity after action: $10,600
Reward: (10,600 - 10,500) / 10,000 = 0.01
```

---

## 📈 How P&L is Calculated

### LONG Position
When you're LONG, you profit when price goes up:
```
pnl = (current_price - entry_price) / entry_price * balance
```

**Example:**
- Entry price: $20,000
- Current price: $21,000
- Balance: $10,000
- P&L: (21,000 - 20,000) / 20,000 × 10,000 = $500

### SHORT Position
When you're SHORT, you profit when price goes down:
```
pnl = -(current_price - entry_price) / entry_price * balance
```

**Example:**
- Entry price: $20,000
- Current price: $19,000
- Balance: $10,000
- P&L: -(19,000 - 20,000) / 20,000 × 10,000 = $500

### FLAT Position
When FLAT, P&L is 0 (you're in cash).

---

## 🔄 Episode Lifecycle

1. **reset()**: Start a new episode
   - Chooses a random time window from the data
   - Resets balance to initial_balance
   - Sets position to FLAT
   - Returns initial observation

2. **step(action)**: Execute one time step
   - If action differs from current position:
     - Close old position (realize P&L)
     - Apply transaction cost
     - Open new position
   - Move forward one time step
   - Calculate new equity and reward
   - Return (obs, reward, done, truncated, info)

3. **Episode ends when**:
   - Reached the end of the time window (max 500 steps)
   - Portfolio value drops to $0 or below

---

## 🎛️ Configuration Parameters

### window_size
- **Type**: int
- **Default**: 30
- **Description**: Number of past candles the agent observes
- **Recommendation**: 
  - Smaller (10-20): Agent reacts faster, less context
  - Larger (50-100): More context, slower reactions

### initial_balance
- **Type**: float
- **Default**: 10000.0
- **Description**: Starting portfolio value in USD
- **Recommendation**: 
  - Use realistic amounts ($1,000 - $100,000)
  - Affects the scale of rewards

### transaction_cost
- **Type**: float
- **Default**: 0.001
- **Description**: Trading fee as a fraction (0.001 = 0.1%)
- **Recommendation**: 
  - Crypto exchanges: 0.001 - 0.002 (0.1% - 0.2%)
  - Stock brokers: 0.0001 - 0.001 (0.01% - 0.1%)
  - Zero cost: 0.0 (for testing only)

---

## 📊 Info Dictionary

The `info` dict returned by `step()` contains:

```python
{
    'equity': 10523.45,      # Current portfolio value
    'balance': 10000.00,     # Cash balance
    'position': 1,           # Current position (0/1/2)
    'step': 142              # Current step number
}
```

Use this for logging and debugging.

---

## 🧪 Testing Your Environment

Quick test script:
```python
# Create environment
env = TradingEnv(df, window_size=30, initial_balance=10000)

# Reset
obs, info = env.reset(seed=42)
print(f"Observation shape: {obs.shape}")
print(f"Initial equity: ${env.equity:.2f}")

# Take 10 random steps
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Equity=${info['equity']:.2f}")
    if done:
        break
```

---

## 🔍 Debugging Tips

### Check observation shape
```python
obs, _ = env.reset()
expected_shape = (env.window_size * 5 + 2,)
assert obs.shape == expected_shape, f"Shape mismatch: {obs.shape} vs {expected_shape}"
```

### Verify equity calculations
```python
env.reset()
initial = env.equity
obs, reward, done, truncated, info = env.step(0)  # FLAT action
# Equity should stay the same (minus small transaction cost if changing position)
```

### Track position changes
```python
position_history = []
for _ in range(100):
    action = env.action_space.sample()
    _, _, done, _, info = env.step(action)
    position_history.append(info['position'])
    if done:
        break

# Count position changes
changes = sum(1 for i in range(1, len(position_history)) 
              if position_history[i] != position_history[i-1])
print(f"Position changed {changes} times")
```

---

## 🚀 Advanced Customization

Want to modify the environment? Here are common customizations:

### 1. Add Technical Indicators
Extend the observation with RSI, MACD, etc.:
```python
# In _get_observation()
rsi = calculate_rsi(self.prices, window=14)
observation = np.concatenate([flat_window, [position_encoded, equity_ratio, rsi]])
```

### 2. Multi-Asset Trading
Support multiple assets:
```python
# Store multiple DataFrames
self.dfs = {'BTC': btc_df, 'ETH': eth_df}
# Extend action space: FLAT, LONG_BTC, SHORT_BTC, LONG_ETH, SHORT_ETH
self.action_space = spaces.Discrete(5)
```

### 3. Different Reward Functions
Try alternative rewards:
```python
# Sharpe ratio-based reward
reward = (returns - risk_free_rate) / returns.std()

# Log return-based reward
reward = np.log(equity_after / equity_before)

# Sortino ratio (penalize downside volatility)
# ... custom implementation
```

### 4. Position Sizing
Add fractional positions:
```python
# Instead of Discrete(3), use Box for continuous actions
self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
# -1 = full short, 0 = flat, +1 = full long
```

---

## 📚 Further Reading

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Custom Environments Guide](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Stable-Baselines3 Env Checker](https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html)

---

## ⚠️ Known Limitations

This is a **simplified** trading environment. It does NOT account for:
- Slippage (price moves while you're placing an order)
- Market impact (your order affects the price)
- Order book dynamics
- Funding rates (for perpetual futures)
- Margin requirements
- Liquidation mechanics
- Multiple timeframes
- News events or market sentiment

For learning purposes, these simplifications are acceptable. For real trading, you'd need a much more sophisticated simulator.

---

**Happy coding! 🚀**
