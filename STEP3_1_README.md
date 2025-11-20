# Step 3.1 - Enhanced Features & Trading Costs

This document describes the Step 3.1 enhancements to AlphaTraderLab, which build upon Step 3 by adding richer observations and configurable trading costs.

## Overview

**Step 3.1 Goals:**
1. Give the RL agent **richer state inputs** (more candlestick features and additional indicators)
2. Add **configurable trading costs** to discourage pathological over-trading
3. Expose **PPO hyperparameters via CLI** for easier experimentation
4. Maintain **100% backward compatibility** with Step 3

**Key Principle:** We are NOT adding hard-coded trading rules. The agent still learns purely from the enriched observation space and reward signal.

---

## What's New in Step 3.1

### 1. Candlestick-Derived Features (per candle)

Added 6 new candlestick geometry features when `use_indicators=True`:

| Feature | Description | Range | Formula |
|---------|-------------|-------|---------|
| `body_pct` | Body size as % of range | [0, 1] | `|close - open| / (high - low)` |
| `upper_wick_pct` | Upper wick as % of range | [0, 1] | `(high - max(close, open)) / (high - low)` |
| `lower_wick_pct` | Lower wick as % of range | [0, 1] | `(min(close, open) - low) / (high - low)` |
| `is_bullish` | Bullish candle flag | {0, 1} | `1` if `close > open` and `body_pct > 0.3` |
| `is_bearish` | Bearish candle flag | {0, 1} | `1` if `close < open` and `body_pct > 0.3` |
| `is_doji` | Doji candle flag | {0, 1} | `1` if `body_pct < 0.1` |

**Why useful:** These features help the agent recognize candlestick patterns (e.g., long wicks indicate rejection, dojis indicate indecision).

### 2. Additional Technical Indicators

Added 3 new indicator families when `use_indicators=True`:

#### MACD (Moving Average Convergence Divergence)
- **MACD Line:** `EMA(12) - EMA(26)` - Normalized by first close price
- **MACD Signal:** `EMA(9) of MACD Line` - Normalized by first close price  
- **MACD Histogram:** `MACD - Signal` - Standardized (z-score)

**Why useful:** MACD is a momentum indicator that shows trend strength and potential reversals.

#### Bollinger Bands (20-period, 2 std dev)
- **%B (Percent B):** Position within bands, `(price - lower_band) / (upper_band - lower_band)` - Clipped to [0, 1]
  - `%B = 0` → Price at lower band (oversold)
  - `%B = 0.5` → Price at middle band (neutral)
  - `%B = 1` → Price at upper band (overbought)
- **Band Width:** `upper_band - lower_band` - Normalized by first close price

**Why useful:** Bollinger Bands measure volatility and relative price levels. Tight bands indicate low volatility (potential breakout), wide bands indicate high volatility.

#### ATR (Average True Range, 14-period)
- **ATR:** Exponential moving average of True Range - Normalized by first close price
- **True Range:** `max(high - low, |high - prev_close|, |low - prev_close|)`

**Why useful:** ATR measures market volatility. High ATR = volatile market, Low ATR = calm market.

### 3. Trading Costs

Added two configurable cost mechanisms to discourage excessive trading:

#### Transaction Cost (`transaction_cost_pct`)
- **Default:** `0.0` (backward compatible)
- **Applied:** On every position change (FLAT→LONG, LONG→SHORT, etc.)
- **Formula:** `cost = equity × transaction_cost_pct`
- **Example:** With `transaction_cost_pct=0.001` (0.1%), a $10,000 position change costs $10

#### Switch Penalty (`switch_penalty`)
- **Default:** `0.0` (backward compatible)
- **Applied:** Only when direction flips (LONG↔SHORT)
- **Formula:** `reward -= switch_penalty`
- **Example:** With `switch_penalty=0.0005`, each LONG→SHORT or SHORT→LONG switch subtracts 0.0005 from reward

**Why useful:** Without costs, agents may over-trade due to noise. Realistic costs encourage the agent to hold positions longer when appropriate.

### 4. Configurable PPO Hyperparameters

Added CLI flags for all key PPO hyperparameters:

| Flag | Default | Description |
|------|---------|-------------|
| `--learning-rate` | `3e-4` | Learning rate for policy updates |
| `--gamma` | `0.99` | Discount factor (how much to value future rewards) |
| `--n-steps` | `2048` | Steps collected per policy update |
| `--batch-size` | `64` | Minibatch size for gradient descent |
| `--ent-coef` | `0.01` | Entropy coefficient (encourages exploration) |
| `--clip-range` | `0.2` | PPO clipping parameter (prevents large policy updates) |

**Why useful:** Allows experimentation without modifying code. For example:
- Higher `ent_coef` → More exploration
- Higher `learning_rate` → Faster learning (but less stable)
- Higher `gamma` → More long-term thinking

---

## Observation Space Changes

### Step 3 vs Step 3.1 Comparison

| Metric | Step 3 | Step 3.1 |
|--------|--------|----------|
| Features per candle (with indicators) | 11 | 23 |
| Total observation size (window=30) | 332 | 692 |
| Observation components | OHLCV + 6 indicators + portfolio | OHLCV + 18 indicators + portfolio |

### Step 3.1 Feature Order (per candle)

When `use_indicators=True`, each candle has **23 features**:

```
[0-4]   OHLCV (Open, High, Low, Close, Volume)
[5-10]  Step 3 indicators (SMA_20, SMA_50, EMA_20, RSI_14, log_return, volatility_20)
[11-16] Step 3.1 indicators (MACD_line, MACD_signal, MACD_hist, BB_%B, BB_width, ATR_14)
[17-22] Candlestick features (body_pct, upper_wick_pct, lower_wick_pct, is_bullish, is_bearish, is_doji)
```

**Total observation:** `(window_size × 23) + 2` = `692` features for `window_size=30`

When `use_indicators=False`, behavior is identical to Step 2: **5 features per candle** (OHLCV only).

---

## Usage

### Basic Training with Step 3.1 Features

```bash
# Train with all Step 3.1 enhancements (20k timesteps for quick test)
python train_ppo.py \
  --timesteps 20000 \
  --use-indicators \
  --transaction-cost 0.001 \
  --switch-penalty 0.0005
```

### Training with Custom Hyperparameters

```bash
# Higher exploration (ent_coef) and longer horizon (gamma)
python train_ppo.py \
  --timesteps 100000 \
  --learning-rate 1e-4 \
  --gamma 0.995 \
  --ent-coef 0.02 \
  --transaction-cost 0.001 \
  --switch-penalty 0.0005
```

### Backward Compatibility Test (Step 3 behavior)

```bash
# Use Step 3 behavior (no costs, default hyperparameters)
python train_ppo.py \
  --timesteps 20000 \
  --use-indicators
```

### Disable Indicators (Step 2 behavior)

```bash
# Use Step 2 behavior (OHLCV only, no indicators)
python train_ppo.py \
  --timesteps 20000 \
  --no-indicators
```

---

## Configuration Output

When you run `train_ppo.py`, you'll see a comprehensive configuration summary:

```
🔧 Configuration:
   Environment:
     - Use indicators: True
     - Window size: 30
     - Transaction cost: 0.0010
     - Switch penalty: 0.0005
   PPO Hyperparameters:
     - Learning rate: 0.0003
     - Gamma: 0.99
     - N-steps: 2048
     - Batch size: 64
     - Entropy coef: 0.01
     - Clip range: 0.2
```

This makes it easy to verify your experimental setup.

---

## Indicator Normalization Summary

All indicators are normalized to be neural network friendly:

| Indicator | Normalization Method | Why |
|-----------|---------------------|-----|
| SMA, EMA | Divide by first close price | Makes them scale-invariant |
| RSI | Divide by 100 (→ [0, 1]) | Already bounded [0, 100] |
| Log returns, Volatility | Z-score standardization | Varies widely across regimes |
| MACD line/signal | Divide by first close price | Price-relative |
| MACD histogram | Z-score standardization | Unbounded, regime-dependent |
| BB %B | Natural [0, 1] range | Already normalized |
| BB width | Divide by first close price | Price-relative |
| ATR | Divide by first close price | Price-relative |
| Candlestick features | Natural [0, 1] or {0, 1} | Already percentages or binary |

---

## Expected Results

### Without Costs (Baseline)
With `--transaction-cost 0 --switch-penalty 0`:
- PPO may over-trade (100-300 trades on test set)
- Win rate typically 30-40%
- May underperform Buy & Hold due to churn

### With Costs (Recommended)
With `--transaction-cost 0.001 --switch-penalty 0.0005`:
- PPO should trade less frequently (50-150 trades)
- Higher average profit per trade
- Better risk-adjusted returns (Sharpe ratio)
- Closer to or potentially better than Buy & Hold

**Note:** With only 20k timesteps (smoke test), the agent is still learning. For meaningful results, train with **100k-500k timesteps** (20-60 minutes).

---

## Code Examples

### Manual Environment Creation (Python)

```python
from envs.trading_env import TradingEnv
import pandas as pd

# Create environment with Step 3.1 features
env = TradingEnv(
    df=your_dataframe,
    window_size=30,
    initial_balance=10000.0,
    transaction_cost=0.001,  # Legacy parameter (keep for compatibility)
    transaction_cost_pct=0.001,  # Step 3.1 parameter
    switch_penalty=0.0005,       # Step 3.1 parameter
    use_indicators=True          # Enables all 23 features per candle
)

# Observation space: (30 × 23 + 2,) = (692,)
print(env.observation_space.shape)  # (692,)
```

### Training with Custom PPO Config

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrap environment
vec_env = DummyVecEnv([lambda: env])

# Create PPO with custom hyperparameters
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=1e-4,      # Lower for stability
    gamma=0.995,             # Higher for longer-term thinking
    n_steps=4096,            # More steps per update
    batch_size=128,          # Larger batches
    ent_coef=0.02,           # More exploration
    clip_range=0.2,
    verbose=1
)

model.learn(total_timesteps=100_000)
model.save("models/ppo_step3_1.zip")
```

---

## Design Decisions

### 1. Why Not More Indicators?

**Decision:** Limited to 3 new indicator families (MACD, BB, ATR) + candlestick features.

**Rationale:**
- **Diminishing returns:** Adding 20+ indicators can lead to overfitting and slower learning
- **Correlation:** Many indicators are correlated (e.g., ATR and Bollinger width both measure volatility)
- **Beginner-friendly:** Keep the project educational and understandable
- **Future extensibility:** Users can easily add more in their own forks

### 2. Why Separate `transaction_cost_pct` and `switch_penalty`?

**Decision:** Two separate parameters instead of one combined cost.

**Rationale:**
- **Flexibility:** Transaction costs apply to all position changes, switch penalty only applies to direction flips (LONG↔SHORT)
- **Realism:** Real-world costs are multi-faceted (fees + slippage + opportunity cost)
- **Experimentation:** Users can test "only transaction costs" vs "only switch penalty" vs "both"

### 3. Why Default Costs to 0.0?

**Decision:** Both `transaction_cost_pct` and `switch_penalty` default to `0.0`.

**Rationale:**
- **Backward compatibility:** Step 3 users should get identical results by default
- **Educational:** Allows comparison of "frictionless" vs "realistic" markets
- **User choice:** Users explicitly opt-in to costs based on their use case

### 4. Why Not Learn Costs from Data?

**Decision:** Costs are hyperparameters, not learned parameters.

**Rationale:**
- **Simplicity:** Learning costs would require complex multi-task learning or meta-learning
- **Domain knowledge:** Real-world costs are known (e.g., exchange fees are public)
- **Scope:** Step 3.1 focuses on single-agent refinement, not meta-learning

---

## Troubleshooting

### Problem: Agent still over-trades with costs enabled

**Possible causes:**
1. Costs too low (try increasing `transaction_cost_pct` to 0.002-0.005)
2. Switch penalty too low (try increasing `switch_penalty` to 0.001-0.002)
3. Training too short (try 100k-500k timesteps)
4. Entropy coefficient too high (try reducing `ent_coef` to 0.005)

### Problem: Training slower than Step 3

**Explanation:** Observation space increased from 332 to 692 features, so neural network is larger.

**Solutions:**
- Reduce `window_size` to 20 (reduces observation to 462 features)
- Use smaller `n_steps` (e.g., 1024 instead of 2048) for faster updates
- This is expected; richer state = more computation

### Problem: NaN values in indicators

**Should not happen:** All indicators have proper warmup handling (`min_start_index=50`).

**If it happens:**
1. Check that your data has at least 80+ candles
2. Verify no NaN values in input OHLCV data
3. File a bug report with your data sample

---

## Next Steps

After Step 3.1, possible directions:

1. **Longer training:** Run with 500k-1M timesteps for fully trained agents
2. **Hyperparameter tuning:** Use Optuna or grid search to find optimal PPO config
3. **Multi-asset training:** Train on multiple cryptocurrencies
4. **Step 4 (Future):** Population-based training with multiple agents

---

## File Changes Summary

| File | Changes | Lines Added |
|------|---------|-------------|
| `envs/trading_env.py` | Added 12 new indicators + candlestick features, costs logic | ~300 |
| `train_ppo.py` | Added CLI flags, config printing, hyperparameter passing | ~50 |
| `STEP3_1_README.md` | New documentation | ~400 |

**Backward compatibility:** Verified. Running with default flags gives identical Step 3 behavior.

---

## References

- **Step 3 README:** `STEP3_README.md` - Describes the base indicator system
- **PPO Paper:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **MACD:** [Investopedia - MACD](https://www.investopedia.com/terms/m/macd.asp)
- **Bollinger Bands:** [Investopedia - Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp)
- **ATR:** [Investopedia - ATR](https://www.investopedia.com/terms/a/atr.asp)
