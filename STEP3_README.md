# 📊 AlphaTraderLab Step 3 - Technical Indicators & Risk Metrics

**Enhanced RL trading with richer observations and comprehensive risk analysis**

---

## 🎯 What's New in Step 3

Step 3 adds two major enhancements to make the trading agent more realistic and the evaluation more comprehensive:

1. **Technical Indicators**: Enrich observations with standard indicators (SMA, EMA, RSI, volatility, etc.)
2. **Risk Metrics**: Add comprehensive risk analysis (max drawdown, Sharpe ratio, win rate, etc.)

### Key Improvements

✅ **Richer Observations**: Agent now sees 11 features per candle (was 5)  
✅ **Risk-Aware Evaluation**: Sharpe ratio, max drawdown, volatility  
✅ **Trade Analytics**: Win rate, average win/loss, profit factor  
✅ **Backward Compatible**: Can disable indicators with `use_indicators=False`  
✅ **Clear Documentation**: All indicators and metrics explained  

---

## 📊 Technical Indicators Added

The environment now includes 6 standard technical indicators:

| Indicator | Description | Purpose |
|-----------|-------------|---------|
| **SMA_20** | 20-period Simple Moving Average | Trend identification |
| **SMA_50** | 50-period Simple Moving Average | Long-term trend |
| **EMA_20** | 20-period Exponential Moving Average | Responsive trend |
| **RSI_14** | 14-period Relative Strength Index | Overbought/oversold |
| **Log Return** | Daily logarithmic returns | Price momentum |
| **Volatility_20** | 20-day rolling volatility | Market risk |

### Observation Structure

**Before (Step 2)**: 5 features × 30 candles + 2 scalars = **152 features**
- [Open, High, Low, Close, Volume] × 30 + [Position, Equity Ratio]

**After (Step 3)**: 11 features × 30 candles + 2 scalars = **332 features**
- [Open, High, Low, Close, Volume, SMA20, SMA50, EMA20, RSI14, LogReturn, Vol20] × 30 + [Position, Equity Ratio]

### Normalization Methods

- **Price indicators** (SMA, EMA): Divided by first close price
- **RSI**: Scaled from 0-100 to 0-1 range
- **Returns & Volatility**: Z-score standardization (mean=0, std=1)

---

## 📈 Risk Metrics Added

### 1. Maximum Drawdown (%)
**What**: Largest peak-to-trough decline in equity  
**Formula**: `max((peak - trough) / peak) × 100`  
**Interpretation**:
- Lower is better (less downside risk)
- Example: -20% means you lost 20% from peak at worst point

### 2. Sharpe Ratio
**What**: Risk-adjusted return metric  
**Formula**: `(mean_return / std_return) × √252`  
**Interpretation**:
- Higher is better
- > 1.0: Good risk-adjusted returns
- > 2.0: Excellent
- < 0: Losing money on risk-adjusted basis

### 3. Volatility
**What**: Standard deviation of returns  
**Interpretation**:
- Measures how much equity fluctuates
- Lower = more stable
- Higher = more risky/volatile

### 4. Win Rate (%)
**What**: Percentage of profitable trades  
**Formula**: `(winning_trades / total_trades) × 100`  
**Interpretation**:
- 50%+: More wins than losses
- Professional traders often have 40-60%

### 5. Average Win/Loss (%)
**What**: Average return of winning/losing trades  
**Interpretation**:
- Compare avg_win vs avg_loss
- Good traders: avg_win > |avg_loss| (win bigger than lose)

### 6. Profit Factor
**What**: Ratio of total gains to total losses  
**Formula**: `sum(winning_trades) / sum(|losing_trades|)`  
**Interpretation**:
- > 1.0: Profitable overall
- > 2.0: Very good
- < 1.0: Losing money

---

## 🚀 Quick Start

### Option 1: Python Script (Recommended for Testing)

```bash
# Train with indicators (default)
python train_ppo.py --timesteps 100000

# Train without indicators (Step 2 behavior)
python train_ppo.py --timesteps 100000 --no-indicators

# Quick test with indicators
python train_ppo.py --timesteps 10000

# Evaluate existing model
python train_ppo.py --test
```

### Option 2: Jupyter Notebook

1. **Open notebook**: `notebooks/AlphaTraderLab_PPO_v1.ipynb`
2. **Update Step 6** (Environment Creation):
   ```python
   # Add use_indicators parameter
   def make_env(df, window_size=WINDOW_SIZE, initial_balance=INITIAL_BALANCE):
       def _init():
           return TradingEnv(
               df=df,
               window_size=window_size,
               initial_balance=initial_balance,
               transaction_cost=TRANSACTION_COST,
               use_indicators=True  # ← ADD THIS
           )
       return DummyVecEnv([_init])
   
   # Also update test_env creation
   test_env = TradingEnv(
       df=test_df,
       window_size=WINDOW_SIZE,
       initial_balance=INITIAL_BALANCE,
       transaction_cost=TRANSACTION_COST,
       use_indicators=True  # ← ADD THIS
   )
   ```

3. **Add explanation** (new markdown cell after Step 6):
   ```markdown
   ### 🔬 Technical Indicators
   
   The agent now sees **enriched observations** with 6 technical indicators:
   - **SMA_20, SMA_50**: Moving averages for trend
   - **EMA_20**: Exponential moving average
   - **RSI_14**: Relative Strength Index (overbought/oversold)
   - **Log Return**: Daily price momentum
   - **Volatility_20**: Market risk measure
   
   This gives the agent more context to make better decisions!
   ```

4. **Run all cells** - The comparison table will automatically show new metrics

---

## 📊 Understanding the New Output

### Example Comparison Table

```
Agent        Final Equity  Total Return (%)  Max Drawdown (%)  Sharpe Ratio  Win Rate (%)
Buy & Hold      $15,707.90           53.31%           -28.14%          0.76         100.0%
PPO             $12,450.00           24.50%           -35.20%          0.45          42.5%
Random           $5,115.88          -48.72%           -50.41%         -0.74          20.2%
```

### Interpretation

**Buy & Hold**:
- ✅ Highest return (53%)
- ✅ Best Sharpe (0.76) - good risk-adjusted return
- ✅ Lowest drawdown (-28%) - less risky
- 🎯 **Verdict**: Strong benchmark in bull market

**PPO**:
- ✅ Beat Random significantly
- ✅ Positive Sharpe (0.45) - decent risk-adjusted return
- ⚠️ Higher drawdown than Buy & Hold
- ⚠️ Didn't beat Buy & Hold
- 🎯 **Verdict**: Agent learned, but passive strategy better

**Random**:
- ❌ Negative return (-49%)
- ❌ Negative Sharpe (-0.74)
- ❌ Highest drawdown (-50%)
- 🎯 **Verdict**: As expected, random is worst

### What to Look For

**Good Signs**:
- PPO > Random in return AND Sharpe
- PPO win rate > 40%
- PPO Sharpe > 0
- PPO drawdown reasonable (< 50%)

**Bad Signs**:
- PPO ≈ Random performance
- Sharpe < 0 (taking risk for negative returns)
- Win rate < 30%
- Excessive drawdown (> 60%)

---

## 🔧 Technical Details

### Design Decisions

#### 1. Indicator Normalization
- **Moving Averages**: Divided by first close price (same as OHLC)
- **RSI**: Scaled to 0-1 range (from 0-100)
- **Returns/Volatility**: Z-score normalized

**Why**: Neural networks learn better with normalized inputs

#### 2. Indicator Warm-up
- SMA_50 requires 50 candles to compute
- Episodes start from index 50+ (not 30) when indicators enabled
- Ensures all indicators are valid (no NaN)

**Why**: Prevents feeding garbage (NaN) to the agent

#### 3. Trade Detection Logic
- **Trade starts**: Position changes from 0 → 1/2
- **Trade ends**: Position → 0, or switches (1↔2)
- **Final trade**: Closed at episode end

**Why**: Tracks actual trading activity for metrics

#### 4. Backward Compatibility
- `use_indicators=False` → Step 2 behavior (OHLCV only)
- Default: `use_indicators=True`

**Why**: Allows fair comparison and testing

### File Changes

**Modified**:
1. `envs/trading_env.py`:
   - Added `use_indicators` parameter
   - Added `_compute_indicators()` method
   - Added `_compute_rsi()` method
   - Updated `_get_observation()` to include indicators
   - Updated observation space shape

2. `utils/evaluation.py`:
   - Added `_calculate_risk_metrics()` function
   - Added `_calculate_trade_stats()` function
   - Updated `evaluate_agent()` to compute new metrics
   - Updated `compare_agents()` to show new metrics

3. `train_ppo.py`:
   - Added `--use-indicators` / `--no-indicators` flags
   - Updated `make_env()` to accept `use_indicators`
   - Updated `evaluate_all_agents()` to accept `use_indicators`
   - Updated output to show risk metrics

**New**:
- `STEP3_README.md` (this file)
- `STEP3_GUIDE.md` (detailed guide)

---

## 🎯 Usage Examples

### Basic Training
```bash
# Default: with indicators, 100k steps
python train_ppo.py

# Output includes new metrics:
# Max drawdown: -35.20%
# Sharpe ratio: 0.45
# Win rate: 42.5%
```

### Compare With/Without Indicators
```bash
# Train without indicators
python train_ppo.py --timesteps 50000 --no-indicators
# Model saved to: models/ppo_btc_usd.zip

# Train with indicators (rename model to avoid overwrite)
python train_ppo.py --timesteps 50000
# Compare results!
```

### Quick Test
```bash
# Fast test (10k steps, ~1 minute)
python train_ppo.py --timesteps 10000

# Check if indicators help:
# - Compare Sharpe ratios
# - Compare drawdowns
# - Compare win rates
```

### Different Assets
```bash
# Ethereum
python train_ppo.py --ticker ETH-USD --timesteps 100000

# Apple stock
python train_ppo.py --ticker AAPL --timesteps 100000
```

---

## 📖 Code Example: Manual Usage

```python
import yfinance as yf
from envs.trading_env import TradingEnv
from utils.evaluation import evaluate_agent, compare_agents
from stable_baselines3 import PPO

# Download data
df = yf.download("BTC-USD", start="2020-01-01", end="2024-01-01")

# Create environment WITH indicators
env_with = TradingEnv(
    df=df,
    window_size=30,
    initial_balance=10000,
    transaction_cost=0.001,
    use_indicators=True  # NEW!
)

# Create environment WITHOUT indicators (Step 2)
env_without = TradingEnv(
    df=df,
    window_size=30,
    initial_balance=10000,
    transaction_cost=0.001,
    use_indicators=False
)

# Train on environment with indicators
model = PPO("MlpPolicy", env_with, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate
results = evaluate_agent(env_with, model=model)

# Check new metrics
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Win Rate: {results['win_rate']:.1f}%")
print(f"Avg Win: {results['avg_win']:.2f}%")
print(f"Avg Loss: {results['avg_loss']:.2f}%")
```

---

## 🔍 Troubleshooting

### Issue: Observation shape mismatch

**Symptoms**: Error about observation shape

**Solution**: Make sure train and test envs use same `use_indicators` value
```python
# Both should match
train_env = make_env(train_df, use_indicators=True)
test_env = TradingEnv(test_df, use_indicators=True)  # Must match!
```

### Issue: All indicators are zero

**Symptoms**: Indicators show 0.0 in all cells

**Solution**: Check that data has enough history (50+ candles)

### Issue: NaN in observations

**Symptoms**: Agent gets NaN values

**Solution**: This shouldn't happen if `min_start_index` is set correctly. If it does:
```python
# Check indicator computation
env = TradingEnv(df, use_indicators=True)
print(f"Min start index: {env.min_start_index}")  # Should be >= 50
```

### Issue: Training slower with indicators

**Symptoms**: Training takes longer

**Explanation**: More features = more computation. This is expected.
- Without indicators: 152 features
- With indicators: 332 features

**Solution**: This is normal. For faster training:
- Use fewer timesteps for testing
- Use smaller window_size
- Train on GPU (Colab)

---

## 📊 Expected Results

### Realistic Expectations

**With Indicators vs Without**:
- Indicators give agent more context
- May or may not improve returns (depends on market)
- Should improve risk metrics (Sharpe, drawdown)
- More consistent behavior (less random)

**Don't Expect**:
- Indicators won't magically make agent profitable
- Won't always beat buy & hold (that's hard!)
- Results still vary between runs (RL is stochastic)

**Good Outcomes**:
- PPO >> Random (learned something!)
- Sharpe > 0 (positive risk-adjusted return)
- Drawdown < 50% (reasonable risk)
- Win rate 40-60% (selective trading)

---

## 🎓 Learning Objectives

After Step 3, you'll understand:
- ✅ How technical indicators work
- ✅ Why normalization matters for neural networks
- ✅ What risk-adjusted returns mean (Sharpe ratio)
- ✅ How to measure trading performance beyond returns
- ✅ Trade-level analytics (win rate, avg win/loss)
- ✅ The importance of drawdown analysis

---

## 🚀 Next Steps

### Immediate Experiments

1. **Train longer**: 200k-500k timesteps
2. **Compare indicators on/off**: Use `--no-indicators` flag
3. **Try different assets**: ETH, stocks, etc.
4. **Adjust window size**: Test 20, 40, 50 candles
5. **Multiple runs**: Average results over 5-10 runs

### Future Enhancements (Step 4+)

- **More indicators**: Bollinger Bands, MACD, Stochastic
- **Multiple timeframes**: Combine daily + hourly data
- **Walk-forward testing**: Multiple train/test splits
- **Hyperparameter optimization**: Grid search for best config
- **Ensemble methods**: Combine multiple models

---

## ⚠️ Important Notes

### What Step 3 Does NOT Do

❌ **No hard-coded rules**: Agent still learns from reward signal  
❌ **No forced exits**: Agent decides when to exit (no "flatten every N days")  
❌ **No strategy constraints**: Agent can stay in one position if it wants  

### What This Means

The agent observes richer state (indicators) but still learns purely through trial and error. It's not following any pre-programmed strategy.

### Why This Matters

- Agent can discover its own patterns
- More realistic RL setup
- Harder to learn (more state complexity)
- But potentially better strategies

---

## 📚 References

### Technical Indicators
- [RSI Explained](https://www.investopedia.com/terms/r/rsi.asp)
- [Moving Averages](https://www.investopedia.com/terms/m/movingaverage.asp)
- [Technical Analysis](https://www.investopedia.com/terms/t/technicalanalysis.asp)

### Risk Metrics
- [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp)
- [Maximum Drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp)
- [Win Rate](https://www.investopedia.com/terms/w/win-loss-ratio.asp)

### RL with Features
- [Feature Engineering for RL](https://spinningup.openai.com/)
- [Stable Baselines3 Tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

---

**Happy Training with Enhanced Features! 🚀📊🤖**

*Remember: More features ≠ guaranteed better performance. Always compare and validate!*
