# 🤖 PPO Training Guide - AlphaTraderLab Step 2

**Complete guide to training and evaluating PPO agents for trading**

---

## 📋 Table of Contents

1. [What is PPO?](#what-is-ppo)
2. [Quick Start](#quick-start)
3. [Understanding the Results](#understanding-the-results)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Common Issues](#common-issues)
6. [Advanced Topics](#advanced-topics)

---

## 🧠 What is PPO?

**PPO (Proximal Policy Optimization)** is a reinforcement learning algorithm that learns by trial and error.

### How It Works (Simple Explanation)

Imagine teaching a robot to trade:

1. **Observe**: Robot looks at market data (price charts, volume, etc.)
2. **Act**: Robot decides what to do (FLAT, LONG, or SHORT)
3. **Feedback**: Robot sees if it made or lost money
4. **Learn**: Robot adjusts its decision-making to do better next time
5. **Repeat**: Do this thousands of times until the robot gets good

### Why PPO?

- ✅ **Stable**: Doesn't make drastic changes that break learning
- ✅ **Efficient**: Learns relatively quickly
- ✅ **Proven**: Used by OpenAI, DeepMind, and researchers worldwide
- ✅ **General**: Works for many different problems

### Technical Details (For the Curious)

- **Policy Network**: Neural network that maps observations → action probabilities
- **Value Network**: Estimates how good the current situation is
- **Clipping**: Prevents too-large updates (the "proximal" part)
- **On-Policy**: Learns from recent experience (unlike DQN which uses replay buffer)

---

## 🚀 Quick Start

### Option 1: Google Colab (Easiest)

1. **Open Colab**: [colab.research.google.com](https://colab.research.google.com/)
2. **Upload notebook**: `notebooks/AlphaTraderLab_PPO_v1.ipynb`
3. **Upload files**: 
   - `envs/trading_env.py`
   - `utils/evaluation.py`
4. **Run all cells**: `Runtime > Run all`
5. **Wait**: ~5 minutes for training
6. **See results**: Charts and comparison table at the end

### Option 2: Local Training (Recommended for Multiple Runs)

```bash
# Navigate to project
cd alpha_trader_lab

# Install dependencies (if not done)
pip install -r requirements.txt

# Train PPO agent (100k steps, ~3-5 min)
python train_ppo.py

# Train longer for better results (200k steps, ~6-10 min)
python train_ppo.py --timesteps 200000

# Quick test (50k steps, ~2-3 min)
python train_ppo.py --timesteps 50000

# Evaluate only (skip training)
python train_ppo.py --test
```

### What You'll Get

- ✅ **Trained model**: `models/ppo_btc_usd.zip`
- ✅ **Evaluation results**: Printed to console
- ✅ **Comparison table**: PPO vs Random vs Buy & Hold
- ✅ (Notebook only) **Charts**: Equity curves, action distributions

---

## 📊 Understanding the Results

### Key Metrics Explained

#### 1. Final Equity
**What it is**: Portfolio value at the end of testing  
**Initial**: $10,000  
**Example**: $12,500 (25% gain) or $8,500 (15% loss)

**Interpretation**:
- **> $10,000**: Made money
- **< $10,000**: Lost money
- **Compare across agents**: Higher is better

#### 2. Total Return (%)
**What it is**: Percentage gain/loss from start to finish  
**Formula**: `((Final - Initial) / Initial) × 100`

**Interpretation**:
- **Positive**: Profitable
- **Negative**: Unprofitable
- **> 0%**: Beat cash (doing nothing)
- **> Market return**: Beat the market

#### 3. Number of Trades
**What it is**: How many times the agent changed position

**Interpretation**:
- **Too few (< 5)**: Agent may be too conservative
- **Reasonable (10-50)**: Strategic trading
- **Too many (> 100)**: Over-trading, high transaction costs
- **Buy & Hold**: Should be ~1 (buy once, hold forever)

#### 4. Total Reward
**What it is**: Sum of all step rewards  
**Note**: Rewards are normalized by initial balance

**Interpretation**:
- Use for relative comparison, not absolute value
- Higher is generally better
- Can be negative even if final equity is positive (due to drawdowns)

### Comparing Agents

#### PPO vs Random
**What it tests**: Did the agent learn anything?

| PPO Return | Random Return | Interpretation |
|------------|---------------|----------------|
| +20% | -10% | ✅ Excellent! Agent learned valuable patterns |
| +10% | +5% | ✅ Good! Agent learned some strategies |
| +5% | +4% | ⚠️ Marginal. May need more training |
| -5% | -3% | ❌ Bad. Agent didn't learn effectively |

**Key Point**: PPO should ALWAYS beat random. If not, something is wrong.

#### PPO vs Buy & Hold
**What it tests**: Is active trading worth it?

| PPO Return | Buy & Hold Return | Interpretation |
|------------|-------------------|----------------|
| +30% | +20% | 🎉 Exceptional! Rare achievement |
| +20% | +25% | ✅ Good! Close to benchmark |
| +10% | +30% | ℹ️ Common. Buy & hold is hard to beat |
| -10% | +20% | ❌ Poor. Active trading hurt performance |

**Key Point**: Buy & hold is a STRONG benchmark. Many professional traders can't beat it consistently.

### Example Results Analysis

```
Agent Comparison:
┌──────────────────────┬────────────────┬──────────────┬──────────────┐
│ Agent                │ Final Equity   │ Total Return │ Trades       │
├──────────────────────┼────────────────┼──────────────┼──────────────┤
│ Buy & Hold           │ $12,834.50     │ 28.35%       │ 1            │
│ PPO (Trained)        │ $11,245.20     │ 12.45%       │ 23           │
│ Random (Baseline)    │ $9,123.40      │ -8.77%       │ 87           │
└──────────────────────┴────────────────┴──────────────┴──────────────┘
```

**Analysis**:
- ✅ **PPO > Random**: Agent learned (beat baseline by 21%)
- ℹ️ **PPO < Buy & Hold**: Active trading didn't beat passive (common result)
- ✅ **PPO positive return**: Made money (beat cash)
- ⚠️ **Random traded too much**: 87 trades = high transaction costs

**Verdict**: **Decent result**. PPO learned meaningful strategies but couldn't beat the strong bull market. Consider as a success for learning purposes.

---

## 🎛️ Hyperparameter Tuning

### Default Configuration (Good Starting Point)

```python
PPO(
    policy="MlpPolicy",
    learning_rate=3e-4,      # How fast to learn
    n_steps=2048,            # Steps per update
    batch_size=64,           # Training batch size
    n_epochs=10,             # Training epochs per update
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE parameter
    clip_range=0.2,          # PPO clipping
    ent_coef=0.01,           # Exploration bonus
)
```

### What Each Parameter Does

#### learning_rate (default: 3e-4)
**What it controls**: How big the learning steps are

**Adjust if**:
- Training is **unstable** → Decrease to `1e-4`
- Training is **too slow** → Increase to `5e-4`
- **Very stable** problem → Can try `1e-3`

**Values to try**: `[1e-4, 3e-4, 5e-4, 1e-3]`

#### n_steps (default: 2048)
**What it controls**: How many steps before updating the network

**Adjust if**:
- Want **faster updates** → Decrease to `1024`
- Want **more stable learning** → Increase to `4096`
- **Limited memory** → Decrease

**Values to try**: `[1024, 2048, 4096]`

#### gamma (default: 0.99)
**What it controls**: How much to value future rewards

**Interpretation**:
- **0.95**: Short-term focus (next few steps matter most)
- **0.99**: Balanced (standard)
- **0.995**: Long-term planning (future matters a lot)

**Values to try**: `[0.95, 0.99, 0.995]`

#### ent_coef (default: 0.01)
**What it controls**: Exploration vs exploitation trade-off

**Adjust if**:
- Agent **too conservative** → Increase to `0.02` or `0.05`
- Agent **too random** → Decrease to `0.005` or `0.001`
- **Early training**: Keep higher for exploration
- **Late training**: Decrease for exploitation

**Values to try**: `[0.001, 0.01, 0.05]`

### Training Duration

#### total_timesteps
**How long to train**: More = better (usually)

**Guidelines**:
- **Quick test**: 50,000 steps (~2-3 min)
- **Standard**: 100,000 steps (~5 min)
- **Better results**: 200,000-500,000 steps (~10-25 min)
- **Serious training**: 1,000,000+ steps (1+ hours)

**Tip**: Monitor training progress. If performance plateaus, stop early.

### Recommended Experiments

#### Experiment 1: Conservative Agent
```python
# Less exploration, more exploitation
ent_coef=0.001          # Lower exploration
learning_rate=1e-4       # Slower, more stable learning
```

#### Experiment 2: Aggressive Agent
```python
# More exploration, faster learning
ent_coef=0.05           # Higher exploration
learning_rate=5e-4       # Faster learning
```

#### Experiment 3: Long-term Planner
```python
# Focus on long-term rewards
gamma=0.995             # Value future more
n_steps=4096            # Longer rollouts
```

---

## 🔧 Common Issues

### Issue 1: PPO Performs Like Random

**Symptoms**:
- PPO and Random have similar returns
- PPO equity curve is very volatile
- Actions seem random

**Possible Causes & Solutions**:

1. **Not trained enough**
   - ✅ Solution: Increase `total_timesteps` to 200k-500k

2. **Learning rate too high**
   - ✅ Solution: Decrease `learning_rate` to `1e-4`

3. **Too much exploration**
   - ✅ Solution: Decrease `ent_coef` to `0.001`

4. **Environment is too hard**
   - ✅ Solution: Try a simpler asset or longer window_size

### Issue 2: Training is Unstable

**Symptoms**:
- Rewards jump up and down wildly
- Performance gets worse over time
- Error messages about NaN or Inf

**Solutions**:
1. Decrease `learning_rate` to `1e-4` or `5e-5`
2. Decrease `clip_range` to `0.1`
3. Check environment: ensure observations are normalized
4. Reduce `batch_size` if memory issues

### Issue 3: Agent Only Takes One Action

**Symptoms**:
- Agent stays FLAT the entire time
- Or only goes LONG
- Or only goes SHORT

**Possible Causes & Solutions**:

1. **Learned a dominant strategy**
   - ℹ️ May be correct for the data! Check if Buy & Hold (LONG) is best

2. **Reward function issue**
   - ✅ Check that transaction costs aren't too high
   - ✅ Ensure rewards are balanced (not always negative/positive)

3. **Need more exploration**
   - ✅ Increase `ent_coef` to `0.05` or `0.1`
   - ✅ Train longer

### Issue 4: Training Takes Too Long

**Symptoms**:
- Training runs for hours
- Colab times out

**Solutions**:
1. Reduce `total_timesteps` to 50k for testing
2. Reduce `n_steps` to 1024
3. Use GPU in Colab (`Runtime > Change runtime type > GPU`)
4. Train locally on a powerful machine

### Issue 5: Results Vary Wildly Between Runs

**Symptoms**:
- Sometimes PPO wins, sometimes loses badly
- Inconsistent performance

**Explanation**: This is NORMAL! RL training is stochastic.

**Solutions**:
1. **Run multiple times** (5-10 runs) and average results
2. **Set random seeds** for reproducibility:
   ```python
   np.random.seed(42)
   env.reset(seed=42)
   ```
3. **Increase training time** for more stable learning
4. **Use larger window_size** for more context

---

## 🚀 Advanced Topics

### Multiple Train/Test Splits

Don't rely on a single test period! Try:

```python
# Walk-forward analysis
splits = [
    ("2018-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),  # Train: 2018-2021, Test: 2022-2023
    ("2019-01-01", "2022-12-31", "2023-01-01", "2024-12-31"),  # Train: 2019-2022, Test: 2023-2024
]

for train_start, train_end, test_start, test_end in splits:
    # Train and evaluate on each split
    ...
```

### Ensemble Methods

Combine multiple models:

```python
# Train 5 different models
models = []
for seed in [42, 123, 456, 789, 1011]:
    env.reset(seed=seed)
    model = PPO(...).learn(100_000)
    models.append(model)

# Aggregate predictions (majority vote or average)
def ensemble_predict(obs):
    actions = [model.predict(obs)[0] for model in models]
    # Return most common action
    return max(set(actions), key=actions.count)
```

### Custom Reward Functions

Modify the environment to reward different behaviors:

```python
# In trading_env.py, modify the step() function:

# Original: Simple equity change
reward = (self.equity - equity_before) / self.initial_balance

# Alternative 1: Penalize risk (volatility)
returns = np.diff(equity_history) / equity_history[:-1]
sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
reward = sharpe

# Alternative 2: Penalize drawdowns
max_equity = max(equity_history)
drawdown = (max_equity - self.equity) / max_equity
reward = (self.equity - equity_before) / self.initial_balance - 0.1 * drawdown

# Alternative 3: Reward winning streaks
if self.equity > equity_before:
    reward = (self.equity - equity_before) / self.initial_balance
    if self.win_streak > 3:  # Bonus for consistency
        reward *= 1.2
```

### Feature Engineering

Add technical indicators to observations:

```python
# In trading_env.py, add to _get_observation():

# Calculate RSI
rsi = compute_rsi(self.prices, period=14)

# Calculate MACD
macd, signal = compute_macd(self.prices)

# Add to observation
observation = np.concatenate([
    flat_window,
    [position_encoded, equity_ratio, rsi[-1], macd[-1]]
])
```

### Transfer Learning

Train on one asset, fine-tune on another:

```python
# Train on BTC
model = PPO("MlpPolicy", btc_env)
model.learn(100_000)
model.save("ppo_btc.zip")

# Fine-tune on ETH
model = PPO.load("ppo_btc.zip", env=eth_env)
model.learn(50_000)  # Continue training on ETH
```

---

## 📚 Additional Resources

### Documentation
- [Stable-Baselines3 PPO Docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [PPO Paper](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

### Tutorials
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Stable-Baselines3 Tutorial](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) (pre-trained models)

### Communities
- [Stable-Baselines3 GitHub](https://github.com/DLR-RM/stable-baselines3)
- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/)
- [RL Discord Server](https://discord.gg/xhfNqQv)

---

## ⚠️ Final Reminders

1. **This is for EDUCATION**: Not financial advice
2. **Results vary**: RL is inherently stochastic
3. **Overfitting is real**: Good training ≠ good testing
4. **Transaction costs matter**: Real trading has fees and slippage
5. **Market changes**: Past patterns may not repeat
6. **Start simple**: Master the basics before adding complexity

---

## 🎓 Next Steps

After mastering Step 2, consider:

1. **Step 3**: Add technical indicators and advanced features
2. **Step 4**: Build a backtesting framework with metrics
3. **Step 5**: Implement paper trading (live simulation)
4. **Research**: Read papers on RL for trading
5. **Contribute**: Share your improvements on GitHub!

---

**Happy Training! 🚀📈🤖**

*Remember: The goal is to learn RL concepts, not to get rich quick!*
