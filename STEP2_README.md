# 🤖 AlphaTraderLab Step 2 - PPO Training & Evaluation

**Train intelligent trading agents using Proximal Policy Optimization (PPO)**

---

## 🎯 What's New in Step 2

Step 2 adds **real machine learning** to AlphaTraderLab! Instead of random actions, we train an agent that learns profitable trading strategies from historical data.

### Key Features

✅ **PPO Training**: Train agents using Stable-Baselines3  
✅ **Baseline Comparison**: Compare against Random and Buy & Hold  
✅ **Comprehensive Evaluation**: Detailed metrics and visualizations  
✅ **Easy to Use**: Works in Colab or locally  
✅ **Well Documented**: Extensive comments and guides  

---

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended for Learning)

**Best for**: Understanding the full pipeline step-by-step

1. **Open notebook**: `notebooks/AlphaTraderLab_PPO_v1.ipynb`
2. **Upload files** (if in Colab):
   - `trading_env.py` (from `envs/`)
   - `evaluation.py` (from `utils/`)
3. **Run all cells**: Will train and evaluate automatically
4. **See results**: Charts and comparison tables

**Time**: ~5-10 minutes for 100k training steps

### Option 2: Python Script (Recommended for Multiple Runs)

**Best for**: Quick experiments and hyperparameter tuning

```bash
# Basic training (100k steps, ~5 minutes)
python train_ppo.py

# Longer training for better results (200k steps, ~10 minutes)
python train_ppo.py --timesteps 200000

# Quick test (10k steps, ~1 minute)
python train_ppo.py --timesteps 10000

# Evaluate existing model (skip training)
python train_ppo.py --test

# Try different asset
python train_ppo.py --ticker ETH-USD

# Custom window size
python train_ppo.py --window-size 50
```

---

## 📊 What You'll See

### Training Output

```
🚀 AlphaTraderLab - PPO Training Pipeline
============================================================

📊 Downloading BTC-USD data from 2018-01-01...
✅ Downloaded 2876 days of data
   Date range: 2018-01-01 to 2025-11-15

📈 Data split:
   Training:   2013 days (2018-01-01 to 2023-07-06)
   Testing:    863 days (2023-07-07 to 2025-11-15)

🤖 Training PPO agent for 100,000 timesteps...
Using cpu device

[Training progress bars and metrics...]

✅ Model saved to: models/ppo_btc_usd.zip
```

### Evaluation Results

```
============================================================
📊 Agent Comparison
============================================================
     Agent   Initial Equity  Final Equity  Total Return (%)
Buy & Hold      $10,028.91    $37,279.47          271.72%
       PPO      $10,000.00    $15,234.12           52.34%
    Random       $9,924.28     $2,317.53          -76.65%
============================================================
```

### Visualization (Notebook Only)

- 📈 **Equity Curves**: Portfolio value over time for all agents
- 🎯 **Action Distribution**: What actions each agent took
- 📊 **Performance Table**: Side-by-side comparison
- 💡 **Interpretation**: What the results mean

---

## 🧠 Understanding PPO

### What is PPO?

**PPO (Proximal Policy Optimization)** is a reinforcement learning algorithm that learns optimal trading strategies through trial and error.

#### How It Works (Simple)

1. **Observe**: Look at market data (30 days of OHLCV)
2. **Act**: Choose FLAT, LONG, or SHORT
3. **Feedback**: Get reward based on profit/loss
4. **Learn**: Adjust decision-making to maximize rewards
5. **Repeat**: Do this thousands of times

#### Why PPO?

- ✅ **Stable**: Doesn't make wild changes
- ✅ **Proven**: Used by OpenAI and DeepMind
- ✅ **Efficient**: Learns relatively quickly
- ✅ **General**: Works for many problems

### Training Process

```
Start
  ↓
Download historical data
  ↓
Split into train (70%) and test (30%)
  ↓
Create training environment
  ↓
Initialize PPO agent (random policy)
  ↓
Training Loop (100k steps):
  ├─ Agent observes market
  ├─ Agent takes actions
  ├─ Environment gives rewards
  └─ Agent updates neural network
  ↓
Save trained model
  ↓
Evaluate on test data:
  ├─ PPO agent
  ├─ Random agent (baseline)
  └─ Buy & Hold (benchmark)
  ↓
Compare and visualize results
```

---

## 📈 Interpreting Results

### Good Results

✅ **PPO significantly beats Random**
```
PPO:    +25.5%
Random: -8.2%
→ Agent learned meaningful strategies!
```

✅ **PPO beats or matches Buy & Hold**
```
PPO:        +32.1%
Buy & Hold: +28.4%
→ Excellent! Active trading paid off!
```

### Mixed Results (Common)

⚠️ **PPO beats Random but not Buy & Hold**
```
PPO:        +15.2%
Random:     -12.3%
Buy & Hold: +45.8%
→ Agent learned, but couldn't beat market trend
```

**Interpretation**: This is NORMAL and EXPECTED! Buy & Hold is a strong benchmark, especially in bull markets. The agent still learned (beat random).

### Poor Results (Needs Improvement)

❌ **PPO performs like Random**
```
PPO:    -5.2%
Random: -6.1%
→ Agent didn't learn effectively
```

**Solutions**:
- Train longer (increase `--timesteps`)
- Adjust hyperparameters (see PPO_GUIDE.md)
- Try different window size
- Check if data quality is good

---

## 🎛️ Configuration

### Command-Line Arguments

```bash
python train_ppo.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--timesteps` | 100000 | Total training steps |
| `--test` | False | Skip training, only evaluate |
| `--ticker` | BTC-USD | Asset ticker symbol |
| `--window-size` | 30 | Observation window size (days) |

### Examples

```bash
# Train on Ethereum
python train_ppo.py --ticker ETH-USD

# Train on Apple stock with larger window
python train_ppo.py --ticker AAPL --window-size 50

# Quick 10k step test on Bitcoin
python train_ppo.py --timesteps 10000

# Evaluate existing model
python train_ppo.py --test
```

### Environment Parameters

In the notebook or when creating environments manually:

```python
TradingEnv(
    df=dataframe,              # Historical OHLCV data
    window_size=30,            # Past candles in observation
    initial_balance=10000.0,   # Starting portfolio ($)
    transaction_cost=0.001     # Trading fee (0.1%)
)
```

### PPO Hyperparameters

Default configuration (in `train_ppo.py`):

```python
PPO(
    policy="MlpPolicy",      # Neural network policy
    learning_rate=3e-4,      # How fast to learn
    n_steps=2048,            # Steps per update
    batch_size=64,           # Mini-batch size
    n_epochs=10,             # Training epochs
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE parameter
    clip_range=0.2,          # PPO clipping
    ent_coef=0.01,           # Exploration bonus
    verbose=1                # Print progress
)
```

**See [PPO_GUIDE.md](PPO_GUIDE.md) for detailed hyperparameter tuning guide**

---

## 📂 Files Created

After training, you'll have:

```
alpha_trader_lab/
├── models/
│   └── ppo_btc_usd.zip         # Trained PPO model
├── ppo_results.csv              # PPO evaluation data (notebook)
├── random_results.csv           # Random agent data (notebook)
├── bh_results.csv               # Buy & Hold data (notebook)
└── agent_comparison.csv         # Comparison table (notebook)
```

### Loading Saved Models

```python
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("models/ppo_btc_usd.zip")

# Use for predictions
obs = env.reset()
action, _states = model.predict(obs, deterministic=True)
```

---

## 🔧 Troubleshooting

### Issue: PPO performs like Random

**Symptoms**: PPO and Random have similar returns

**Solutions**:
1. Train longer: `--timesteps 200000`
2. Check hyperparameters (see PPO_GUIDE.md)
3. Try different window size: `--window-size 50`
4. Verify data quality

### Issue: Training takes too long

**Symptoms**: Training doesn't finish in reasonable time

**Solutions**:
1. Use fewer timesteps: `--timesteps 50000`
2. Use GPU in Colab (`Runtime > Change runtime type > GPU`)
3. Reduce n_steps in PPO configuration

### Issue: Import errors

**Symptoms**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
```bash
# Install/update dependencies
pip install -r requirements.txt

# Or install individually
pip install stable-baselines3 gymnasium yfinance
```

### Issue: Results vary wildly

**Symptoms**: Different runs give very different results

**Explanation**: This is NORMAL! RL training is stochastic.

**Solutions**:
1. Run multiple times (5-10 runs) and average
2. Train longer for more stable learning
3. Set random seeds for reproducibility

---

## 🎓 Learning Path

### Beginner Path

1. ✅ Complete Step 1 (random agent)
2. ✅ Run `AlphaTraderLab_PPO_v1.ipynb` start to finish
3. ✅ Read the markdown explanations
4. ✅ Understand what each metric means
5. ✅ Try different hyperparameters

### Intermediate Path

1. ✅ Run `train_ppo.py` with different timesteps
2. ✅ Try different assets (ETH-USD, stocks)
3. ✅ Experiment with window sizes
4. ✅ Read [PPO_GUIDE.md](PPO_GUIDE.md)
5. ✅ Modify hyperparameters

### Advanced Path

1. ✅ Implement custom reward functions
2. ✅ Add technical indicators to observations
3. ✅ Try other algorithms (A2C, DQN, SAC)
4. ✅ Build ensemble models
5. ✅ Read the PPO paper

---

## 📚 Documentation

- **[PPO_GUIDE.md](PPO_GUIDE.md)**: Comprehensive guide to PPO training
  - What is PPO?
  - Hyperparameter tuning
  - Common issues
  - Advanced topics

- **[ENVIRONMENT_GUIDE.md](envs/ENVIRONMENT_GUIDE.md)**: Technical environment details
  - Observation/action spaces
  - Reward function
  - Customization options

- **[SETUP.md](SETUP.md)**: Installation and setup help
  - Local installation
  - Colab usage
  - Troubleshooting

---

## 🚀 Next Steps

### Immediate Experiments

1. **Train longer**: `--timesteps 200000` or more
2. **Try other assets**: ETH-USD, AAPL, TSLA, etc.
3. **Adjust window size**: 10, 20, 50, 100 days
4. **Tune hyperparameters**: See PPO_GUIDE.md

### Future Enhancements (Step 3+)

- **Technical indicators**: RSI, MACD, Bollinger Bands
- **Multi-asset trading**: Trade multiple assets simultaneously
- **Risk metrics**: Sharpe ratio, max drawdown, etc.
- **Live paper trading**: Test on real-time data
- **Advanced features**: See project roadmap in main README

---

## ⚠️ Important Reminders

1. **Educational purposes only**: Not financial advice
2. **Results vary**: RL is stochastic and markets are unpredictable
3. **Past ≠ Future**: Historical performance doesn't guarantee future results
4. **Simplifications**: Real trading has additional complexity
5. **Start simple**: Master basics before adding complexity

---

## 🤝 Contributing

Found a bug or have an improvement?

1. Keep code beginner-friendly
2. Add comments and documentation
3. Test with the notebooks
4. Follow existing code style

---

## 📖 Additional Resources

### Papers
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Original PPO algorithm
- [Spinning Up](https://spinningup.openai.com/) - OpenAI's RL guide

### Documentation
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - Library docs
- [Gymnasium](https://gymnasium.farama.org/) - Environment API

### Tutorials
- [SB3 Tutorial](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) - Pre-trained models

---

**Congratulations on completing Step 2! 🎉**

You've successfully trained an RL agent from scratch. Keep experimenting and learning!

**Happy Trading! 🚀📈🤖**

*Remember: The goal is to learn RL concepts, not to get rich quick!*
