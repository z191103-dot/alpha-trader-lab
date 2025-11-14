# ⚡ Quick Start Guide

**Get up and running in 5 minutes!**

---

## 🚀 Fastest Path: Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/AlphaTraderLab_v0.ipynb`
3. When prompted, upload `envs/trading_env.py`
4. Click `Runtime > Run all`
5. Done! ✅

---

## 💻 Local Installation (Quick)

### Prerequisites Check
```bash
python --version  # Should be 3.10+
pip --version     # Should be installed
```

### 3-Step Setup
```bash
# 1. Navigate to project folder
cd alpha_trader_lab

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test it works
python test_env.py
```

You should see: `🎉 All tests passed!`

---

## 📓 Run the Notebook

```bash
# Install Jupyter (if needed)
pip install jupyter

# Launch notebook
jupyter notebook notebooks/AlphaTraderLab_v0.ipynb
```

Then click `Cell > Run All` in Jupyter.

---

## ✅ Verify Installation

Run this Python code:
```python
from envs.trading_env import TradingEnv
import yfinance as yf

# Download sample data
df = yf.download("BTC-USD", start="2023-01-01", end="2024-01-01", progress=False)

# Create environment
env = TradingEnv(df, window_size=30, initial_balance=10000)

# Test it
obs, info = env.reset()
print(f"✅ Environment created! Observation shape: {obs.shape}")
```

If this works, you're all set! 🎉

---

## 🆘 Quick Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'gymnasium'`  
**Fix**: `pip install -r requirements.txt`

**Problem**: `ImportError: cannot import name 'TradingEnv'`  
**Fix**: Make sure you're in the `alpha_trader_lab` folder

**Problem**: yfinance download fails  
**Fix**: Check internet connection, try again in a few minutes

**Problem**: Jupyter won't start  
**Fix**: `pip install jupyter` then try again

---

## 📚 Where to Go Next

1. ✅ Run `test_env.py` → Verify environment works
2. ✅ Open `notebooks/AlphaTraderLab_v0.ipynb` → See the demo
3. ✅ Read `README.md` → Understand the project
4. ✅ Check `ENVIRONMENT_GUIDE.md` → Learn the technical details
5. ✅ Experiment! → Change parameters, try different assets

---

## 🎯 What You'll See

The notebook will:
- Download Bitcoin price data from 2018-present
- Create a trading environment
- Run a random agent for 200 steps
- Show you:
  - Price chart
  - Portfolio equity over time
  - Actions taken
  - Rewards earned

**Expected Result**: Random agent will probably lose money or break even. That's normal! In Step 2, we'll train a smart RL agent.

---

## 💡 Quick Customization

Want to try something different? Edit these in the notebook:

```python
# Try a different asset
df = yf.download("ETH-USD", ...)  # Ethereum
df = yf.download("AAPL", ...)     # Apple stock

# Change environment parameters
env = TradingEnv(
    df=df,
    window_size=50,        # See more history
    initial_balance=50000, # More money
    transaction_cost=0.002 # Higher fees
)

# Run for more steps
num_steps = 500  # Instead of 200
```

---

## 🎓 Learning Path

**Complete Beginner?** Follow this order:
1. QUICKSTART.md (you are here)
2. README.md
3. Run the notebook
4. SETUP.md (if you need help)
5. ENVIRONMENT_GUIDE.md (when you want to dig deeper)

**Experienced Programmer?** Skip to:
1. ENVIRONMENT_GUIDE.md
2. Review `envs/trading_env.py`
3. Run `test_env.py`
4. Start customizing!

---

## ⏱️ Time Estimates

- **Colab Setup**: 2 minutes
- **Local Installation**: 5 minutes
- **Running Notebook**: 3 minutes
- **Understanding the Code**: 30 minutes
- **First Customization**: 10 minutes

---

**You've got this! 🚀**

If you get stuck, check SETUP.md for detailed troubleshooting.
