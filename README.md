# AlphaTraderLab 🤖📈

**A Reinforcement Learning Trading Laboratory for Beginners**

---

## 🎯 What is AlphaTraderLab?

AlphaTraderLab is an educational project that teaches you how to build AI trading agents using **Reinforcement Learning (RL)**. 

Think of it like teaching a robot to trade:
- The robot (RL agent) observes the market (historical price data)
- It decides what to do: stay FLAT, go LONG (bet on price going up), or go SHORT (bet on price going down)
- It learns from its mistakes and successes to get better over time

**Complete Steps:**
- ✅ **Step 1**: Trading environment skeleton with random agent demo
- ✅ **Step 2**: PPO training and evaluation with baselines

---

## 📁 Project Structure

```
alpha_trader_lab/
├── envs/                              # Trading environment code
│   ├── __init__.py
│   ├── trading_env.py                # Main environment (Gymnasium-style)
│   └── ENVIRONMENT_GUIDE.md          # Technical deep-dive
├── utils/                             # Utility functions
│   ├── __init__.py
│   └── evaluation.py                 # Agent evaluation utilities
├── notebooks/                         # Jupyter notebooks for experiments
│   ├── AlphaTraderLab_v0.ipynb       # Step 1: Random agent demo
│   └── AlphaTraderLab_PPO_v1.ipynb   # Step 2: PPO training & evaluation
├── models/                            # Trained models (created during training)
├── data/                              # Data folder (populated by yfinance)
├── train_ppo.py                       # Python script for PPO training
├── requirements.txt                   # Python dependencies
├── PPO_GUIDE.md                       # Comprehensive PPO training guide
└── README.md                          # This file
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Easiest!)

1. Open the notebook in Google Colab:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click `File > Upload notebook`
   - Upload `notebooks/AlphaTraderLab_v0.ipynb`

2. The notebook will automatically:
   - Install all dependencies
   - Download market data
   - Run a demo with a random agent

3. Just run all cells from top to bottom! ▶️

### Option 2: Local Installation

#### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

#### Installation Steps

1. **Clone or download this project** to your computer

2. **Open a terminal** and navigate to the project folder:
   ```bash
   cd path/to/alpha_trader_lab
   ```

3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Mac/Linux:
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch Jupyter**:
   ```bash
   jupyter notebook notebooks/AlphaTraderLab_v0.ipynb
   ```

6. **Run all cells** in the notebook!

---

## 🧪 What Can You Do?

### Step 1: Environment & Random Agent (`AlphaTraderLab_v0.ipynb`)
1. **Data Download**: Fetches historical Bitcoin (BTC-USD) price data using `yfinance`
2. **Environment Creation**: Sets up the trading environment with this data
3. **Random Agent Test**: Runs a simple agent that takes random actions
4. **Visualization**: Shows price charts and equity curves

### Step 2: PPO Training & Evaluation (`AlphaTraderLab_PPO_v1.ipynb` or `train_ppo.py`)
1. **Data Splitting**: Split data into training (70%) and testing (30%) periods
2. **PPO Training**: Train a smart RL agent using Stable-Baselines3
3. **Evaluation**: Compare PPO vs Random vs Buy & Hold strategies
4. **Visualization**: Equity curves, action distributions, performance tables
5. **Model Saving**: Save trained models for later use

**Quick Start for Step 2**:
```bash
# Train PPO agent (100k steps, ~5 minutes)
python train_ppo.py

# Or open the notebook for detailed walkthrough
jupyter notebook notebooks/AlphaTraderLab_PPO_v1.ipynb
```

---

## 📚 Understanding the Code

### TradingEnv Explained

The `TradingEnv` class (in `envs/trading_env.py`) is the core of this project. Here's what it does:

#### Observation Space
The agent sees:
- **Last N candles**: Historical OHLCV (Open, High, Low, Close, Volume) data
- **Current position**: Whether it's FLAT, LONG, or SHORT
- **Portfolio performance**: Current equity vs. starting balance

#### Action Space
The agent can choose from 3 actions:
- `0` = **FLAT**: No position (sitting in cash)
- `1` = **LONG**: Betting the price will go up
- `2` = **SHORT**: Betting the price will go down

#### Reward Function
```
reward = (equity_after - equity_before) / initial_balance
```
- Positive reward = the action made money
- Negative reward = the action lost money
- Transaction costs are subtracted when changing positions

#### Simple Trading Logic
- When you go LONG and price goes up → you make money
- When you go LONG and price goes down → you lose money
- When you go SHORT and price goes down → you make money
- When you go SHORT and price goes up → you lose money

---

## 🔧 Configuration Options

You can customize the environment when creating it:

```python
from envs.trading_env import TradingEnv

env = TradingEnv(
    df=your_dataframe,           # Historical OHLCV data
    window_size=30,              # How many past candles the agent sees
    initial_balance=10000.0,     # Starting portfolio value ($10,000)
    transaction_cost=0.001       # Trading fee (0.1% per trade)
)
```

---

## 📊 Project Roadmap

- ✅ **Step 1**: Trading environment skeleton with random agent
- ✅ **Step 2**: PPO training and evaluation framework
- 🔜 **Step 3**: Advanced features (technical indicators, multiple assets)
- 🔜 **Step 4**: Comprehensive backtesting and risk metrics
- 🔜 **Step 5**: Live paper trading (simulated real-time trading)

### Learn More
- **Step 2 Details**: See [PPO_GUIDE.md](PPO_GUIDE.md) for comprehensive PPO training guide
- **Environment Details**: See [ENVIRONMENT_GUIDE.md](envs/ENVIRONMENT_GUIDE.md) for technical documentation
- **Setup Help**: See [SETUP.md](SETUP.md) for installation troubleshooting

---

## ⚠️ Disclaimer

**This is an educational project for learning purposes only.**

- Do NOT use this to trade real money without extensive testing
- Trading cryptocurrencies and financial instruments involves risk
- Past performance does not guarantee future results
- The authors are not responsible for any financial losses

---

## 🤝 Contributing

This is a learning project! If you find bugs or have suggestions:
1. Understand that this is designed for beginners
2. Keep code simple and well-commented
3. Test your changes with the notebook

---

## 📖 Learning Resources

If you're new to RL or trading, check out these resources:

### Reinforcement Learning
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

### Algorithmic Trading
- [QuantStart](https://www.quantstart.com/)
- [Investopedia - Algorithmic Trading](https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp)

---

## 📝 License

MIT License - Feel free to use this for learning and experimentation.

---

## 🙏 Acknowledgments

- **Gymnasium**: For the amazing RL environment API
- **Stable-Baselines3**: For making RL accessible
- **yfinance**: For easy access to market data
- **The RL Community**: For all the educational resources

---

**Happy Learning! 🚀**

*Remember: The goal is to learn, not to get rich quick. Master the fundamentals first.*
