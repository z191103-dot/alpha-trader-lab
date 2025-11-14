# AlphaTraderLab 🤖📈

**A Reinforcement Learning Trading Laboratory for Beginners**

---

## 🎯 What is AlphaTraderLab?

AlphaTraderLab is an educational project that teaches you how to build AI trading agents using **Reinforcement Learning (RL)**. 

Think of it like teaching a robot to trade:
- The robot (RL agent) observes the market (historical price data)
- It decides what to do: stay FLAT, go LONG (bet on price going up), or go SHORT (bet on price going down)
- It learns from its mistakes and successes to get better over time

**This is Step 1**: We're building the basic skeleton and testing that everything works with a random agent (no learning yet).

---

## 📁 Project Structure

```
alpha_trader_lab/
├── envs/                          # Trading environment code
│   ├── __init__.py
│   └── trading_env.py            # Main environment (Gymnasium-style)
├── notebooks/                     # Jupyter notebooks for experiments
│   └── AlphaTraderLab_v0.ipynb   # Step 1: Demo with random agent
├── data/                          # Data folder (will be populated by yfinance)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
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

## 🧪 What Does the Demo Do?

The `AlphaTraderLab_v0.ipynb` notebook demonstrates:

1. **Data Download**: Fetches historical Bitcoin (BTC-USD) price data using `yfinance`
2. **Environment Creation**: Sets up the trading environment with this data
3. **Random Agent Test**: Runs a simple agent that takes random actions
4. **Visualization**: Shows:
   - The price chart
   - The equity curve (how the portfolio value changes over time)

**Important**: The random agent will perform poorly (that's expected!). In Step 2, we'll train a smart RL agent using PPO (Proximal Policy Optimization).

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

## 📊 Next Steps (Future Releases)

This is just **Step 1** of the AlphaTraderLab project. Here's what's coming:

- **Step 2**: Train a real RL agent (PPO from Stable-Baselines3)
- **Step 3**: Add more sophisticated features (technical indicators, multiple assets)
- **Step 4**: Backtesting framework and performance metrics
- **Step 5**: Live paper trading (simulated real-time trading)

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
