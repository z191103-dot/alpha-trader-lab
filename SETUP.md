# 🚀 AlphaTraderLab - Setup Guide

This guide will help you get AlphaTraderLab up and running on your machine or in Google Colab.

---

## 📋 Prerequisites

Before you start, make sure you have:
- **Python 3.10 or higher** installed
- **pip** (Python package manager)
- **Git** (optional, for version control)
- Internet connection (to download market data)

---

## 🌐 Option 1: Google Colab (Easiest!)

Perfect for beginners who don't want to install anything locally.

### Steps:

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com/)

2. **Upload the notebook**:
   - Click `File > Upload notebook`
   - Navigate to `notebooks/AlphaTraderLab_v0.ipynb` and select it

3. **Upload the environment file**:
   - When you reach the cell that asks for file upload
   - Select `envs/trading_env.py` from your computer

4. **Run all cells**:
   - Click `Runtime > Run all` or press `Ctrl+F9`
   - Wait for the results!

### Advantages:
- ✅ No local installation needed
- ✅ Free GPU access (for future RL training)
- ✅ Easy to share and collaborate
- ✅ Runs in the cloud

---

## 💻 Option 2: Local Installation

For users who prefer to run everything on their own machine.

### Step 1: Install Python

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. ⚠️ **Important**: Check "Add Python to PATH" during installation

**Mac:**
```bash
# Using Homebrew (recommended)
brew install python@3.10

# Or download from python.org
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3-pip

# Fedora
sudo dnf install python3.10
```

### Step 2: Verify Installation

Open a terminal/command prompt and run:
```bash
python --version
# Should show: Python 3.10.x or higher

pip --version
# Should show pip version
```

### Step 3: Download the Project

**Option A: Using Git (recommended)**
```bash
git clone <your-repo-url>
cd alpha_trader_lab
```

**Option B: Manual Download**
1. Download the project as a ZIP file
2. Extract it to a folder
3. Open terminal in that folder

### Step 4: Create a Virtual Environment

A virtual environment keeps your project dependencies isolated.

```bash
# Navigate to the project folder
cd alpha_trader_lab

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt when activated.

### Step 5: Install Dependencies

```bash
# Make sure you're in the alpha_trader_lab folder
# and your virtual environment is activated

pip install -r requirements.txt
```

This will install:
- numpy (numerical computing)
- pandas (data handling)
- matplotlib (plotting)
- yfinance (market data)
- gymnasium (RL framework)
- stable-baselines3 (RL algorithms)
- scipy (scientific computing)

### Step 6: Test the Installation

Run the test script:
```bash
python test_env.py
```

You should see:
```
🎉 All tests passed!
✅ Your TradingEnv is working correctly!
```

### Step 7: Launch Jupyter Notebook

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter notebook
```

This will open Jupyter in your web browser.

Navigate to: `notebooks/AlphaTraderLab_v0.ipynb`

---

## 🛠️ Troubleshooting

### Problem: "python: command not found"

**Solution**: 
- Windows: Make sure Python is in your PATH
- Mac/Linux: Try `python3` instead of `python`

### Problem: "pip: command not found"

**Solution**:
- Try `python -m pip` instead of `pip`
- Or: `python3 -m pip`

### Problem: "No module named 'gymnasium'"

**Solution**:
- Make sure your virtual environment is activated
- Run: `pip install -r requirements.txt` again

### Problem: yfinance download fails

**Solution**:
- Check your internet connection
- Try a different ticker symbol
- The Yahoo Finance API sometimes has rate limits—wait a few minutes and try again

### Problem: Jupyter kernel crashes

**Solution**:
- Restart the kernel: `Kernel > Restart`
- Close other memory-intensive programs
- Try running fewer steps in the demo (change `num_steps = 200` to `num_steps = 50`)

### Problem: Import errors in Jupyter

**Solution**:
- Make sure you're running Jupyter from the activated virtual environment
- Restart the kernel after installing packages

---

## 📂 Project Structure Explained

```
alpha_trader_lab/
│
├── envs/                          # Environment code
│   ├── __init__.py               # Makes envs a Python package
│   └── trading_env.py            # Main TradingEnv class
│
├── notebooks/                     # Jupyter notebooks
│   └── AlphaTraderLab_v0.ipynb   # Step 1 demo notebook
│
├── data/                          # Data folder (initially empty)
│   └── (yfinance will populate this)
│
├── __init__.py                    # Makes alpha_trader_lab a package
├── requirements.txt               # Python dependencies
├── test_env.py                    # Quick test script
├── README.md                      # Project overview
├── SETUP.md                       # This file
└── .gitignore                     # Git ignore rules
```

---

## 🎯 Next Steps

Once everything is set up:

1. ✅ Run `test_env.py` to verify the environment works
2. ✅ Open and run `AlphaTraderLab_v0.ipynb` 
3. ✅ Experiment with different parameters:
   - Try different crypto assets (ETH-USD, BNB-USD)
   - Change the window size
   - Modify transaction costs
4. ✅ Read the code and comments to understand how it works
5. ✅ Wait for Step 2 where we'll train a real RL agent!

---

## 🆘 Getting Help

If you're stuck:

1. **Read the error message carefully** - it often tells you what's wrong
2. **Check the Troubleshooting section** above
3. **Google the error** - add "Python" or "pip" to your search
4. **Ask for help** - provide:
   - Your operating system
   - Python version (`python --version`)
   - The full error message
   - What you were trying to do

---

## 🎓 Learning Resources

### Python Basics
- [Python.org Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)

### Virtual Environments
- [Python venv Guide](https://docs.python.org/3/library/venv.html)

### Jupyter Notebooks
- [Jupyter Documentation](https://jupyter-notebook.readthedocs.io/)

### Reinforcement Learning
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

---

**Good luck! 🚀**
