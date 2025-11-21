# Step 3.2 - Multi-Asset Single-Agent PPO

## Overview

**Step 3.2** extends the AlphaTraderLab pipeline to run the same single-agent PPO training and evaluation across **multiple assets** (tickers), enabling comparison of agent performance across different markets.

**What Step 3.2 IS:**
- ✅ Same **single-agent PPO** architecture (no changes to core RL algorithm)
- ✅ Same **single-asset environment** per run (TradingEnv unchanged)
- ✅ **Multi-asset comparison table** aggregating results across tickers
- ✅ Standardized pipeline for testing on cryptocurrencies, equities, ETFs, etc.

**What Step 3.2 IS NOT:**
- ❌ NOT population-based training
- ❌ NOT multi-agent competition ("arena")
- ❌ NOT a multi-asset portfolio environment
- ❌ NOT Step 4 (future: evolutionary/population methods)

**Goal:** Validate that the Step 3.1 setup (enriched indicators + trading costs) is robust across different asset classes before investing in more complex multi-agent architectures.

---

## What's New in Step 3.2

### 1. Asset Universe Configuration

**File:** `config/assets.py`

Defines the canonical list of assets for multi-asset experiments:

```python
DEFAULT_ASSETS = [
    "BTC-USD",   # Bitcoin (cryptocurrency)
    "SPY",       # S&P 500 ETF (US large-cap stocks)
    "QQQ",       # Nasdaq-100 ETF (US tech stocks)
    "IWM",       # Russell 2000 ETF (US small-cap stocks)
]
```

**Functions:**
- `normalize_ticker_to_slug(ticker)`: Converts `"BTC-USD"` → `"btc_usd"` for filenames
- `get_asset_display_name(ticker)`: Returns human-readable names

### 2. Refactored Training Pipeline

**File:** `train_ppo.py`

Added `run_experiment_for_ticker()` function that can be called programmatically:

```python
results = run_experiment_for_ticker(
    ticker="SPY",
    timesteps=100_000,
    use_indicators=True,
    transaction_cost_pct=0.001,
    switch_penalty=0.0005,
    output_dir="results"
)
```

**Behavior:**
- Downloads and splits data for the specified ticker
- Trains PPO agent (or skips if `skip_training=True`)
- Evaluates PPO, Random, Buy & Hold
- Saves per-asset results with standardized naming

**Backward Compatibility:**
- `train_ppo.py` CLI still works exactly as before for single-asset runs
- No breaking changes to existing Step 3.1 workflows

### 3. Multi-Asset Runner Script

**File:** `run_multi_assets.py`

Main script for running experiments across multiple assets.

**Usage:**

```bash
# Run on default assets (BTC-USD, SPY, QQQ, IWM)
python run_multi_assets.py --timesteps 20000

# Run on custom assets
python run_multi_assets.py --tickers BTC-USD,SPY,QQQ --timesteps 50000

# With trading costs
python run_multi_assets.py --tickers BTC-USD,SPY --timesteps 20000 \
  --transaction-cost 0.001 --switch-penalty 0.0005
```

**Features:**
- Accepts all Step 3.1 CLI flags (hyperparameters, costs, indicators)
- Runs experiments sequentially (one asset at a time)
- Continues on error (if one asset fails, others still process)
- Generates multi-asset comparison table automatically

### 4. Per-Asset Result Files

**Naming Convention:**

```
results/
├── ppo_results_btc_usd.csv       # PPO episode history for BTC-USD
├── agent_comparison_btc_usd.csv  # Agent comparison for BTC-USD
├── ppo_results_spy.csv           # PPO episode history for SPY
├── agent_comparison_spy.csv      # Agent comparison for SPY
├── ppo_results_qqq.csv
├── agent_comparison_qqq.csv
└── multi_asset_comparison.csv    # Aggregated summary (all assets)
```

**File Descriptions:**
- `ppo_results_{asset}.csv`: Per-episode equity, position, reward history for PPO
- `agent_comparison_{asset}.csv`: Comparison table for PPO, Random, Buy & Hold on this asset
- `multi_asset_comparison.csv`: Master table with all assets and agents

### 5. Multi-Asset Comparison Table

**File:** `utils/multi_asset_summary.py`

Aggregates per-asset results into a single comparison table.

**Output:** `results/multi_asset_comparison.csv`

**Columns:**
- `Asset`: Ticker symbol (e.g., "BTC-USD", "SPY")
- `Agent`: Agent type ("PPO", "Buy & Hold", "Random")
- `Final Equity`: Final portfolio value
- `Total Return (%)`: Percentage return
- `Max Drawdown (%)`: Maximum peak-to-trough decline
- `Sharpe Ratio`: Risk-adjusted return
- `Volatility`: Standard deviation of returns
- `Win Rate (%)`: Percentage of winning trades
- `Avg Win (%)`: Average winning trade size
- `Avg Loss (%)`: Average losing trade size
- `Trades`: Number of trades executed

**Summary Insights:**

The utility also prints:
1. **PPO Sharpe Ratio Ranking**: Which assets PPO performs best on (by Sharpe)
2. **PPO vs Buy & Hold**: Which assets PPO beats/loses to Buy & Hold
3. **PPO vs Random**: Baseline comparison (PPO should beat Random on all assets)

---

## Usage

### Quick Start (3 Assets, 20k Timesteps)

```bash
python run_multi_assets.py \
  --tickers BTC-USD,SPY,QQQ \
  --timesteps 20000 \
  --use-indicators \
  --transaction-cost 0.001 \
  --switch-penalty 0.0005
```

**Expected Runtime:** ~5-10 minutes (depends on CPU)

### Full Run (4 Assets, 100k Timesteps)

```bash
python run_multi_assets.py \
  --timesteps 100000 \
  --use-indicators \
  --transaction-cost 0.001 \
  --switch-penalty 0.0005
```

**Expected Runtime:** ~30-60 minutes

### Custom Asset Selection

```bash
# Crypto-only
python run_multi_assets.py --tickers BTC-USD,ETH-USD --timesteps 50000

# Equities-only
python run_multi_assets.py --tickers SPY,QQQ,IWM,DIA --timesteps 50000

# Mixed
python run_multi_assets.py --tickers BTC-USD,SPY,GLD --timesteps 50000
```

### With Custom Hyperparameters

```bash
python run_multi_assets.py \
  --tickers BTC-USD,SPY,QQQ \
  --timesteps 50000 \
  --learning-rate 1e-4 \
  --gamma 0.995 \
  --ent-coef 0.02 \
  --transaction-cost 0.001 \
  --switch-penalty 0.0005
```

### Evaluate Existing Models Only

```bash
# Skip training, just evaluate existing models
python run_multi_assets.py --tickers BTC-USD,SPY,QQQ --test
```

---

## Output Files

### Per-Asset Files

For each asset, you'll get:

**1. `ppo_results_{asset}.csv`** - Episode-level history
```csv
step,equity,position,action,reward
0,10000.00,0,0,0.0
1,10050.23,1,1,0.005023
...
```

**2. `agent_comparison_{asset}.csv`** - Agent comparison
```csv
Agent,Final Equity,Total Return (%),Max Drawdown (%),Sharpe Ratio,Trades
Buy & Hold,$15,000.00,50.00%,-20.00%,1.20,1
PPO,$14,500.00,45.00%,-25.00%,1.10,120
Random,$6,000.00,-40.00%,-50.00%,-0.80,300
```

### Multi-Asset Summary

**`multi_asset_comparison.csv`** - Aggregated results across all assets

Example:

| Asset   | Agent      | Final Equity | Total Return (%) | Max Drawdown (%) | Sharpe Ratio | Trades |
|---------|------------|--------------|------------------|------------------|--------------|--------|
| BTC-USD | Buy & Hold | $22,973.08   | 141.53%          | -28.14%          | 1.24         | 1      |
| BTC-USD | PPO        | $13,968.24   | 37.61%           | -32.04%          | 0.65         | 158    |
| BTC-USD | Random     | $3,244.36    | -68.44%          | -71.59%          | -1.59        | 324    |
| SPY     | Buy & Hold | $18,500.00   | 85.00%           | -15.00%          | 1.80         | 1      |
| SPY     | PPO        | $16,200.00   | 62.00%           | -18.00%          | 1.50         | 95     |
| SPY     | Random     | $7,800.00    | -22.00%          | -35.00%          | -0.60        | 280    |
| ...     | ...        | ...          | ...              | ...              | ...          | ...    |

---

## Interpretation

### What to Look For

**1. PPO Robustness Across Assets**
- ✅ Good: PPO beats Random on **all** assets (positive learning signal)
- ✅ Good: PPO has positive Sharpe on **most** assets
- ⚠️ Warning: PPO loses to Random on any asset → Investigate hyperparameters

**2. Asset-Specific Patterns**
- **High volatility** (BTC-USD): Higher returns but also higher drawdowns
- **Low volatility** (SPY, QQQ): Lower returns, more stable
- **Different time horizons**: Some assets may require longer training (100k+ timesteps)

**3. PPO vs Buy & Hold**
- ✅ PPO outperforms B&H on some assets → Good signal for active trading
- ⚠️ PPO underperforms B&H on all assets → May need:
  - Longer training (100k-500k timesteps)
  - Different cost parameters
  - Asset-specific hyperparameter tuning

**4. Trade Frequency**
- **Too many trades** (300+): May be over-trading despite costs
  - Try increasing `transaction_cost` and `switch_penalty`
- **Too few trades** (<20): May be "Buy & Hold in disguise"
  - Try decreasing costs or increasing `ent_coef` for more exploration

---

## Expected Results (20k Timesteps Smoke Test)

Based on our smoke tests, here's what to expect:

| Metric | BTC-USD | SPY | QQQ |
|--------|---------|-----|-----|
| **PPO Return** | +30% to +50% | +20% to +40% | +25% to +45% |
| **PPO Sharpe** | 0.5 to 1.0 | 0.8 to 1.5 | 0.7 to 1.3 |
| **PPO Trades** | 100-200 | 80-150 | 90-160 |
| **PPO vs Random** | ✅ PPO wins | ✅ PPO wins | ✅ PPO wins |
| **PPO vs B&H** | Varies | Varies | Varies |

**Note:** With only 20k timesteps, PPO is still learning. For fully trained agents, use **100k-500k timesteps**.

---

## Design Decisions

### 1. Why Sequential Processing (Not Parallel)?

**Decision:** Process assets one at a time in a loop.

**Rationale:**
- **Simplicity**: Easier to debug and monitor progress
- **Resource management**: Large neural networks can saturate memory if run in parallel
- **Educational**: Clear output flow, easy to follow
- **Future-proof**: Easy to add parallel processing later if needed (e.g., using `multiprocessing`)

### 2. Why Not a Multi-Asset Portfolio Environment?

**Decision:** Each experiment uses a single-asset TradingEnv.

**Rationale:**
- **Step 3.2 scope**: Focus is on comparing agent performance **across** assets, not within a portfolio
- **Portfolio management**: Requires different action space (asset allocation) and reward structure
- **Complexity**: Multi-asset portfolios introduce correlation, rebalancing, and position sizing challenges
- **Future work**: Portfolio environments are a good candidate for Step 5 or beyond

### 3. Why Default to 4 Assets (BTC-USD, SPY, QQQ, IWM)?

**Decision:** Small, diverse set of liquid assets.

**Rationale:**
- **Coverage**: Crypto (BTC), large-cap (SPY), tech (QQQ), small-cap (IWM)
- **Liquidity**: All highly liquid, well-known tickers
- **Data availability**: yfinance has complete history for all
- **Runtime**: 4 assets × 100k timesteps ≈ 40-60 minutes (manageable)

### 4. Why Not Asset-Specific Hyperparameters?

**Decision:** Use same hyperparameters for all assets (for now).

**Rationale:**
- **Step 3.2 goal**: Test **robustness** with consistent settings
- **Simplicity**: Easier to compare results (apples-to-apples)
- **Future work**: Asset-specific tuning is a good extension (Step 3.3 or beyond)

---

## Troubleshooting

### Problem: Some assets fail to download

**Possible causes:**
- Ticker doesn't exist or is delisted
- yfinance API issues or rate limiting
- Network connectivity issues

**Solution:**
- Check ticker symbol (use Yahoo Finance website to verify)
- Try again later (API may be temporarily unavailable)
- Use `--tickers` flag to exclude problematic assets

### Problem: PPO underperforms Random on all assets

**Possible causes:**
- Training timesteps too low (try 100k-500k)
- Hyperparameters not tuned for this asset class
- Costs too high (agent penalized for trading)

**Solution:**
- Increase `--timesteps` to 100000 or more
- Try lower costs: `--transaction-cost 0.0005 --switch-penalty 0.0`
- Experiment with higher exploration: `--ent-coef 0.02`

### Problem: "File not found" error in multi_asset_summary.py

**Cause:** Per-asset comparison files weren't created (training failed).

**Solution:**
- Check console output for errors during training
- Verify that `results/agent_comparison_{asset}.csv` files exist
- Re-run experiments for failed assets

---

## Next Steps

After Step 3.2, possible directions:

1. **Longer training**: Run full 500k-1M timestep experiments
2. **Asset-specific tuning**: Use Optuna for per-asset hyperparameter optimization
3. **More assets**: Expand to 10+ assets (commodities, currencies, bonds)
4. **Ensemble analysis**: Combine PPO agents from multiple assets
5. **Step 4 (Future)**: Population-based training with agent competition

---

## Backward Compatibility

✅ **Step 3.1 single-asset behavior fully preserved:**

```bash
# This still works exactly as before
python train_ppo.py --timesteps 20000 --ticker BTC-USD
```

✅ **All existing CLI flags supported**

✅ **No changes to TradingEnv or core RL logic**

---

## File Structure Summary

```
alpha_trader_lab/
├── config/
│   ├── __init__.py
│   └── assets.py                      # NEW: Asset universe config
├── envs/
│   └── trading_env.py                 # Unchanged
├── utils/
│   ├── evaluation.py                  # Unchanged
│   └── multi_asset_summary.py         # NEW: Multi-asset aggregation
├── train_ppo.py                       # Modified: Added run_experiment_for_ticker()
├── run_multi_assets.py                # NEW: Multi-asset runner script
├── results/                           # NEW: Per-asset and summary CSVs
│   ├── ppo_results_btc_usd.csv
│   ├── agent_comparison_btc_usd.csv
│   ├── ppo_results_spy.csv
│   ├── agent_comparison_spy.csv
│   └── multi_asset_comparison.csv
├── STEP3_README.md                    # Step 3 docs
├── STEP3_1_README.md                  # Step 3.1 docs
└── STEP3_2_README.md                  # This file
```

---

## References

- **Step 3 README**: `STEP3_README.md` - Base indicator system
- **Step 3.1 README**: `STEP3_1_README.md` - Enhanced features and costs
- **Asset Universe**: `config/assets.py` - Default ticker list
- **Multi-Asset Summary**: `utils/multi_asset_summary.py` - Aggregation logic
