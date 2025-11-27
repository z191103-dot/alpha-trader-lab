# Step 3.3 - Automated PPO Evaluation Pipeline

## Overview

**Step 3.3** implements an automated evaluation pipeline for analyzing PPO agent performance from CSV outputs. This step focuses purely on **analysis tools** without modifying the RL environment, training logic, or reward function.

**What Step 3.3 Provides:**
- ✅ Automated performance metrics computation from CSVs
- ✅ Multi-asset comparison summaries
- ✅ Equity curve and drawdown visualizations
- ✅ Reproducible analysis pipeline (no more manual Excel work!)
- ✅ CLI tools for quick analysis

**What Step 3.3 Does NOT Do:**
- ❌ Does NOT modify the trading environment
- ❌ Does NOT change reward functions
- ❌ Does NOT alter PPO training hyperparameters
- ❌ Does NOT implement new trading strategies

---

## Key Insights from Analysis

Based on the automated analysis of our recent 20k timestep multi-asset run:

### BTC-USD (Cryptocurrency) - **Strong Performance** ✅

| Metric | Value | Assessment |
|--------|-------|------------|
| **Total Return** | +191.81% | ✅ Excellent |
| **Sharpe Ratio** | 1.38 | ✅ Good risk-adjusted returns |
| **Max Drawdown** | -20.95% | ✅ Moderate (better than Buy & Hold) |
| **Win Rate** | 33.3% | ⚠️ Low but acceptable given high avg win |
| **Avg Win/Loss** | +5.05% / -2.04% | ✅ Good risk/reward ratio (2.5:1) |
| **Number of Trades** | 108 | ✅ Reasonable trading activity |

**Conclusion:** PPO significantly outperforms Buy & Hold (+191.81% vs +62.71%) and Random agents on BTC-USD. The agent successfully captures high-volatility trends with manageable drawdown.

### SPY (S&P 500 ETF) - **Needs Improvement** ❌

| Metric | Value | Assessment |
|--------|-------|------------|
| **Total Return** | -29.28% | ❌ Poor |
| **Sharpe Ratio** | -1.51 | ❌ Negative risk-adjusted returns |
| **Max Drawdown** | -30.37% | ❌ Deep drawdown |
| **Win Rate** | 13.3% | ❌ Very low |
| **Avg Win/Loss** | +0.78% / -0.78% | ⚠️ Even risk/reward (1:1) |
| **Number of Trades** | 150 | ⚠️ Possibly over-trading |

**Conclusion:** PPO underperforms on SPY. While it beats Random agent (-29.28% vs -45.21%), it's far behind Buy & Hold (+68.54%). Possible issues:
- Training timesteps too low (20k) for lower-volatility assets
- Environment/reward not optimized for mean-reverting markets
- Over-trading small moves with poor win rate

### QQQ (Nasdaq-100 ETF) - **Barely Profitable** ⚠️

| Metric | Value | Assessment |
|--------|-------|------------|
| **Total Return** | +2.38% | ⚠️ Barely positive |
| **Sharpe Ratio** | -0.10 | ❌ Negative |
| **Max Drawdown** | -25.87% | ⚠️ Moderate-high |
| **Win Rate** | 23.6% | ❌ Low |
| **Avg Win/Loss** | +2.23% / -1.52% | ⚠️ Moderate (1.5:1) |
| **Number of Trades** | 165 | ⚠️ High trading frequency |

**Conclusion:** PPO marginally profitable on QQQ but poor risk-adjusted performance. Beats Random significantly (+2.38% vs -62.58%) but far behind Buy & Hold (+69.75%). Similar issues to SPY.

---

## What Was Implemented

### 1. Analysis Module Structure

```
analysis/
├── __init__.py                    # Module initialization
├── analyze_ppo_results.py         # Main analysis script (~500 lines)
└── plots.py                       # Visualization utilities (~350 lines)
```

### 2. Core Analysis Functions

**`analyze_ppo_results.py`** provides:

- **`compute_trades_from_position()`**: Detects trades from position changes
  - Entry: 0 → 1 (LONG) or 0 → 2 (SHORT)
  - Exit: Position changes or flips (1→2, 2→1, any→0)
  - Computes per-trade returns

- **`compute_max_drawdown()`**: Calculates maximum peak-to-trough decline
  - Uses running maximum of equity curve
  - Returns percentage drawdown (negative value)

- **`compute_sharpe_ratio()`**: Risk-adjusted return metric
  - Formula: `(mean_return / std_return) * sqrt(252)`
  - Annualized assuming 252 trading days
  - Assumes 0% risk-free rate

- **`analyze_ppo_results()`**: Main analysis function
  - Loads CSV with columns: step, equity, position, reward
  - Computes all metrics listed below
  - Returns dictionary of results

- **`analyze_multiple_assets()`**: Batch processing
  - Processes multiple tickers in parallel
  - Creates summary DataFrame
  - Handles missing files gracefully

### 3. Metrics Computed

For each asset, the pipeline computes:

| Metric | Description | Formula/Method |
|--------|-------------|----------------|
| **total_return** | Cumulative return | `(final_equity - initial_equity) / initial_equity * 100` |
| **avg_daily_return** | Mean per-step return | `mean(daily_returns)` |
| **volatility** | Std deviation of returns | `std(daily_returns)` |
| **sharpe_ratio** | Risk-adjusted returns | `mean(returns) / std(returns) * sqrt(252)` |
| **max_drawdown** | Maximum peak-to-trough | `min((equity - running_max) / running_max * 100)` |
| **num_trades** | Total trades executed | Count of position entries/exits |
| **win_rate** | % of winning trades | `(winning_trades / total_trades) * 100` |
| **avg_trade_return** | Mean trade return | `mean(trade_returns)` |
| **avg_win** | Mean winning trade | `mean(positive_returns)` |
| **avg_loss** | Mean losing trade | `mean(negative_returns)` |

### 4. Visualization Tools

**`plots.py`** provides three types of plots for each asset:

1. **Equity Curve** (`equity_curve_{asset}.png`)
   - Line plot of equity over time
   - Shows initial equity reference line
   - Currency-formatted y-axis

2. **Agent Comparison** (`agent_comparison_{asset}.png`)
   - Overlays PPO equity curve
   - Marks start/end points
   - Can be extended to overlay Buy & Hold / Random

3. **Drawdown Curve** (`drawdown_{asset}.png`)
   - Two-panel plot:
     - Top: Equity with running maximum
     - Bottom: Drawdown filled area (red when underwater)
   - Shows worst drawdown periods visually

---

## Usage

### Basic Analysis (Console Output)

```bash
# Analyze all default assets (BTC-USD, SPY, QQQ)
python analysis/analyze_ppo_results.py

# Analyze specific assets
python analysis/analyze_ppo_results.py --assets BTC-USD,ETH-USD

# Point to different results directory
python analysis/analyze_ppo_results.py --results-dir my_results/
```

**Output:** Prints summary table, agent comparisons, and key insights to console.

### Generate Plots

```bash
# Generate all plots for default assets
python analysis/plots.py

# Custom assets and output directory
python analysis/plots.py --assets BTC-USD,SPY --output-dir my_plots/

# Display plots (if display available)
python analysis/plots.py --show
```

**Output:** Saves 3 plots per asset to `analysis/plots/`:
- `equity_curve_{asset}.png`
- `agent_comparison_{asset}.png`
- `drawdown_{asset}.png`

---

## Example Output

### Summary Table

```
====================================================================================================
📊 PPO AGENT PERFORMANCE SUMMARY
====================================================================================================
  asset total_return avg_daily_return volatility sharpe_ratio max_drawdown num_trades win_rate avg_trade_return avg_win avg_loss initial_equity final_equity
BTC-USD      191.81%          0.0037%     0.0429         1.38      -20.95%        108    33.3%            1.25%   5.05%   -2.04%      $9,919.61   $28,946.73
    SPY      -29.28%         -0.0007%     0.0069        -1.51      -30.37%        150    13.3%           -0.00%   0.78%   -0.78%     $10,000.00    $7,072.19
    QQQ        2.38%         -0.0001%     0.0147        -0.10      -25.87%        165    23.6%            0.27%   2.23%   -1.52%     $10,063.49   $10,302.71
====================================================================================================
```

### Agent Comparison (per asset)

```
────────────────────────────────────────────────────────────────────────────────────────────────────
Asset: BTC-USD
────────────────────────────────────────────────────────────────────────────────────────────────────
     Agent Final Equity Total Return (%) Max Drawdown (%)  Sharpe Ratio  Win Rate (%)
       PPO   $28,946.73          191.81%          -20.95%          1.65        33.3%
Buy & Hold   $16,457.52           62.71%          -28.14%          0.83       100.0%
    Random    $3,654.14          -63.29%          -78.84%         -1.20        17.1%
```

### Key Insights

```
====================================================================================================
💡 KEY INSIGHTS
====================================================================================================

BTC-USD:
  • Total Return: 191.81%
  • Sharpe Ratio: 1.38
  • Number of Trades: 108
  • Win Rate: 33.3%
  ✅ Strong positive returns
  ✅ Good risk-adjusted performance

SPY:
  • Total Return: -29.28%
  • Sharpe Ratio: -1.51
  • Number of Trades: 150
  • Win Rate: 13.3%
  ❌ Negative returns - needs improvement
  ❌ Poor risk-adjusted performance
```

---

## File Structure

```
alpha_trader_lab/
├── analysis/                      # NEW: Analysis module
│   ├── __init__.py
│   ├── analyze_ppo_results.py    # Main analysis script
│   ├── plots.py                   # Plotting utilities
│   └── plots/                     # Generated plots (9 PNGs)
│       ├── equity_curve_btc_usd.png
│       ├── drawdown_btc_usd.png
│       ├── agent_comparison_btc_usd.png
│       └── ... (SPY, QQQ)
├── results/                       # Existing CSV outputs
│   ├── ppo_results_btc_usd.csv
│   ├── agent_comparison_btc_usd.csv
│   └── ... (SPY, QQQ)
├── STEP3_3_README.md              # This file
└── ... (other project files)
```

---

## Design Decisions

### 1. Why Separate Analysis Module?

**Decision:** Create `analysis/` directory instead of extending `utils/`.

**Rationale:**
- **Clear separation**: Analysis is distinct from training/evaluation utilities
- **Modular**: Can be run independently without loading RL code
- **Extensible**: Easy to add new analysis tools (Monte Carlo, regime detection, etc.)
- **Educational**: Clear entry points for understanding agent performance

### 2. Why Compute Trades from Position Changes?

**Decision:** Detect trades by monitoring `position` column changes.

**Rationale:**
- **Accurate**: Captures actual entry/exit points
- **Handles flips**: Correctly counts LONG→SHORT switches as two trades
- **Flexible**: Works with any position encoding (0/1/2, -1/0/1, etc.)
- **Reproducible**: Same logic as `utils/evaluation.py`

### 3. Why Annualized Sharpe Ratio?

**Decision:** Multiply by `sqrt(252)` to annualize Sharpe.

**Rationale:**
- **Industry standard**: Allows comparison with published strategies
- **Assumption**: ~252 trading days per year
- **Caveat**: Per-step returns in our environment != daily returns
  - Our "steps" are backtest steps, not calendar days
  - Annualization is approximate but consistent across runs

### 4. Why No Baseline CSV Reprocessing?

**Decision:** Use existing `agent_comparison_{asset}.csv` instead of recomputing Random/Buy & Hold.

**Rationale:**
- **Efficiency**: Baselines already computed during training
- **Consistency**: Uses exact same baseline runs
- **Simplicity**: One source of truth (Step 3.2 CSVs)
- **Extension**: Could add baseline recomputation if needed

---

## Limitations & Future Work

### Current Limitations

1. **Trade Detection**: Assumes specific position encoding (0, 1, 2)
   - Would need adjustment for -1/0/1 encoding

2. **Step ≠ Time**: Analysis treats each row as one "period"
   - In reality, steps may not correspond to calendar days
   - Sharpe annualization is approximate

3. **No Transaction Costs in Metrics**: Computed metrics don't explicitly account for costs
   - Costs are already reflected in equity curve
   - But per-trade returns don't show cost breakdown

4. **Single Agent**: Only analyzes PPO
   - Could extend to compare multiple RL algorithms (DQN, A2C, etc.)

### Potential Extensions (Step 3.4+)

1. **Regime Analysis**: Identify market regimes (trending, mean-reverting)
   - Compute performance stratified by regime
   - Helps explain why PPO works on BTC but not SPY

2. **Monte Carlo Validation**: Bootstrap confidence intervals
   - Determine if performance is statistically significant
   - Detect overfitting to specific test period

3. **Feature Importance**: Analyze which indicators matter
   - Correlate indicator values with trade outcomes
   - Guide Step 3.1 indicator selection

4. **Comparative Analysis**: Multi-algorithm benchmarking
   - Train DQN, A2C, SAC, etc. with same setup
   - Systematic comparison table

5. **Real-time Dashboard**: Streamlit/Dash web interface
   - Interactive equity curves
   - Dynamic metric filtering
   - Live training monitoring

---

## Troubleshooting

### Problem: "No module named 'config'"

**Cause:** Analysis scripts can't import from parent directory.

**Solution:** Already handled via `sys.path.insert(0, ...)` in scripts. If issues persist:
```bash
# Run from project root
cd /path/to/alpha_trader_lab
python analysis/analyze_ppo_results.py
```

### Problem: "CSV file not found"

**Cause:** Results CSVs don't exist for specified assets.

**Solution:**
```bash
# Check available CSVs
ls results/ppo_results_*.csv

# Run analysis only on available assets
python analysis/analyze_ppo_results.py --assets BTC-USD
```

### Problem: Plots don't display

**Cause:** Running in headless environment (no display).

**Solution:** Plots are automatically saved to `analysis/plots/` directory. Use `--show` flag only if display is available.

### Problem: Metrics seem wrong

**Cause:** CSV structure doesn't match expected format.

**Solution:** Verify CSV has required columns:
```python
import pandas as pd
df = pd.read_csv('results/ppo_results_btc_usd.csv')
print(df.columns.tolist())
# Expected: ['step', 'equity', 'position', 'action', 'reward', ...]
```

---

## Next Steps

Based on Step 3.3 analysis results:

### Immediate Actions (Step 3.4 Candidates)

1. **Increase Training Timesteps for SPY/QQQ**
   - Current: 20k timesteps
   - Try: 100k-500k timesteps
   - Hypothesis: Low-volatility assets need more exploration

2. **Adjust Reward Function**
   - Current: Equity change normalized by initial balance
   - Try: Sharpe-aware rewards, drawdown penalties
   - Hypothesis: Better align reward with desired outcomes (Sharpe, not just return)

3. **Tune Hyperparameters per Asset Class**
   - BTC (crypto): Current settings work well
   - SPY/QQQ (equity): May need:
     - Lower `ent_coef` (less exploration)
     - Different `learning_rate`
     - Longer `n_steps` (more context)

4. **Add More Technical Indicators**
   - Current: 23 features per candle (Step 3.1)
   - Consider: Regime indicators (trending vs mean-reverting)
   - For SPY/QQQ: Mean-reversion signals (RSI extremes, BB squeeze)

### Step 4 Preparation

Step 3.3 analysis provides **baseline metrics** for:
- Population-based training (Step 4)
- Multi-agent comparison
- Fitness function design (use Sharpe ratio for selection?)

---

## Reproducibility

All analysis is **deterministic** given the same input CSVs:
- No randomness in metric computation
- Pandas operations are reproducible
- Plots use fixed seeds (where applicable)

**To reproduce results:**
```bash
# 1. Run multi-asset training (Step 3.2)
python run_multi_assets.py --timesteps 20000 --assets BTC-USD,SPY,QQQ

# 2. Run analysis
python analysis/analyze_ppo_results.py

# 3. Generate plots
python analysis/plots.py

# 4. Results are identical every time (given same CSVs)
```

---

## References

- **Step 3.1 README**: `STEP3_1_README.md` - Enhanced features and costs
- **Step 3.2 README**: `STEP3_2_README.md` - Multi-asset training
- **Sharpe Ratio**: [Investopedia](https://www.investopedia.com/terms/s/sharperatio.asp)
- **Max Drawdown**: [Investopedia](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp)
