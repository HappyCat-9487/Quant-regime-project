# Regime-Aware Quant Strategy Research (Yahoo Finance)

A reproducible **quant research + engineering** project that evaluates **Momentum** vs **Mean Reversion** signals under different **volatility regimes**, using daily OHLCV data from **Yahoo Finance**.

This repo emphasizes **robust methodology** (bias control, transaction costs, out-of-sample validation, parameter robustness) rather than headline returns.

> Not investment advice. Educational / research purpose only.

---

## What This Project Shows (for Recruiters)

- End-to-end research pipeline: **data → features → backtest → risk metrics → robustness checks → report artifacts**
- Realistic assumptions: **no look-ahead**, **transaction costs**, **long-only** (avoid unrealistic shorting on indices)
- Robustness work:
  - **Walk-forward (OOS) evaluation**
  - **Grid search** over key parameters
  - **Heatmaps** (Sharpe vs `mr_window × vol_q`) under different transaction costs
- Engineering:
  - Cached data (parquet)
  - **CLI** to generate reports
  - **Weights & Biases (W&B)** experiment tracking (tables + figures)

---

## Project Objective

1. Build a reproducible research pipeline from **data → signals → backtest → evaluation → robustness**.
2. Compare baseline strategies:
   - **Momentum (MOM)**: sign of past returns over a lookback window (tested as **long-only**)
   - **Mean Reversion (MR)**: z-score based contrarian signal (tested as **long-only**)
3. Test whether performance changes across **volatility regimes**:
   - **MR_lowvol_filter**: activate MR only in **low-vol** regimes
4. Validate robustness using:
   - **Walk-forward (out-of-sample) evaluation**
   - Transaction cost sensitivity (0.5 / 1.0 / 2.0 bps)
   - Parameter search + heatmaps
   - W&B logging for experiment tracking

---

## Data

- Source: Yahoo Finance (`yfinance`)
- Frequency: Daily
- Instruments (current):
  - `SPY` (US equity index ETF)
  - `2330.TW` (TSMC)
- Fields: Open / High / Low / Close / Volume
- Price adjustment: `auto_adjust=True` (dividends/splits adjusted)
- Caching: parquet under `data/` for reproducibility and speed

---

## Signals (Features)

### Returns
- `ret_1d = Close.pct_change()`

### Momentum (MOM)
- `mom = Close.pct_change(lookback)`
- `signal_mom = sign(mom)`
- Implemented as **long-only**:
  - positive signal → long (1)
  - negative signal → flat (0)

### Mean Reversion (MR)
- Rolling z-score of Close vs rolling mean/std:
  - `z = (Close - rolling_mean) / rolling_std`
- `signal_mr = -sign(z)` (contrarian)
- Implemented as **long-only**

### Volatility Regime
- Rolling volatility of returns:
  - `vol = std(ret_1d, window=vol_window)`
- Define high volatility using a rolling quantile threshold:
  - `high_vol = vol > rolling_quantile(vol, q=vol_q)`
- **MR_lowvol_filter** trades only when `high_vol == 0` (low-vol)

---

## Backtesting Assumptions

- **No look-ahead bias**: positions are shifted by 1 day  
  (trade today using yesterday’s signal)
- **Transaction costs**: applied on turnover (position changes), configurable in bps
- Outputs include:
  - strategy returns (`strat_ret`)
  - equity curve (`equity`)
  - drawdown curve (`drawdown`)
  - turnover, fees

---

## Evaluation Metrics

- Annualized return (reference)
- Annualized volatility
- Sharpe ratio (annualized)
- Max drawdown (MaxDD)
- Calmar ratio (AnnRet / |MaxDD|)

---

## Robustness Checks

### Walk-forward (Out-of-sample)
Rolling train/test evaluation to assess OOS behavior (avoid overfitting by evaluating on future windows).

### Parameter & Cost Sensitivity (Grid Search)
Grid search over:
- `mr_window ∈ {10, 20, 30, 50}`
- `vol_q ∈ {0.6, 0.7, 0.8}`
- `fee_bps ∈ {0.5, 1.0, 2.0}`

Artifacts:
- Heatmaps: Sharpe vs (`mr_window`, `vol_q`) under different transaction costs
- W&B tables for fast comparison and filtering

---

## Key Findings (High-level)

- **SPY**: MR_lowvol_filter tends to show a more contiguous “good parameter region” in Sharpe heatmaps; the overall pattern largely persists under higher transaction costs → suggests better robustness.
- **2330.TW**: MR_lowvol_filter appears more parameter-sensitive (patchier heatmaps) → weaker cross-market generalization; may require market-specific tuning.
- Regime filtering often improves **risk structure** (drawdowns / risk-adjusted metrics) even if raw returns may lag Buy&Hold in strong bull markets.

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```
### Generate metrics + figures (CSV + equity/drawdown plots)
```bash
python -m src.run --symbols SPY 2330.TW --fee_bps 1.0
```
### Run with Weights & Biases logging
```bash
python -m src.run --symbols SPY 2330.TW --fee_bps 1.0 --use_wandb --wandb_project quant-regime-project
```
---

## Repository Structure

```text
quant-regime-project
├── LICENSE
├── README.md
├── notebooks
│   ├── 01_data_check.ipynb
│   ├── 02_walkforward.ipynb
│   ├── 03_baseline_longonly_regime.ipynb
│   ├── 04_report_metrics.ipynb
│   ├── 05_linkedin_summary_plot.ipynb
│   ├── 07_grid_search_wandb.ipynb
│   ├── console.ipynb
│   └── reports
├── reports
│   ├── figures
│   └── summary_metrics.csv
├── requirements.txt
└── src
    ├── backtest.py
    ├── data.py
    ├── features.py
    ├── grid.py
    ├── metrics.py
    ├── run.py
    └── walkforward.py
```
---

## Notes / Limitations

- Daily data and simple execution assumptions; not modeling intraday microstructure, slippage beyond turnover-based bps, or market impact.

- Results are research-oriented; performance may change with data quality, execution assumptions, and regime definitions.