# Regime-Aware Quant Strategy Research (Yahoo Finance)

A research-driven quant project that evaluates **momentum** vs **mean-reversion** signals under different **volatility regimes**, using daily OHLCV data from **Yahoo Finance**.  
Focus is on **robust methodology** (bias control, transaction costs, walk-forward evaluation) rather than headline returns.

> Not investment advice. Educational/research purpose only.

---

## Project Objective

1. Build a reproducible research pipeline from **data → features → backtest → risk metrics → robustness checks**.
2. Compare baseline strategies:
   - **Momentum**: sign of past returns over a lookback window
   - **Mean Reversion**: z-score based contrarian signal
3. Test whether performance changes under **volatility regimes** (high-vol vs low-vol).
4. Validate robustness using **walk-forward (out-of-sample) evaluation** and realistic assumptions (transaction costs).

---

## Methodology

### Data
- Source: Yahoo Finance via `yfinance`
- Frequency: Daily
- Instruments: e.g. `SPY` (can extend to `QQQ`, `IWM`, etc.)
- Fields: Open / High / Low / Close / Volume  
- Adjustments: `auto_adjust=True` (dividends/splits adjusted)

### Signals (Features)
- **Returns**: daily close-to-close return
- **Momentum**:
  - `mom = Close.pct_change(lookback)`
  - `signal = sign(mom)`
- **Mean Reversion**:
  - rolling z-score of Close vs rolling mean/std
  - `signal = -sign(z)`
- **Volatility Regime**:
  - rolling volatility of returns
  - classify **high_vol** using a rolling quantile threshold

### Backtesting Assumptions
- **No look-ahead bias**: positions are shifted by 1 day (trade today using yesterday’s signal)
- **Transaction costs**: applied on turnover (position changes), configurable in **bps**
- Outputs:
  - strategy returns (`strat_ret`)
  - equity curve (`equity`)
  - turnover, fee

### Evaluation Metrics
- Sharpe ratio (annualized)
- Max drawdown
- Approx annualized return (for reference only)
- Average turnover (proxy for capacity / cost sensitivity)

---

## Key Findings (Example)
> Replace these bullets with your actual results and plots.

- Strategy performance varies significantly across **high-vol vs low-vol regimes**.
- Transaction costs materially affect high-turnover strategies.
- Walk-forward evaluation provides a more realistic view of out-of-sample stability.

---

## Repository Structure

```text
quant-regime-project/
  README.md
  requirements.txt
  data/                  # cached Yahoo Finance data (parquet)
  notebooks/
    01_data_check.ipynb
    02_first_backtest.ipynb
    03_walk_forward.ipynb
  src/
    data.py              # data loader + caching
    features.py          # signals + regime features
    backtest.py          # backtest engine (shift + cost)
    metrics.py           # sharpe, maxDD, summary
  reports/
    figures/             # exported charts for README/LinkedIn
