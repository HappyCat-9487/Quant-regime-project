from __future__ import annotations
import pandas as pd
import numpy as np

def sharpe(ret: pd.Series, periods: int = 252) -> float:
    r = ret.dropna()
    if r.std(ddof=0) == 0:
        return 0.0
    return (r.mean() / r.std(ddof=0)) * np.sqrt(periods)

def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return dd.min() if len(dd) else 0.0

def summary(bt: pd.DataFrame) -> dict:
    r = bt["strat_ret"]
    eq = bt["equity"]
    return {
        "sharpe": float(sharpe(r)),
        "max_drawdown": float(max_drawdown(eq)),
        "AnnRet(approx)": float(eq.iloc[-1] ** (252 / len(eq)) - 1.0 if len(eq) > 0 else 0.0),
        "AvgTurnover": float(bt["turnover"].mean()),
    }
    
def regime_summary(bt: pd.DataFrame, regime_col: str = "high_vol") -> dict:
    out = {}
    for k, sub in bt.dropna(subset=["strat_ret"]).groupby(regime_col):
        out[f"regime={k}"] = summary(sub)
    return out