from __future__ import annotations
import pandas as pd
import numpy as np

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["Close"].pct_change()
    return out

def add_momentum_signal(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Simple momentum: Sign of past returns over a lookback period
    """
    out = df.copy()
    out["mom"] = out["ret_1d"].pct_change(lookback)
    out["signal_mom"] = np.sign(out["mom"]).replace({0: np.nan}).ffill().fillna(0.0)
    return out

def add_mean_reversion_signal(df: pd.DataFrame, z_window: int = 20) -> pd.DataFrame:
    """
    Mean reversion: Z-score of prices v.s. rolling mean/std
    signal = -sign(z) (z high => short, z low => long)
    """
    out = df.copy()
    roll_mean = out["Close"].rolling(window=z_window).mean()
    roll_std = out["Close"].rolling(window=z_window).std(ddof=0)
    z = (out["Close"] - roll_mean) / roll_std
    out["z"] = z
    out["signal_mr"] = (-np.sign(z)).replace({0: np.nan}).ffill().fillna(0.0)
    return out

def add_vol_regime(df: pd.DataFrame, vol_window: int = 20, q: float = 0.7) -> pd.DataFrame:
    """
    Volatility regime: high volatility => rolling std of returns > rolling quantile threshold
    """
    out = df.copy()
    r = out["Close"].pct_change()
    vol = r.rolling(vol_window).std(ddof=0)
    thresh = vol.rolling(252).quantile(q)
    out["vol"] = vol
    out["high_vol"] = (vol > thresh).astype(int)
    return out
