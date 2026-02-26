from __future__ import annotations
import pandas as pd
import numpy as np

def backtest_from_signal(df: pd.DataFrame,signal_col: str, ret_col: str = "ret_1d",
                         fee_bps: float = 1.0,) -> pd.DataFrame:
    """
    Basic daily backtest: 
    - position = signal shifted by 1 day (avoid look-ahead)
    - apply transaction cost on turnover
    fee_bps: per trade cost in basis points(1 bps = 0.01%)
    """
    out = df.copy()
    
    # It is crucial that shift(1) is "places orders today using yesterday's signals"
    pos = out[signal_col].shift(1).fillna(0.0)
    
    #turnover: position change magnitude
    turnover = pos.diff().abs().fillna(0.0)
    
    fee = (fee_bps / 10000) * turnover #bps -> fraction
    strat_ret = pos * out[ret_col] - fee
    
    out["pos"] = pos
    out["turnover"] = turnover
    out["fee"] = fee
    out["strat_ret"] = strat_ret
    out["equity"] = (1.0 + strat_ret.fillna(0.0)).cumprod()
    return out
    
def backtest_long_only(df: pd.DataFrame,
                       signal_col: str,
                       ret_col: str = "ret_1d",
                       fee_bps: float = 1.0) -> pd.DataFrame:
    """
    long-only:
        signal > 0 => pos=1
        else => pos=0
    Still shift(1) to avoid look-ahead
    """
    out = df.copy()
    sig = out[signal_col].fillna(0.0)
    pos_raw = (sig > 0).astype(float)
    pos = pos_raw.shift(1).fillna(0.0)
    
    turnover = pos.diff().abs().fillna(0.0)
    fee = (fee_bps / 10000) * turnover #bps -> fraction
    strat_ret = pos * out[ret_col] - fee
    
    out["pos"] = pos
    out["turnover"] = turnover
    out["fee"] = fee
    out["strat_ret"] = strat_ret
    out["equity"] = (1.0 + strat_ret.fillna(0.0)).cumprod()
    return out

def backtest_long_only_with_regime(df: pd.DataFrame,
    signal_col: str,
    regime_col: str,
    regime_active: int = 0,
    ret_col: str = "ret_1d",
    fee_bps: float = 1.0) -> pd.DataFrame:
    """
    long-only with regime filtering:
        - only trade when df[regime_col] == regime_active
        - and signal > 0
        - else stay flat
    """
    out = df.copy()
    sig = out[signal_col].fillna(0.0)
    active = (out[regime_col].fillna(1).astype(int) == int(regime_active))
    pos_raw = ((sig > 0) & active).astype(float)
    pos = pos_raw.shift(1).fillna(0.0)
    
    turnover = pos.diff().abs().fillna(0.0)
    fee = (fee_bps / 10000.0) * turnover  #bps -> fraction
    strat_ret = pos * out[ret_col] - fee
    
    out["pos"] = pos
    out["turnover"] = turnover
    out["fee"] = fee
    out["strat_ret"] = strat_ret
    out["equity"] = (1.0 + strat_ret.fillna(0.0)).cumprod()
    return out