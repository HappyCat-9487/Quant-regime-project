from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.backtest import backtest_from_signal
from src.metrics import sharpe

@dataclass(frozen=True)
class WalkForwardConfig:
    train_years: int
    test_years: int
    fee_bps: float
    periods_per_year: int=252
    
def _year_slices(index: pd.DatetimeIndex, train_years: int, test_years: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    years = sorted(set(index.year))
    slices = []
    step = test_years
    for i in range(0, len(years) - train_years - test_years + 1, step):
        train_start_year = years[i]
        train_end_year = years[i + train_years - 1]
        test_start_year = years[i + train_years]
        test_end_year = years[i + train_years + test_years - 1]
        
        train_start = pd.Timestamp(f"{train_start_year}-01-01")
        train_end = pd.Timestamp(f"{train_end_year}-12-31")
        test_start = pd.Timestamp(f"{test_start_year}-01-01")
        test_end = pd.Timestamp(f"{test_end_year}-12-31")
        
        slices.append((train_start, train_end, test_start, test_end))
    return slices


def walkforward_select_param(
    df: pd.DataFrame, 
    build_features: Callable[[pd.DataFrame, int], pd.DataFrame],
    signal_col: str,
    para_grid: List[int],
    cfg: Optional[WalkForwardConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each WF window:
      - build features on full df for each param (safe here since features use rolling past values)
      - choose param with best train Sharpe
      - evaluate chosen param on test window
    Returns a table of window results and an out-of-sample equity curve.
    """
    if cfg is None:
        cfg = WalkForwardConfig(train_years=5, test_years=1, fee_bps=1.0)

    df = df.copy().dropna()
    windows = _year_slices(df.index, cfg.train_years, cfg.test_years)
    
    rows = []
    oos_parts = []
    
    for(tr_s, tr_e, te_s, te_e) in windows:
        tr = df.loc[tr_s:tr_e]
        te = df.loc[te_s:te_e]
        if len(tr) < 252 or len(te) < 100:
            continue
        
        best_score = -np.inf
        best_para = None
        
        for para in para_grid:
            dff = build_features(df, para)
            bt = backtest_from_signal(dff, signal_col=signal_col, fee_bps=cfg.fee_bps)
            bt_tr = bt.loc[tr_s:tr_e].dropna(subset=["strat_ret"])
            score = sharpe(bt_tr["strat_ret"], periods=cfg.periods_per_year)
            if score > best_score:
                best_score = score
                best_para = para
                
        # evaluate on test with best_param
        dff_best = build_features(df, best_para)
        bt_best = backtest_from_signal(dff_best, signal_col=signal_col, fee_bps=cfg.fee_bps)
        bt_te = bt_best.loc[te_s:te_e].dropna(subset=["strat_ret"]).copy()
        
        # Store OOS piece
        oos_parts.append(bt_te["strat_ret"])
        
        rows.append({
            "train_start": tr_s.date(),
            "train_end": tr_e.date(),
            "test_start": te_s.date(),
            "test_end": te_e.date(),
            "best_para": best_para,
            "train_sharpe": float(best_score),
            "test_sharpe": float(sharpe(bt_te["strat_ret"], periods=cfg.periods_per_year)),
            "test_total_return": float((1+ bt_te["strat_ret"]).prod() - 1),
        })
    
    res = pd.DataFrame(rows)
    if oos_parts:
        oos = pd.concat(oos_parts).sort_index().to_frame(name="strat_ret")
        oos["oos_equity"] = (1.0 + oos["strat_ret"].fillna(0.0)).cumprod()
    else:
        oos = pd.DataFrame(columns=["strat_ret", "oos_equity"])
        
    return res, oos