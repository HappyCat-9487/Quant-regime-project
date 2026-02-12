from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

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
    cfg: WalkForwardConfig = WalkForwardConfig()) -> pd.DataFrame:
    """
    For each WF window:
      - build features on full df for each param (safe here since features use rolling past values)
      - choose param with best train Sharpe
      - evaluate chosen param on test window
    Returns a table of window results and an out-of-sample equity curve.
    """