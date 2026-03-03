from __future__ import annotations
import itertools
from dataclasses import asdict
import pandas as pd

from src.run import Config, run_symbol #reuse existing logic

def grid_search(
    symbols: list[str],
    base_cfg: Config,
    grid: dict[str, list],
) -> pd.DataFrame:
    """
    grid keys can include: mom_lookback, mr_window, vol_q, fee_bps, vol_window, regime_active_value
    returns a dataframe of results for all combinations and symbols.
    """
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    
    all_rows = []
    for values in combos:
        cfg_kwargs = asdict(base_cfg).copy()
        for key, value in zip(keys, values):
            cfg_kwargs[key] = value
        cfg = Config(**cfg_kwargs)
        
        for symbol in symbols:
            table, eq_map, dd_map = run_symbol(symbol, cfg)
            for _, row in table.iterrows():
                rec = row.to_dict()
                rec.update({f"param_{k}": cfg_kwargs[k] for k in keys})
                all_rows.append(rec)
        
    return pd.DataFrame(all_rows)
    