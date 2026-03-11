from __future__ import annotations
import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data import fetch_yahoo
from src.features import add_returns, add_mean_reversion_signal, add_momentum_signal, add_vol_regime
from src.backtest import backtest_long_only, backtest_long_only_with_regime
from src.metrics import summary, regime_summary

#--- Figure saving helper ---#
def save_figure(filename: str, assets_dir: Path, reports_dir: Path | None = None, dpi: int = 250) -> Path:
    """Save the current matplotlib figure to assets_dir.
    If reports_dir is provided, also copy the file there.
    Returns the path saved in assets_dir.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    out = assets_dir / filename
    plt.savefig(out, dpi=dpi)
    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, reports_dir / filename)
    return out


#--- Metrics utilities ---#
def equity_from_returns(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()

def drawdown_from_equity(eq: pd.Series) -> pd.Series:
    peak = eq.cummax()
    return (eq / peak) - 1.0

def perf_metrics(strat_ret: pd.Series, periods: int = 252):
    r = strat_ret.copy()
    r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eq = equity_from_returns(r)
    T = len(r)
    if T == 0:
        metrics = dict(FinalEquity=np.nan, AnnRet=np.nan, AnnVol=np.nan, Sharpe=np.nan, MaxDD=np.nan, Calmar=np.nan)
        return metrics, eq, pd.Series(dtype=float)

    final_eq = float(eq.iloc[-1])
    ann_ret = float(final_eq ** (periods / T) - 1.0)
    ann_vol = float(r.std(ddof=0) * np.sqrt(periods)) if r.std(ddof=0) > 0 else 0.0
    sharpe = float((r.mean() / r.std(ddof=0)) * np.sqrt(periods)) if r.std(ddof=0) > 0 else 0.0
    
    dd = drawdown_from_equity(eq)
    maxdd = float(dd.min()) if len(dd) else 0.0
    calmar = float(ann_ret / abs(maxdd)) if maxdd < 0 else np.nan

    metrics = dict(
        FinalEquity=final_eq, 
        AnnRet=ann_ret, 
        AnnVol=ann_vol, 
        Sharpe=sharpe, 
        MaxDD=maxdd, 
        Calmar=calmar)
    return metrics, eq, dd


#---Core run---#

@dataclass(frozen=True)
class Config:
    start: str = "2010-01-01"
    end: str = "2025-01-01"
    fee_bps: float = 1.0
    mom_lookback: int = 60
    mr_window: int = 20
    vol_window: int = 20
    vol_q: float = 0.7
    regime_active: int = 0 # trade when high_vol==0 (low vol)
    
def prepare(symbol: str, cfg: Config) -> pd.DataFrame:
    df = fetch_yahoo(symbol, cfg.start, cfg.end)
    df = add_returns(df)
    df = add_momentum_signal(df, lookback=cfg.mom_lookback)
    df = add_mean_reversion_signal(df, z_window=cfg.mr_window)
    df = add_vol_regime(df, vol_window=cfg.vol_window, q=cfg.vol_q)
    return df.dropna()

def plot_and_save(
    symbol: str,
    cfg: Config,
    assets_dir: Path,
    eq_map: dict,
    dd_map: dict,
    reports_dir: Path | None = None,
):
    #Equity plot
    plt.figure()
    for name, series in eq_map.items():
        series.plot(label=name)
    plt.title(f"Equity: {symbol} (fees: {cfg.fee_bps}bps)")
    plt.legend()
    plt.tight_layout()
    eq_path = save_figure(f"{symbol}_equity.png", assets_dir, reports_dir)
    plt.close()

    #Drawdown plot
    plt.figure()
    for name, series in dd_map.items():
        series.plot(label=name)
    plt.title(f"Drawdown: {symbol}")
    plt.legend()
    plt.tight_layout()
    dd_path = save_figure(f"{symbol}_drawdown.png", assets_dir, reports_dir)
    plt.close()

    return eq_path, dd_path

def run_symbol(symbol: str, cfg: Config):
    df = prepare(symbol, cfg)
    
    #Baseline buy&hold
    bh_metrics, bh_eq, bh_dd = perf_metrics(df["ret_1d"])
    
    #Strategies
    mom_bt = backtest_long_only(df, "signal_mom", fee_bps=cfg.fee_bps)
    mr_bt = backtest_long_only(df, "signal_mr", fee_bps=cfg.fee_bps)
    mrf_bt = backtest_long_only_with_regime(
        df, "signal_mr", "high_vol", 
        regime_active=cfg.regime_active,
        fee_bps=cfg.fee_bps
    )
    
    mom_metrics, mom_eq, mom_dd = perf_metrics(mom_bt["strat_ret"]) 
    mr_metrics, mr_eq, mr_dd = perf_metrics(mr_bt["strat_ret"])
    mrf_metrics, mrf_eq, mrf_dd = perf_metrics(mrf_bt["strat_ret"])
    
    rows = [
        {"Symbol": symbol, "Strategy": "Buy&Hold", **bh_metrics},
        {"Symbol": symbol, "Strategy": "Momentum", **mom_metrics},
        {"Symbol": symbol, "Strategy": "Mean Reversion", **mr_metrics},
        {"Symbol": symbol, "Strategy": "Mean Reversion with Regime", **mrf_metrics}
    ]
    table = pd.DataFrame(rows)
    
    eq_map = {
        "Buy&Hold": bh_eq,
        "Momentum": mom_eq,
        "Mean Reversion": mr_eq,
        "Mean Reversion with Regime": mrf_eq
    }
    dd_map = {
        "Buy&Hold": bh_dd,
        "MOM long-only": mom_dd,
        "MR long-only": mr_dd,
        "MR + low-vol filter": mrf_dd,
    }
    
    return table, eq_map, dd_map


#---W&B logging---#
def wandb_init_and_log(config_dict: dict, metrics_df: pd.DataFrame, image_paths: list[Path]):
    import wandb
    run = wandb.init(project=config_dict.get("wandb_project", "quant-regime-project"), 
                                             config={k:v for k,v in config_dict.items() if k not in ["wandb_project"]})
    
    #log table and images
    wandb.log({"metrics_table": wandb.Table(dataframe=metrics_df)})
    
    for p in image_paths:
        wandb.log({p.name: wandb.Image(str(p))})
    
    run.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, nargs="+", required=True, default=["SPY", "2330.TW"])
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date")
    parser.add_argument("--fee_bps", type=float, default=1.0, help="Fee in basis points")
    parser.add_argument("--mom_lookback", type=int, default=60, help="Momentum lookback period")
    parser.add_argument("--mr_window", type=int, default=20, help="Mean reversion window")
    parser.add_argument("--vol_window", type=int, default=20, help="Volatility window")
    parser.add_argument("--vol_q", type=float, default=0.7, help="Volatility quantile")
    parser.add_argument("--regime_active", type=int, default=0, help="Regime active value")
    
    parser.add_argument("--assets_dir", default="assets", help="Directory for presentation-grade figures")
    parser.add_argument("--reports_dir", default="reports/figures", help="Directory for full report artifacts")
    parser.add_argument("--also_save_reports", action="store_true",
                        help="Also copy figures to --reports_dir in addition to --assets_dir")
    parser.add_argument("--use_wandb", action="store_true", help="Use W&B for logging")
    parser.add_argument("--wandb_project", default="quant-regime-project", help="W&B project name")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        start=args.start,
        end=args.end,
        fee_bps=args.fee_bps,
        mom_lookback=args.mom_lookback,
        mr_window=args.mr_window,
        vol_window=args.vol_window,
        vol_q=args.vol_q,
        regime_active=args.regime_active
    )
    
    assets_dir = Path(args.assets_dir)
    reports_dir = Path(args.reports_dir) if args.also_save_reports else None

    all_tables = []
    images = []

    for sym in args.symbols:
        table, eq_map, dd_map = run_symbol(sym, cfg)
        all_tables.append(table)
        eq_path, dd_path = plot_and_save(sym, cfg, assets_dir, eq_map, dd_map, reports_dir)
        images.extend([eq_path, dd_path])
    
    report = pd.concat(all_tables, ignore_index=True)
    report_path = Path("reports/summary_metrics.csv")
    report.to_csv(report_path, index=False)
    print("Saved:", report_path)
    
    if args.use_wandb:
        config_dict = vars(args)
        wandb_init_and_log(config_dict, report, images)
        
    print("Done!")
    
    
if __name__ == "__main__":
    main()