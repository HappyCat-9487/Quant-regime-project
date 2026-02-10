from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import yfinance as yf

@dataclass(frozen=True)
class YahooConfig:
    cache_dir: Path = Path("data")
    auto_adjust: bool = True  #Automatically adjust prices for splits and dividends
    interval: str = "1d" #Daily data
    
def fetch_yahoo(symbol: str, start:str, end: str, cfg: YahooConfig = YahooConfig()) -> pd.DataFrame:
    """Fetch data from Yahoo Finance, and return a clean OHLCV dataframe indexed by date (UTC naive), with columns:
    Open, High, Low, Close, Volume"""
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.cache_dir / f"{symbol}_{start}_{end}.parquet"
    
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        
        # flatten yfinance multiindex columns (single symbol case)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        
        return df
    
    df = yf.download(symbol, 
                     start=start, 
                     end=end, 
                     interval=cfg.interval, 
                     auto_adjust=cfg.auto_adjust,
                     progress=False)
    
    # flatten yfinance multiindex columns (single symbol case)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    
    if df.empty:
        raise ValueError(f"No data found for {symbol}. Check symbol and date range.")
    
    #Standardize fields
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index)
    df.to_parquet(cache_path)
    return df
    
    
    