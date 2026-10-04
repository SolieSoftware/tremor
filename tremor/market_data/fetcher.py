from datetime import datetime
from io import StringIO

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

# Map causal network node names to data sources and computation methods
NODE_CONFIG = {
    "d_fed_funds": {"source": "FRED", "ticker": "DFF", "method": "weekly_change"},
    "d_treasury_10y": {"source": "FRED", "ticker": "DGS10", "method": "weekly_change"},
    "d_credit_spread": {"source": "FRED", "ticker": "BAMLH0A0HYM2", "method": "weekly_change"},
    "d_vix": {"source": "yahoo", "ticker": "^VIX", "method": "weekly_change"},
    "sp500_ret": {"source": "yahoo", "ticker": "^GSPC", "method": "weekly_log_return"},
}


def fetch_daily_node_data(node_name: str, start_date: datetime, end_date: datetime) -> pd.Series:
    """Fetch daily market data for a causal network node. No resampling.

    Returns raw daily closing prices/levels for use in event study windows.
    """
    config = NODE_CONFIG.get(node_name)
    if not config:
        raise ValueError(f"Unknown node: {node_name}")

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if config["source"] == "yahoo":
        return _fetch_yahoo(config["ticker"], start_str, end_str)
    elif config["source"] == "FRED":
        return _fetch_fred(config["ticker"], start_str, end_str)
    else:
        raise ValueError(f"Unknown data source: {config['source']}")


def fetch_node_data(node_name: str, start_date: datetime, end_date: datetime) -> pd.Series:
    """Fetch market data for a causal network node over a date range.

    Returns a pandas Series of the computed variable (weekly changes or returns).
    """
    config = NODE_CONFIG.get(node_name)
    if not config:
        raise ValueError(f"Unknown node: {node_name}")

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if config["source"] == "yahoo":
        raw = _fetch_yahoo(config["ticker"], start_str, end_str)
    elif config["source"] == "FRED":
        raw = _fetch_fred(config["ticker"], start_str, end_str)
    else:
        raise ValueError(f"Unknown data source: {config['source']}")

    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    weekly = raw.resample("W").last().dropna()

    if config["method"] == "weekly_change":
        return weekly.diff().dropna()
    elif config["method"] == "weekly_log_return":
        return np.log(weekly / weekly.shift(1)).dropna()
    else:
        return weekly


def _fetch_yahoo(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch closing prices from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    if "Close" in data.columns:
        return data["Close"].squeeze()
    return pd.Series(dtype=float)


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a daily FRED series via the public CSV endpoint (no API key).

    yfinance does not serve FRED series, so this goes to FRED directly.
    Missing observations (holidays) are published as blanks or "." and dropped.
    """
    try:
        response = httpx.get(
            FRED_CSV_URL,
            params={"id": series_id, "cosd": start, "coed": end},
            timeout=30,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return pd.Series(dtype=float)

    df = pd.read_csv(StringIO(response.text), na_values=["."], index_col=0, parse_dates=True)
    if df.empty:
        return pd.Series(dtype=float)
    return df.iloc[:, 0].astype(float).dropna()
