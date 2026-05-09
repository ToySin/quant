"""Universe construction — which tickers we consider tradable."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from quant.cache import cache_dir

# Curated 30-name liquid US universe — fast for end-to-end smoke tests.
LIQUID_30 = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "JPM", "V", "WMT", "MA", "UNH", "XOM", "JNJ", "PG", "HD", "CVX",
    "MRK", "PEP", "KO", "ABBV", "BAC", "COST", "MCD", "DIS", "ADBE",
    "NFLX", "CRM", "ORCL",
)


def liquid_30() -> list[str]:
    """Hand-picked 30-name large-cap US universe. Fast to backtest."""
    return list(LIQUID_30)


def sp500(refresh: bool = False) -> list[str]:
    """S&P 500 constituents scraped from Wikipedia, parquet-cached.

    Wikipedia is the canonical free source most retail quants use.
    Cache lives under data/cache/universe_sp500.parquet so subsequent
    calls are instant.
    """
    cache_path = cache_dir() / "universe_sp500.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)["ticker"].tolist()

    # pandas' built-in fetcher uses urllib's default User-Agent which
    # Wikipedia 403s. Pull through requests with a real UA, then parse.
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "quant/0.0 (personal research)"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0][["Symbol"]].rename(columns={"Symbol": "ticker"})
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B → BRK-B
    df.to_parquet(cache_path, index=False)
    return df["ticker"].tolist()
