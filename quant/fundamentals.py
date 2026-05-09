"""Fundamental data via yfinance — P/E, P/B, ROE, debt/equity, market cap.

⚠️ Major caveat: yfinance.Ticker(...).info returns *current* fundamentals,
not point-in-time. Using these in a 16-year backtest is academically
unsound — you're effectively saying "if I had known today's P/B in 2010,
what would I have bought?" That's lookahead.

For real research you need a point-in-time database (Sharadar, Norgate,
Compustat). This module is a *prototype* for getting the pipeline
shape right; treat backtest results as illustrative, not actionable.

The cache is keyed on a single snapshot date, so all tickers share the
same vintage of fundamentals.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.cache import cache_dir


# Subset of yfinance.info keys we actually use. Many keys exist; we keep
# this list short on purpose so missing keys are loud, not silent.
RELEVANT_KEYS: tuple[str, ...] = (
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "returnOnEquity",
    "debtToEquity",
    "profitMargins",
    "operatingMargins",
    "marketCap",
    "enterpriseToEbitda",
    "currentRatio",
    "quickRatio",
)


@dataclass(frozen=True)
class Fundamentals:
    """Cross-sectional fundamentals snapshot. Index = ticker, columns = keys."""
    df: pd.DataFrame
    snapshot_date: pd.Timestamp


def load(tickers: list[str], *, refresh: bool = False) -> Fundamentals:
    """Pull fundamentals snapshot for `tickers`. Cached by snapshot date."""
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    cache_path = cache_dir() / f"fundamentals_{today.strftime('%Y-%m-%d')}.parquet"

    if cache_path.exists() and not refresh:
        df = pd.read_parquet(cache_path)
        missing = [t for t in tickers if t not in df.index]
        if not missing:
            return Fundamentals(df=df.loc[tickers], snapshot_date=today)
        # Cache exists but doesn't cover all requested tickers — extend it
        extra = _fetch_many(missing)
        df = pd.concat([df, extra]).drop_duplicates()
        df.to_parquet(cache_path)
        return Fundamentals(df=df.loc[tickers], snapshot_date=today)

    df = _fetch_many(tickers)
    df.to_parquet(cache_path)
    return Fundamentals(df=df, snapshot_date=today)


def _fetch_many(tickers: list[str]) -> pd.DataFrame:
    """Pull `info` for each ticker; tolerate failures so one bad symbol
    doesn't kill the whole batch."""
    import yfinance as yf

    rows: dict[str, dict] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:  # noqa: BLE001 — network/parsing flakiness
            rows[ticker] = {k: None for k in RELEVANT_KEYS}
            continue
        rows[ticker] = {k: info.get(k) for k in RELEVANT_KEYS}

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    # Force numeric dtypes; yfinance occasionally returns strings or None
    for col in RELEVANT_KEYS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
