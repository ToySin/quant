"""OHLCV download + caching via yfinance.

The cache uses one parquet per ticker so adding tickers later doesn't
invalidate prior downloads. Re-running with the same date range hits
the cache; widening the range triggers a refresh for that ticker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant.cache import cache_dir


@dataclass(frozen=True)
class PriceFrame:
    """Wide-format adjusted-close panel + per-ticker volume.

    Indexed by date, columns are tickers. `close` uses adjusted close
    so split/dividend events are absorbed — that's what factor research
    needs. `volume` is raw volume, used only for liquidity filters.
    """
    close: pd.DataFrame
    volume: pd.DataFrame

    def returns(self) -> pd.DataFrame:
        """Daily simple returns. First row is NaN."""
        return self.close.pct_change()

    def log_returns(self) -> pd.DataFrame:
        """Daily log returns. First row is NaN."""
        import numpy as np
        return pd.DataFrame(
            data=np.log(self.close).diff().to_numpy(),
            index=self.close.index,
            columns=self.close.columns,
        )


def load(tickers: list[str], start: str, end: str | None = None,
         refresh: bool = False) -> PriceFrame:
    """Download (or load from cache) OHLCV for `tickers`.

    `start` / `end` are ISO date strings. Cached parquets store the
    full historical pull and are sliced on read; pass `refresh=True`
    to force a re-download.
    """
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = _load_one(ticker, refresh=refresh)
        if df.empty:
            continue
        sliced = df.loc[start:end] if end else df.loc[start:]
        if sliced.empty:
            continue
        closes[ticker] = sliced["close"]
        volumes[ticker] = sliced["volume"]
    if not closes:
        raise RuntimeError(
            f"no data loaded for any of {len(tickers)} tickers in [{start}, {end}]"
        )
    close_df = pd.concat(closes, axis=1).sort_index()
    volume_df = pd.concat(volumes, axis=1).sort_index()
    return PriceFrame(close=close_df, volume=volume_df)


def _load_one(ticker: str, refresh: bool) -> pd.DataFrame:
    cache_path = cache_dir() / f"ohlcv_{ticker}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    df = _download(ticker)
    if not df.empty:
        df.to_parquet(cache_path)
    return df


def _download(ticker: str) -> pd.DataFrame:
    """Pull a single ticker's full history from yfinance.

    Returns columns: open / high / low / close / volume. Close is the
    *adjusted* close so factor signals don't get distorted by splits.
    Empty DataFrame on failure (delisted, network hiccup, etc.) so
    the orchestrator can keep going.
    """
    import yfinance as yf

    raw = yf.download(
        ticker,
        period="max",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = raw.columns.str.lower()
    raw.index.name = "date"
    return raw[["open", "high", "low", "close", "volume"]]
