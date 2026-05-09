"""Value factors built from fundamentals snapshots.

A "value" factor scores cheap-on-fundamentals stocks higher. We use
the inverse of common multiples so higher score = cheaper:
  - 1 / P/B  (book yield)
  - 1 / P/E  (earnings yield)

Fundamentals are cross-sectional only (one snapshot per ticker), so
the resulting score series is constant in time. The backtest engine
still treats it as a date x ticker DataFrame for uniformity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.fundamentals import Fundamentals


def book_yield(funds: Fundamentals) -> pd.Series:
    """1 / P/B. Higher = cheaper relative to book value.

    Negative or zero P/B (which can happen with negative book equity)
    gets NaN since "cheaper than free" doesn't mean what value
    investors want.
    """
    pb = funds.df["priceToBook"].astype(float)
    pb = pb.where(pb > 0)
    return 1.0 / pb


def earnings_yield(funds: Fundamentals) -> pd.Series:
    """1 / trailing P/E. Higher = more earnings per dollar paid.

    Companies with negative earnings get NaN — they're not value plays
    in the traditional sense (could be growth stories, but a value
    *factor* should not lump them in)."""
    pe = funds.df["trailingPE"].astype(float)
    pe = pe.where(pe > 0)
    return 1.0 / pe


def composite(funds: Fundamentals) -> pd.Series:
    """Equal-weight blend of book-yield and earnings-yield z-scores.

    Z-score normalize each component first so neither dominates by
    virtue of having a wider spread. Returns one score per ticker.
    """
    by = _zscore(book_yield(funds))
    ey = _zscore(earnings_yield(funds))
    return ((by + ey) / 2).rename("value")


def as_panel(score: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Broadcast a per-ticker fundamentals score across `dates` so the
    cross-sectional portfolio builder can consume it.

    Yields a NaN-then-constant frame: NaN until the snapshot_date
    convention (we have no point-in-time, so we just publish the
    score from the start of the panel).
    """
    return pd.DataFrame(
        np.broadcast_to(score.to_numpy(), (len(dates), len(score))),
        index=dates,
        columns=score.index,
    ).astype(float)


def _zscore(s: pd.Series) -> pd.Series:
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index)
    mu = valid.mean()
    sigma = valid.std(ddof=1)
    if sigma == 0:
        return pd.Series(0.0, index=s.index).where(s.notna())
    return ((s - mu) / sigma).astype(float)
