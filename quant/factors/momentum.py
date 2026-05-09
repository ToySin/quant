"""Momentum factor.

Classic Jegadeesh-Titman 12-1: past 12 months return excluding the
most recent month, to avoid the well-known short-term reversal effect
(when a stock spikes today it tends to mean-revert tomorrow, so the
last month is dropped from the formation window).
"""

from __future__ import annotations

import pandas as pd


def momentum_lookback(close: pd.DataFrame, *, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Total return over `lookback` trading days, excluding the most
    recent `skip` trading days.

    Concretely: ratio of `close[t-skip]` to `close[t-skip-lookback]`
    minus 1, evaluated at every t.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if skip < 0:
        raise ValueError("skip must be non-negative")
    end = close.shift(skip)
    start = close.shift(skip + lookback)
    return end / start - 1.0


def momentum_12_1(close: pd.DataFrame) -> pd.DataFrame:
    """12-month-minus-1 momentum on daily-bar inputs.

    Approximates 12 months as 252 trading days and 1 month as 21
    trading days, so the formation window is the 231-day return
    ending 21 days ago.
    """
    return momentum_lookback(close, lookback=252 - 21, skip=21)
