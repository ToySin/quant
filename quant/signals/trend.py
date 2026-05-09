"""Trend-following technical signals.

Each function takes a wide close-price DataFrame (date x ticker) and
returns a wide DataFrame of the same shape — either a continuous
indicator (e.g. the moving average itself) or a 0/1 mask.
"""

from __future__ import annotations

import pandas as pd


def moving_average(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average. NaN until `window` observations are present."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return close.rolling(window=window, min_periods=window).mean()


def ma_cross_signal(close: pd.DataFrame, *, fast: int = 50,
                    slow: int = 200) -> pd.DataFrame:
    """Golden-cross / death-cross long-flat mask.

    Long (1) when the `fast` MA is above the `slow` MA, flat (0) when
    below. The slow MA's lookback dominates: signal is NaN until
    `slow` bars are available, then 0/1 after.

    Convention follows TA literature: fast=50, slow=200 ('golden cross').
    Common variants: 20/50 (short-term), 100/200 (very slow).
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    fast_ma = moving_average(close, fast)
    slow_ma = moving_average(close, slow)
    long = (fast_ma > slow_ma).astype(float)
    # Mask the warmup period so the backtest doesn't trade on NaN
    long = long.where(slow_ma.notna())
    return long
