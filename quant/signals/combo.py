"""Combination signals — overlays of trend filter + entry timing.

The motivation: each individual signal has known weaknesses. MA cross
catches big regimes but lags; MACD enters faster but produces false
signals in chop. Combining a slow filter with a fast trigger filters
out signals that fight the broader trend.

All combos return long-flat masks (0 / 1, NaN during warmup) with the
same shape as the input close panel.
"""

from __future__ import annotations

import pandas as pd

from quant.signals.oscillator import macd
from quant.signals.trend import moving_average


def trend_filtered_macd(close: pd.DataFrame, *,
                        fast_ma: int = 50, slow_ma: int = 200,
                        macd_fast: int = 12, macd_slow: int = 26,
                        macd_signal_window: int = 9) -> pd.DataFrame:
    """Pattern A: regime filter via MA, entry timing via MACD.

    Long only when MA(fast_ma) > MA(slow_ma) AND MACD line > signal line.
    Otherwise flat. Removes MACD's false signals during downtrends.
    """
    fast = moving_average(close, fast_ma)
    slow = moving_average(close, slow_ma)
    regime_up = fast > slow

    macd_result = macd(close, fast=macd_fast, slow=macd_slow,
                       signal=macd_signal_window)
    macd_up = macd_result.macd > macd_result.signal

    signal = (regime_up & macd_up).astype(float)
    # Mask warmup so the backtest doesn't trade on undefined values
    valid = slow.notna() & macd_result.signal.notna()
    return signal.where(valid)


def ma_macd_confirm(close: pd.DataFrame, *,
                    fast_ma: int = 50, slow_ma: int = 200,
                    macd_fast: int = 12, macd_slow: int = 26,
                    macd_signal_window: int = 9) -> pd.DataFrame:
    """Pattern B: both MA cross and MACD must agree.

    Identical implementation to trend_filtered_macd, kept as a separate
    function for clarity and to leave room for asymmetric exit rules
    later (e.g. exit only requires one to flip, but entry needs both).
    """
    return trend_filtered_macd(
        close,
        fast_ma=fast_ma, slow_ma=slow_ma,
        macd_fast=macd_fast, macd_slow=macd_slow,
        macd_signal_window=macd_signal_window,
    )


def price_above_ma_with_macd(close: pd.DataFrame, *,
                             ma_window: int = 200,
                             macd_fast: int = 12, macd_slow: int = 26,
                             macd_signal_window: int = 9) -> pd.DataFrame:
    """Pattern C: price-above-MA200 regime filter + MACD entry.

    Modern variant: instead of waiting for MA50/MA200 to cross, just
    require price itself to be above its long-term MA. Reacts faster
    at trend reversals than MA cross.
    """
    long_ma = moving_average(close, ma_window)
    regime_up = close > long_ma

    macd_result = macd(close, fast=macd_fast, slow=macd_slow,
                       signal=macd_signal_window)
    macd_up = macd_result.macd > macd_result.signal

    signal = (regime_up & macd_up).astype(float)
    valid = long_ma.notna() & macd_result.signal.notna()
    return signal.where(valid)
