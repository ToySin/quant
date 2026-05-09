"""Oscillator-style technical signals — RSI and MACD.

These compare a price's recent behavior to its own history (rather
than to a moving average), and are typically used as mean-reversion
or momentum-confirmation signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rsi(close: pd.DataFrame, *, window: int = 14) -> pd.DataFrame:
    """Wilder's RSI.

    The original Wilder formulation uses an exponential moving average
    of gains and losses with smoothing factor 1/window (equivalent to
    pandas' `ewm(alpha=1/window, adjust=False)`). This is what most
    trading platforms display as "RSI(14)".

    Returns values in [0, 100]. NaN during warmup.
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    alpha = 1.0 / window
    avg_gain = gains.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When losses are zero (flat or always-up window), RSI is 100 by convention
    out = out.where(avg_loss != 0.0, other=100.0)
    return out


def rsi_signal(close: pd.DataFrame, *, window: int = 14,
               lower: float = 30.0, upper: float = 70.0) -> pd.DataFrame:
    """Mean-reversion long-flat mask off RSI.

    Goes long (1) when RSI crosses below `lower` (oversold), exits
    (0) when RSI crosses above `upper` (overbought). Holds the
    position between thresholds — i.e. once long, stay long until
    the upper threshold triggers an exit.

    NaN during warmup so the backtest doesn't trade noise.
    """
    if not 0 < lower < upper < 100:
        raise ValueError("require 0 < lower < upper < 100")

    indicator = rsi(close, window=window)
    state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    long = pd.Series(False, index=close.columns)

    for date, row in indicator.iterrows():
        oversold = row < lower
        overbought = row > upper
        long = long & ~overbought | oversold
        long = long.fillna(False).infer_objects(copy=False)
        state.loc[date] = long.astype(float).where(row.notna())
    return state


@dataclass(frozen=True)
class MACDResult:
    macd: pd.DataFrame          # fast EMA - slow EMA
    signal: pd.DataFrame        # EMA of macd
    histogram: pd.DataFrame     # macd - signal


def macd(close: pd.DataFrame, *, fast: int = 12, slow: int = 26,
         signal: int = 9) -> MACDResult:
    """Classic MACD: 12/26/9.

    Returns the line, the signal-line EMA, and the histogram (the
    difference). All three are continuous, NaN-padded during warmup.
    """
    if fast >= slow:
        raise ValueError("fast must be < slow")
    if signal < 1:
        raise ValueError("signal must be >= 1")

    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = fast_ema - slow_ema
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return MACDResult(macd=line, signal=sig, histogram=hist)


def macd_signal(close: pd.DataFrame, *, fast: int = 12, slow: int = 26,
                signal: int = 9) -> pd.DataFrame:
    """Long-flat mask: long when MACD line is above its signal line.

    This is the classic "MACD bullish crossover" → long, "bearish
    crossover" → flat rule. NaN during warmup.
    """
    result = macd(close, fast=fast, slow=slow, signal=signal)
    long = (result.macd > result.signal).astype(float)
    long = long.where(result.signal.notna())
    return long
