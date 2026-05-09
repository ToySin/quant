"""Technical signals — per-ticker time-series long/flat masks.

Distinguished from `quant.factors`: factors compare *across* tickers
at a single point in time (cross-sectional), while signals look at
*one ticker over time* (time-series). Both produce date x ticker
DataFrames so the backtest framework treats them uniformly.
"""

from quant.signals.oscillator import (
    MACDResult,
    macd,
    macd_signal,
    rsi,
    rsi_signal,
)
from quant.signals.trend import ma_cross_signal, moving_average

__all__ = [
    "MACDResult",
    "ma_cross_signal",
    "macd",
    "macd_signal",
    "moving_average",
    "rsi",
    "rsi_signal",
]
