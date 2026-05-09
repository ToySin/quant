"""Factor signals — each function returns a wide DataFrame (date x ticker)
of cross-sectional scores. Higher = stronger factor exposure.

Two flavors live here:
  - Price-based factors (momentum, volatility): one score per (date, ticker)
  - Fundamentals-based factors (value, quality): one score per ticker,
    broadcast across dates via `value.as_panel(...)`
"""

from quant.factors import quality, value
from quant.factors.momentum import momentum_12_1, momentum_lookback
from quant.factors.volatility import inverse_volatility, realized_volatility

__all__ = [
    "momentum_12_1",
    "momentum_lookback",
    "realized_volatility",
    "inverse_volatility",
    "quality",
    "value",
]
