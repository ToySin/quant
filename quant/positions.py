"""Position sizing for time-series signals.

Bridges the gap between `quant.signals` (per-ticker long/flat masks)
and `quant.backtest.run` (which wants target weights summing to <=1).
"""

from __future__ import annotations

import pandas as pd


def equal_weight_when_long(mask: pd.DataFrame, *,
                           cash_when_empty: bool = True) -> pd.DataFrame:
    """Distribute capital equally across all currently-long tickers.

    On any given date, if k tickers are long (mask == 1), each gets
    weight 1/k. If no ticker is long and `cash_when_empty=True`,
    the portfolio is in cash (zero weights). NaN entries are treated
    as flat.
    """
    import numpy as np

    binary = mask.fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)
    n_long = binary.sum(axis=1).replace(0.0, np.nan)
    weights = binary.div(n_long, axis=0)

    if cash_when_empty:
        weights = weights.fillna(0.0)
    else:
        weights = weights.ffill().fillna(0.0)
    return weights.astype(float)


def fixed_fraction(mask: pd.DataFrame, *, fraction: float) -> pd.DataFrame:
    """Each long ticker gets a fixed fraction of capital, capped so
    total exposure never exceeds 1.0.

    Useful for "always 10% per name" style position sizing where you
    don't want concentration regardless of how few names are long.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    binary = mask.fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)
    raw = binary * fraction
    total = raw.sum(axis=1)
    over = total > 1.0
    if over.any():
        scale = pd.Series(1.0, index=raw.index)
        scale.loc[over] = 1.0 / total.loc[over]
        raw = raw.mul(scale, axis=0)
    return raw
