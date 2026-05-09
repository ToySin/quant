"""Realized volatility — building block for the low-volatility factor.

Convention: `realized_volatility` returns the raw stdev of returns
(higher = more volatile). `inverse_volatility` flips the sign so it
can be used directly as a *factor score* where higher = more
attractive (the low-vol tilt).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(close: pd.DataFrame, *, window: int = 63,
                        annualize: bool = True) -> pd.DataFrame:
    """Rolling stdev of daily simple returns.

    Default window of 63 trading days approximates one quarter.
    Annualized by sqrt(252) when `annualize=True`.
    """
    if window <= 1:
        raise ValueError("window must be > 1")
    rets = close.pct_change()
    vol = rets.rolling(window=window, min_periods=max(2, window // 2)).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def inverse_volatility(close: pd.DataFrame, *, window: int = 63) -> pd.DataFrame:
    """Negated realized vol, suitable as a low-vol factor score.

    Higher score = lower realized vol = more attractive under the
    low-volatility anomaly.
    """
    return -realized_volatility(close, window=window, annualize=False)
