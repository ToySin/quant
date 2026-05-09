"""Quality factors built from fundamentals snapshots.

"Quality" tries to capture business durability — high return on
equity, low leverage, stable margins. Different vendors weight these
differently; we go with a clean three-way blend that matches the
MSCI Quality Index spirit.

Same caveats as `value.py`: snapshot-only, not point-in-time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.factors.value import _zscore
from quant.fundamentals import Fundamentals


def return_on_equity(funds: Fundamentals) -> pd.Series:
    """ROE as-reported. Higher = more efficient use of book equity."""
    return funds.df["returnOnEquity"].astype(float)


def low_leverage(funds: Fundamentals) -> pd.Series:
    """Negated debt-to-equity so higher = less levered.

    Capped at zero on the bottom: companies with negative book equity
    get NaN since the ratio is meaningless.
    """
    de = funds.df["debtToEquity"].astype(float)
    de = de.where(de >= 0)
    return -de


def profit_margin(funds: Fundamentals) -> pd.Series:
    """Net profit margin. Higher = more profit per dollar of revenue."""
    return funds.df["profitMargins"].astype(float)


def composite(funds: Fundamentals) -> pd.Series:
    """Equal-weight z-score blend of ROE, low-leverage, and margin."""
    roe = _zscore(return_on_equity(funds))
    lev = _zscore(low_leverage(funds))
    pm = _zscore(profit_margin(funds))
    return ((roe + lev + pm) / 3).rename("quality")
