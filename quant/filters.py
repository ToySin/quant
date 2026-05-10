"""Sanity filters that catch data-quality issues before they touch capital.

The most common offender for retail data feeds (yfinance, etc.) is
*corporate actions* — spinoffs, mergers, ticker reassignments — that
yfinance silently splices into one continuous series. The result is
a momentum score of +2000% when the ticker today has nothing to do
with the entity that bore the same symbol a year ago.

We don't have access to corporate-action timestamps, so we rely on
a behavioral proxy: any score above an aggressive threshold is far
more likely a data error than a real return. Real momentum top
performers in liquid US large-cap rarely exceed +200-300% on a
single 12-month window; +1000%+ is essentially always an artifact.
"""

from __future__ import annotations

import pandas as pd


def drop_extreme_momentum(scores: pd.Series | pd.DataFrame,
                          *, max_score: float = 4.0) -> pd.Series | pd.DataFrame:
    """Replace momentum scores above `max_score` with NaN.

    Default threshold of 4.0 = +400% return on the 12-1 window. This
    is loose enough to keep semiconductor rally-era leaders (e.g.
    NVDA-class names occasionally clear +200%) but tight enough to
    catch obvious corporate-action splices.
    """
    if max_score <= 0:
        raise ValueError("max_score must be positive")
    return scores.where(scores < max_score)
