"""Combine multiple factor signals into a single composite score.

Each input is a wide DataFrame (date x ticker) of cross-sectional
scores. Z-score normalize across the cross-section per date so no
single factor dominates by virtue of spread, then weighted-average.
"""

from __future__ import annotations

import pandas as pd


def cross_sectional_zscore(scores: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize each row (date) across columns (tickers)."""
    import numpy as np
    mu = scores.mean(axis=1)
    sigma = scores.std(axis=1, ddof=1).replace(0.0, np.nan)
    return scores.sub(mu, axis=0).div(sigma, axis=0).astype(float)


def blend(scores: dict[str, pd.DataFrame], *,
          weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Z-score then weighted-average a dict of factor score panels.

    `weights` defaults to equal-weight if omitted. Missing factors at
    a given (date, ticker) are skipped from the average — a stock
    with no fundamentals data is scored on whatever factors *do*
    have data for it (typically the price-based ones).
    """
    if not scores:
        raise ValueError("scores cannot be empty")

    if weights is None:
        weights = {name: 1.0 for name in scores}
    else:
        missing = set(scores) - set(weights)
        if missing:
            raise ValueError(f"weights missing for: {sorted(missing)}")

    normalized = {name: cross_sectional_zscore(df) for name, df in scores.items()}

    weighted_sum: pd.DataFrame | None = None
    weight_sum: pd.DataFrame | None = None
    for name, z in normalized.items():
        w = weights[name]
        contribution = z * w
        coverage = z.notna().astype(float) * w
        if weighted_sum is None:
            weighted_sum = contribution.fillna(0.0)
            weight_sum = coverage
        else:
            ws_aligned, contrib_aligned = weighted_sum.align(contribution.fillna(0.0),
                                                              fill_value=0.0)
            weighted_sum = ws_aligned + contrib_aligned
            ws_w_aligned, cov_aligned = weight_sum.align(coverage, fill_value=0.0)
            weight_sum = ws_w_aligned + cov_aligned

    import numpy as np
    assert weighted_sum is not None and weight_sum is not None
    return (weighted_sum / weight_sum.replace(0.0, np.nan)).astype(float)
