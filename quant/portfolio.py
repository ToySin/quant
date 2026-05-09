"""Portfolio construction — turn a factor score into target weights.

The signal flow is:
    factor scores (date x ticker)
        -> rebalance dates (subset of the index)
        -> at each rebalance: rank, pick top decile, equal weight
        -> forward-fill weights between rebalances
"""

from __future__ import annotations

import pandas as pd


def month_end_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month present in `index`."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    grouped = pd.Series(index, index=index).groupby([index.year, index.month]).last()
    return pd.DatetimeIndex(grouped.values).sort_values()


def top_decile_long_only(scores: pd.DataFrame, *, top_pct: float = 0.1,
                         rebalance_dates: pd.DatetimeIndex | None = None,
                         min_names: int = 3) -> pd.DataFrame:
    """Long-only top-decile equal-weight portfolio.

    At each `rebalance_dates` row, rank tickers by their score. Take
    the top `top_pct` fraction (rounded up, but at least `min_names`
    if available) and assign equal weight summing to 1. Between
    rebalances, the portfolio drifts on no instructions — i.e.,
    weights are forward-filled until the next rebalance.

    Tickers with NaN scores at a rebalance date are excluded from
    that round.
    """
    if not 0 < top_pct <= 1:
        raise ValueError("top_pct must be in (0, 1]")

    if rebalance_dates is None:
        rebalance_dates = month_end_rebalance_dates(scores.index)
    rebalance_dates = rebalance_dates.intersection(scores.index)

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    last_weights: pd.Series | None = None

    for date in scores.index:
        if date in rebalance_dates:
            row = scores.loc[date].dropna()
            if row.empty:
                last_weights = None
                continue
            n_pick = max(min_names, int(-(-len(row) * top_pct // 1)))  # ceil
            n_pick = min(n_pick, len(row))
            picked = row.nlargest(n_pick).index
            w = pd.Series(0.0, index=scores.columns)
            w.loc[picked] = 1.0 / n_pick
            last_weights = w
        if last_weights is not None:
            weights.loc[date] = last_weights
    return weights


def long_short_decile(scores: pd.DataFrame, *, top_pct: float = 0.1,
                      rebalance_dates: pd.DatetimeIndex | None = None,
                      min_names: int = 3) -> pd.DataFrame:
    """Dollar-neutral long-short: +1 weight long top decile, -1 short
    bottom decile, scaled so gross exposure is 2 (1 long, 1 short)."""
    long_w = top_decile_long_only(
        scores, top_pct=top_pct, rebalance_dates=rebalance_dates, min_names=min_names,
    )
    short_w = top_decile_long_only(
        -scores, top_pct=top_pct, rebalance_dates=rebalance_dates, min_names=min_names,
    )
    return long_w - short_w
