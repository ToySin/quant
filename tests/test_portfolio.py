"""Portfolio construction correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.portfolio import (
    long_short_decile,
    month_end_rebalance_dates,
    top_decile_long_only,
)


def test_month_end_rebalance_dates_picks_last_business_day():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    rebalances = month_end_rebalance_dates(idx)
    assert len(rebalances) == 3
    assert rebalances[0] == pd.Timestamp("2024-01-31")
    assert rebalances[1] == pd.Timestamp("2024-02-29")
    assert rebalances[2] == pd.Timestamp("2024-03-29")  # last business day


def test_top_decile_picks_highest_scores(synthetic_close):
    scores = pd.DataFrame(
        {f"T{i:02d}": [float(i)] * len(synthetic_close)
         for i in range(10)},
        index=synthetic_close.index,
    )
    weights = top_decile_long_only(scores, top_pct=0.3)
    last = weights.iloc[-1]
    held = last[last > 0].index.tolist()
    # Top 30% of 10 names = 3 names, must be highest-scored ones
    assert held == ["T07", "T08", "T09"]
    assert np.isclose(last.sum(), 1.0)
    assert np.isclose(last.loc["T09"], 1 / 3)


def test_top_decile_handles_nan_scores():
    dates = pd.bdate_range("2024-01-01", periods=60)
    cols = list("ABCDE")
    scores = pd.DataFrame(
        np.tile([1.0, 2.0, np.nan, 3.0, 4.0], (60, 1)),
        index=dates, columns=cols,
    )
    weights = top_decile_long_only(scores, top_pct=0.6, min_names=2)
    last = weights.iloc[-1]
    # NaN-scored "C" must never get a weight
    assert last.loc["C"] == 0.0
    held = last[last > 0].index.tolist()
    # Top 60% of 4 valid names = ceil(2.4) = 3 → E, D, B
    assert set(held) == {"B", "D", "E"}


def test_top_decile_holds_through_drift():
    """Between rebalances, weights should be forward-filled (not zeroed)."""
    dates = pd.bdate_range("2024-01-01", periods=80)
    cols = ["A", "B"]
    scores = pd.DataFrame(
        np.tile([0.0, 1.0], (80, 1)),
        index=dates, columns=cols,
    )
    weights = top_decile_long_only(scores, top_pct=0.5, min_names=1)
    rebal = month_end_rebalance_dates(dates)
    # Pick a non-rebalance date
    mid_date = rebal[0] + pd.Timedelta(days=2)
    if mid_date in weights.index:
        assert weights.loc[mid_date].sum() == pytest.approx(1.0)


def test_long_short_is_dollar_neutral():
    dates = pd.bdate_range("2024-01-01", periods=60)
    cols = list("ABCDEFGHIJ")
    scores = pd.DataFrame(
        np.tile(np.arange(10, dtype=float), (60, 1)),
        index=dates, columns=cols,
    )
    weights = long_short_decile(scores, top_pct=0.3)
    last = weights.iloc[-1]
    assert np.isclose(last.sum(), 0.0)        # dollar neutral
    assert np.isclose(last.abs().sum(), 2.0)  # gross 2x
