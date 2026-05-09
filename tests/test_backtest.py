"""Backtest accounting correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest import run


def test_zero_weights_zero_pnl():
    dates = pd.bdate_range("2024-01-01", periods=20)
    cols = ["A", "B"]
    rets = pd.DataFrame(0.01, index=dates, columns=cols)
    weights = pd.DataFrame(0.0, index=dates, columns=cols)
    result = run(rets, weights)
    assert (result.returns == 0.0).all()
    assert (result.gross_returns == 0.0).all()


def test_full_weight_one_asset_matches_return():
    dates = pd.bdate_range("2024-01-01", periods=10)
    rets = pd.DataFrame(
        {"A": [0.01] * 10, "B": [0.05] * 10},
        index=dates,
    )
    # Full weight on A from day 0 → daily return should equal A's return
    # from day 1 onwards (because of t-1 shift). No turnover after day 0.
    weights = pd.DataFrame(
        {"A": [1.0] * 10, "B": [0.0] * 10},
        index=dates,
    )
    result = run(rets, weights, cost_bps=0.0)
    # Day 0: weight applied is 0 (shift of NaN→0). Day 1+: weight is 1.0 in A
    np.testing.assert_allclose(result.gross_returns.iloc[0], 0.0)
    np.testing.assert_allclose(result.gross_returns.iloc[1:], 0.01)


def test_costs_charge_on_turnover():
    """Switching from A to B on day 5 incurs L1 turnover of 2.0 (sell 1, buy 1)."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    rets = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    weights = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    weights.iloc[:5] = [1.0, 0.0]
    weights.iloc[5:] = [0.0, 1.0]
    result_no_cost = run(rets, weights, cost_bps=0.0)
    result_with_cost = run(rets, weights, cost_bps=10.0)  # 10 bps
    cost_diff = result_no_cost.returns - result_with_cost.returns
    # Day of switch: cost = 2.0 turnover * 10/10000 = 0.002
    np.testing.assert_allclose(cost_diff.iloc[5], 0.002)
    # Other days no cost (zero turnover after first allocation, ignoring day 0)
    assert cost_diff.iloc[6:].sum() == pytest.approx(0.0)


def test_negative_cost_rejected():
    dates = pd.bdate_range("2024-01-01", periods=5)
    rets = pd.DataFrame(0.0, index=dates, columns=["A"])
    weights = pd.DataFrame(0.0, index=dates, columns=["A"])
    with pytest.raises(ValueError):
        run(rets, weights, cost_bps=-1.0)
