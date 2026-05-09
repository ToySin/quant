"""Performance metrics correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.report import TRADING_DAYS, metrics


def test_constant_positive_return_yields_high_sharpe():
    dates = pd.bdate_range("2024-01-01", periods=252)
    r = pd.Series(0.001, index=dates)  # 10 bps/day
    m = metrics(r)
    assert m.cagr > 0.25     # ~28%/yr
    assert m.sharpe > 50     # noiseless return → essentially infinite Sharpe
    assert m.max_drawdown == 0.0
    assert m.hit_rate == 1.0


def test_zero_returns_yield_flat_metrics():
    dates = pd.bdate_range("2024-01-01", periods=252)
    r = pd.Series(0.0, index=dates)
    m = metrics(r)
    assert m.cagr == 0.0
    assert m.annualized_volatility == 0.0
    assert m.sharpe == 0.0
    assert m.max_drawdown == 0.0


def test_drawdown_is_negative_and_bounded():
    dates = pd.bdate_range("2024-01-01", periods=20)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(loc=-0.001, scale=0.02, size=20), index=dates)
    m = metrics(r)
    assert m.max_drawdown <= 0.0
    assert m.max_drawdown > -1.0  # can't lose more than 100%


def test_calmar_uses_cagr_over_mdd():
    dates = pd.bdate_range("2024-01-01", periods=252)
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(loc=0.0008, scale=0.012, size=252), index=dates)
    m = metrics(r)
    if m.max_drawdown < 0:
        np.testing.assert_allclose(m.calmar, m.cagr / abs(m.max_drawdown))


def test_empty_returns_rejected():
    with pytest.raises(ValueError):
        metrics(pd.Series([], dtype=float))


def test_metrics_table_renders():
    dates = pd.bdate_range("2024-01-01", periods=60)
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, size=60), index=dates)
    table = metrics(r).as_table()
    assert "Sharpe" in table
    assert "Max drawdown" in table
