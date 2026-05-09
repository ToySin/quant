"""Shared pytest fixtures — synthetic OHLCV so tests don't hit network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_close():
    """500 trading days × 10 tickers, deterministic geometric brownian
    walks with different drifts so factor signals have something to
    pick up on."""
    rng = np.random.default_rng(42)
    n_days = 500
    tickers = [f"T{i:02d}" for i in range(10)]
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    drifts = np.linspace(-0.0005, 0.0008, len(tickers))
    sigma = 0.015
    log_rets = rng.normal(size=(n_days, len(tickers))) * sigma + drifts
    close = pd.DataFrame(
        100.0 * np.exp(log_rets.cumsum(axis=0)),
        index=dates,
        columns=tickers,
    )
    return close


@pytest.fixture
def synthetic_volume(synthetic_close):
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.integers(low=100_000, high=10_000_000,
                     size=synthetic_close.shape).astype(float),
        index=synthetic_close.index,
        columns=synthetic_close.columns,
    )


@pytest.fixture
def synthetic_returns(synthetic_close):
    return synthetic_close.pct_change().fillna(0.0)
