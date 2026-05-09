"""Technical signal correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.signals import (
    ma_cross_signal,
    macd,
    macd_signal,
    moving_average,
    rsi,
    rsi_signal,
)


# ---------- moving_average / ma_cross_signal ----------

def test_moving_average_matches_pandas():
    dates = pd.bdate_range("2024-01-01", periods=20)
    s = pd.DataFrame({"A": range(20)}, index=dates, dtype=float)
    ma = moving_average(s, window=5)
    np.testing.assert_allclose(ma.iloc[4]["A"], (0 + 1 + 2 + 3 + 4) / 5)
    assert ma.iloc[3].isna().all()  # warmup


def test_ma_cross_long_when_uptrending():
    """Strictly rising series → fast MA always > slow MA → always long."""
    dates = pd.bdate_range("2024-01-01", periods=400)
    rising = pd.DataFrame({"A": np.linspace(100, 200, 400)}, index=dates)
    sig = ma_cross_signal(rising, fast=50, slow=200)
    assert (sig.dropna() == 1.0).all().all()


def test_ma_cross_flat_when_downtrending():
    dates = pd.bdate_range("2024-01-01", periods=400)
    falling = pd.DataFrame({"A": np.linspace(200, 100, 400)}, index=dates)
    sig = ma_cross_signal(falling, fast=50, slow=200)
    assert (sig.dropna() == 0.0).all().all()


def test_ma_cross_warmup_is_nan():
    dates = pd.bdate_range("2024-01-01", periods=300)
    series = pd.DataFrame({"A": np.linspace(100, 200, 300)}, index=dates)
    sig = ma_cross_signal(series, fast=50, slow=200)
    assert sig.iloc[:199].isna().all().all()
    assert sig.iloc[199:].notna().all().all()


def test_ma_cross_rejects_bad_windows():
    dates = pd.bdate_range("2024-01-01", periods=10)
    s = pd.DataFrame({"A": range(10)}, index=dates, dtype=float)
    with pytest.raises(ValueError):
        ma_cross_signal(s, fast=50, slow=50)
    with pytest.raises(ValueError):
        ma_cross_signal(s, fast=200, slow=50)


# ---------- RSI ----------

def test_rsi_constant_series_is_nan_or_100():
    """Constant prices → no gain, no loss → RSI undefined; we return 100."""
    dates = pd.bdate_range("2024-01-01", periods=50)
    flat = pd.DataFrame({"A": [100.0] * 50}, index=dates)
    r = rsi(flat, window=14)
    # After warmup, all RSI values should be 100 (no losses → division convention)
    assert (r.iloc[14:] == 100.0).all().all()


def test_rsi_strictly_rising_approaches_100():
    dates = pd.bdate_range("2024-01-01", periods=100)
    rising = pd.DataFrame({"A": np.linspace(100, 200, 100)}, index=dates)
    r = rsi(rising, window=14)
    # Strictly rising = no losses → RSI exactly 100
    assert (r.iloc[14:] == 100.0).all().all()


def test_rsi_strictly_falling_approaches_zero():
    dates = pd.bdate_range("2024-01-01", periods=100)
    falling = pd.DataFrame({"A": np.linspace(200, 100, 100)}, index=dates)
    r = rsi(falling, window=14)
    # Strictly falling → no gains → RSI = 0
    assert (r.iloc[14:] < 1.0).all().all()


def test_rsi_in_zero_to_hundred_range():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=200)
    walk = pd.DataFrame(
        {"A": 100.0 + rng.normal(0, 1, 200).cumsum()},
        index=dates,
    )
    r = rsi(walk, window=14).dropna()
    assert (r >= 0.0).all().all()
    assert (r <= 100.0).all().all()


def test_rsi_signal_goes_long_at_oversold():
    """Construct an obviously-oversold series, confirm signal triggers."""
    dates = pd.bdate_range("2024-01-01", periods=80)
    # 50 days of decline, then 30 days of recovery
    prices = np.concatenate([
        np.linspace(100, 50, 50),
        np.linspace(50, 80, 30),
    ])
    df = pd.DataFrame({"A": prices}, index=dates)
    sig = rsi_signal(df, window=14, lower=30, upper=70)
    # Somewhere during the decline, signal should turn long (1).
    # (Falling RSI eventually breaches 30.)
    assert (sig == 1.0).any().any()


# ---------- MACD ----------

def test_macd_zero_for_constant_series():
    dates = pd.bdate_range("2024-01-01", periods=100)
    flat = pd.DataFrame({"A": [100.0] * 100}, index=dates)
    m = macd(flat)
    np.testing.assert_allclose(m.macd.dropna().to_numpy(), 0.0, atol=1e-10)


def test_macd_positive_during_uptrend():
    dates = pd.bdate_range("2024-01-01", periods=200)
    rising = pd.DataFrame({"A": np.linspace(100, 200, 200)}, index=dates)
    m = macd(rising)
    # In an uptrend, fast EMA > slow EMA → macd line > 0
    assert (m.macd.dropna() > 0).all().all()


def test_macd_signal_long_during_uptrend():
    dates = pd.bdate_range("2024-01-01", periods=200)
    rising = pd.DataFrame({"A": np.linspace(100, 200, 200)}, index=dates)
    sig = macd_signal(rising)
    # Once warmup completes, an uptrend keeps macd > signal_line
    assert (sig.iloc[40:].dropna() == 1.0).all().all()


def test_macd_rejects_bad_windows():
    dates = pd.bdate_range("2024-01-01", periods=20)
    s = pd.DataFrame({"A": range(20)}, index=dates, dtype=float)
    with pytest.raises(ValueError):
        macd(s, fast=26, slow=12)
    with pytest.raises(ValueError):
        macd(s, signal=0)
