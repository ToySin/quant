"""Combination signal correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.signals import (
    ma_macd_confirm,
    price_above_ma_with_macd,
    trend_filtered_macd,
)


def _build_regime_series(up_days: int, down_days: int) -> pd.DataFrame:
    """Synthetic price: rises strictly for `up_days`, then falls for `down_days`."""
    dates = pd.bdate_range("2020-01-01", periods=up_days + down_days)
    up = np.linspace(100, 200, up_days)
    down = np.linspace(200, 50, down_days)
    return pd.DataFrame({"A": np.concatenate([up, down])}, index=dates)


def test_trend_filtered_macd_long_during_clear_uptrend():
    """During a strong uptrend, both filters should agree → long."""
    df = _build_regime_series(up_days=400, down_days=0)
    sig = trend_filtered_macd(df).dropna()
    # Once warmup completes, all signals during sustained uptrend should be 1
    assert (sig == 1.0).all().all()


def test_trend_filtered_macd_flat_during_clear_downtrend():
    """During downtrend, MA50 < MA200 → never long, regardless of MACD."""
    # Long uptrend to establish positive trend, then sustained downtrend
    df = _build_regime_series(up_days=300, down_days=400)
    sig = trend_filtered_macd(df)
    # Late in downtrend: MA50 has dropped below MA200 → no long signals
    late = sig.iloc[-50:].dropna()
    assert (late == 0.0).all().all()


def test_trend_filtered_macd_warmup_is_nan():
    df = _build_regime_series(up_days=300, down_days=0)
    sig = trend_filtered_macd(df, slow_ma=200)
    # First 199 days have no MA200 → must be NaN
    assert sig.iloc[:199].isna().all().all()


def test_ma_macd_confirm_matches_trend_filtered():
    """Pattern B is currently identical to Pattern A — sanity check
    that they produce the same output."""
    df = _build_regime_series(up_days=300, down_days=200)
    sig_a = trend_filtered_macd(df)
    sig_b = ma_macd_confirm(df)
    pd.testing.assert_frame_equal(sig_a, sig_b)


def test_price_above_ma_reacts_faster_than_ma_cross():
    """At a regime change, price-above-MA flips before MA50/MA200 cross.

    Setup: rise then fall. Compare when each filter exits the long.
    """
    df = _build_regime_series(up_days=300, down_days=200)
    fast_combo = price_above_ma_with_macd(df, ma_window=200)
    slow_combo = trend_filtered_macd(df, fast_ma=50, slow_ma=200)

    # First date where each turned flat after the peak
    peak_date = df.idxmax().iloc[0]

    fast_after = fast_combo.loc[peak_date:].iloc[:, 0]
    slow_after = slow_combo.loc[peak_date:].iloc[:, 0]

    fast_first_flat = fast_after[fast_after == 0.0].index.min()
    slow_first_flat = slow_after[slow_after == 0.0].index.min()

    # Price-above-MA should exit at or before MA cross
    assert fast_first_flat <= slow_first_flat


def test_combos_emit_only_zero_or_one_or_nan():
    df = _build_regime_series(up_days=300, down_days=200)
    for fn in (trend_filtered_macd, ma_macd_confirm, price_above_ma_with_macd):
        sig = fn(df)
        non_nan = sig.dropna().to_numpy().ravel()
        assert set(np.unique(non_nan)).issubset({0.0, 1.0})
