"""Factor signal correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.factors import (
    inverse_volatility,
    momentum_12_1,
    momentum_lookback,
    realized_volatility,
)


def test_momentum_lookback_matches_explicit_calc(synthetic_close):
    score = momentum_lookback(synthetic_close, lookback=20, skip=5)
    eval_t = synthetic_close.index[200]
    end_t = synthetic_close.index[200 - 5]
    start_t = synthetic_close.index[200 - 5 - 20]
    expected = synthetic_close.loc[end_t] / synthetic_close.loc[start_t] - 1
    pd.testing.assert_series_equal(
        score.loc[eval_t].rename(None),
        expected.rename(None),
        check_names=False,
    )


def test_momentum_lookback_rejects_bad_inputs(synthetic_close):
    with pytest.raises(ValueError):
        momentum_lookback(synthetic_close, lookback=0)
    with pytest.raises(ValueError):
        momentum_lookback(synthetic_close, lookback=10, skip=-1)


def test_momentum_12_1_skips_recent_month(synthetic_close):
    """Boost the price on the last day for one ticker. 12-1 should NOT
    pick that up because the recent month is excluded."""
    boosted = synthetic_close.copy()
    boosted.iloc[-1, 0] *= 2.0  # giant pop on the latest day for T00
    score = momentum_12_1(boosted)
    last_score = score.iloc[-1]
    # T00's score should be unchanged from the unboosted version
    base_score = momentum_12_1(synthetic_close).iloc[-1, 0]
    assert np.isclose(last_score.iloc[0], base_score)


def test_realized_volatility_positive(synthetic_close):
    vol = realized_volatility(synthetic_close, window=21)
    assert (vol.dropna() > 0).all().all()


def test_inverse_volatility_is_negation(synthetic_close):
    iv = inverse_volatility(synthetic_close, window=21)
    rv = realized_volatility(synthetic_close, window=21, annualize=False)
    pd.testing.assert_frame_equal(iv, -rv)


def test_realized_volatility_rejects_bad_window(synthetic_close):
    with pytest.raises(ValueError):
        realized_volatility(synthetic_close, window=1)
