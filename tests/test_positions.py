"""Position adapter correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.positions import equal_weight_when_long, fixed_fraction


def test_equal_weight_distributes_uniformly():
    dates = pd.bdate_range("2024-01-01", periods=5)
    mask = pd.DataFrame(
        {"A": [1, 1, 1, 0, 0],
         "B": [1, 1, 0, 0, 1],
         "C": [0, 1, 0, 0, 1]},
        index=dates, dtype=float,
    )
    weights = equal_weight_when_long(mask)
    # Day 0: A and B long → 0.5 each
    np.testing.assert_allclose(weights.iloc[0], [0.5, 0.5, 0.0])
    # Day 1: all three long → 1/3 each
    np.testing.assert_allclose(weights.iloc[1], [1 / 3, 1 / 3, 1 / 3])
    # Day 2: only A long → 1.0
    np.testing.assert_allclose(weights.iloc[2], [1.0, 0.0, 0.0])
    # Day 3: nothing long → all cash (zeros)
    np.testing.assert_allclose(weights.iloc[3], [0.0, 0.0, 0.0])


def test_equal_weight_handles_nan_as_flat():
    dates = pd.bdate_range("2024-01-01", periods=3)
    mask = pd.DataFrame(
        {"A": [np.nan, 1, 1], "B": [1, np.nan, 1]},
        index=dates,
    )
    weights = equal_weight_when_long(mask)
    np.testing.assert_allclose(weights.iloc[0], [0.0, 1.0])
    np.testing.assert_allclose(weights.iloc[1], [1.0, 0.0])
    np.testing.assert_allclose(weights.iloc[2], [0.5, 0.5])


def test_fixed_fraction_caps_at_full_exposure():
    dates = pd.bdate_range("2024-01-01", periods=2)
    # 5 longs at 30% each = 150% requested → should scale down to 100%
    mask = pd.DataFrame(
        np.ones((2, 5)),
        index=dates, columns=list("ABCDE"),
    )
    weights = fixed_fraction(mask, fraction=0.3)
    np.testing.assert_allclose(weights.sum(axis=1), [1.0, 1.0])
    # All five should get the same scaled-down weight
    assert weights.iloc[0].nunique() == 1


def test_fixed_fraction_passes_through_when_under_full():
    dates = pd.bdate_range("2024-01-01", periods=1)
    mask = pd.DataFrame(
        [[1, 0, 1, 0, 0]],
        index=dates, columns=list("ABCDE"), dtype=float,
    )
    weights = fixed_fraction(mask, fraction=0.3)
    # 2 longs at 30% = 60% total, no scaling needed
    np.testing.assert_allclose(weights.iloc[0], [0.3, 0.0, 0.3, 0.0, 0.0])


def test_fixed_fraction_rejects_bad_fraction():
    dates = pd.bdate_range("2024-01-01", periods=1)
    mask = pd.DataFrame([[1]], index=dates, columns=["A"], dtype=float)
    with pytest.raises(ValueError):
        fixed_fraction(mask, fraction=0.0)
    with pytest.raises(ValueError):
        fixed_fraction(mask, fraction=1.1)
