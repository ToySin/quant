"""Sanity-filter correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.filters import drop_extreme_momentum


def test_drop_extreme_caps_at_threshold():
    s = pd.Series({"A": 0.5, "B": 2.0, "C": 5.0, "D": 10.0})
    out = drop_extreme_momentum(s, max_score=4.0)
    assert out.loc["A"] == 0.5
    assert out.loc["B"] == 2.0
    assert pd.isna(out.loc["C"])
    assert pd.isna(out.loc["D"])


def test_drop_extreme_works_on_dataframe():
    df = pd.DataFrame({
        "A": [0.5, 1.0, 1.5],
        "B": [10.0, 0.5, 5.0],
    })
    out = drop_extreme_momentum(df, max_score=2.0)
    assert pd.isna(out["B"].iloc[0])
    assert pd.isna(out["B"].iloc[2])
    assert out["A"].notna().all()


def test_negative_threshold_rejected():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        drop_extreme_momentum(s, max_score=-1.0)


def test_preserves_existing_nans():
    s = pd.Series({"A": np.nan, "B": 1.0, "C": 5.0})
    out = drop_extreme_momentum(s, max_score=4.0)
    assert pd.isna(out.loc["A"])
    assert out.loc["B"] == 1.0
    assert pd.isna(out.loc["C"])
