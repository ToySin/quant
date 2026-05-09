"""Value + quality factor correctness on synthetic fundamentals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.factors import quality, value
from quant.fundamentals import RELEVANT_KEYS, Fundamentals


def _make_funds(rows: dict[str, dict]) -> Fundamentals:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    # Ensure all relevant keys present (NaN if missing)
    for k in RELEVANT_KEYS:
        if k not in df.columns:
            df[k] = np.nan
    return Fundamentals(df=df, snapshot_date=pd.Timestamp("2026-05-09"))


def test_book_yield_inverts_pb():
    funds = _make_funds({
        "A": {"priceToBook": 1.0},
        "B": {"priceToBook": 4.0},
        "C": {"priceToBook": 0.5},
    })
    by = value.book_yield(funds)
    assert by.loc["A"] == pytest.approx(1.0)
    assert by.loc["B"] == pytest.approx(0.25)
    assert by.loc["C"] == pytest.approx(2.0)


def test_book_yield_drops_non_positive():
    funds = _make_funds({
        "A": {"priceToBook": 2.0},
        "Bad": {"priceToBook": -1.0},
        "Zero": {"priceToBook": 0.0},
    })
    by = value.book_yield(funds)
    assert pd.notna(by.loc["A"])
    assert pd.isna(by.loc["Bad"])
    assert pd.isna(by.loc["Zero"])


def test_value_composite_higher_when_cheaper():
    funds = _make_funds({
        "Cheap":  {"priceToBook": 0.5, "trailingPE": 5.0},
        "Mid":    {"priceToBook": 2.0, "trailingPE": 15.0},
        "Pricey": {"priceToBook": 10.0, "trailingPE": 50.0},
    })
    score = value.composite(funds)
    assert score.loc["Cheap"] > score.loc["Mid"] > score.loc["Pricey"]


def test_value_as_panel_broadcasts_constant():
    funds = _make_funds({
        "A": {"priceToBook": 1.0, "trailingPE": 10.0},
        "B": {"priceToBook": 2.0, "trailingPE": 20.0},
    })
    score = value.composite(funds)
    dates = pd.bdate_range("2024-01-01", periods=10)
    panel = value.as_panel(score, dates)
    assert panel.shape == (10, 2)
    # Every row identical (snapshot fundamentals)
    np.testing.assert_array_equal(panel.iloc[0].to_numpy(), panel.iloc[5].to_numpy())


def test_quality_composite_prefers_high_roe_low_debt_high_margin():
    funds = _make_funds({
        "Good":  {"returnOnEquity": 0.3, "debtToEquity": 10.0, "profitMargins": 0.2},
        "Mid":   {"returnOnEquity": 0.1, "debtToEquity": 50.0, "profitMargins": 0.05},
        "Bad":   {"returnOnEquity": 0.02, "debtToEquity": 200.0, "profitMargins": 0.01},
    })
    score = quality.composite(funds)
    assert score.loc["Good"] > score.loc["Mid"] > score.loc["Bad"]


def test_quality_low_leverage_drops_negative_debt_equity():
    funds = _make_funds({
        "Normal": {"debtToEquity": 50.0},
        "Negative": {"debtToEquity": -10.0},
    })
    lev = quality.low_leverage(funds)
    assert pd.notna(lev.loc["Normal"])
    assert pd.isna(lev.loc["Negative"])


def test_zscore_handles_constant_input():
    """If all values equal, z-score is 0 (not NaN), so factor blends remain finite."""
    funds = _make_funds({
        "A": {"priceToBook": 2.0, "trailingPE": 20.0},
        "B": {"priceToBook": 2.0, "trailingPE": 20.0},
    })
    score = value.composite(funds)
    assert (score.dropna() == 0.0).all()
