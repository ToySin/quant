"""Backtest accounting — apply target weights to price returns.

Pure pandas, no broker. The model is:
    daily_pnl_t = sum_i  weight_{t-1, i} * return_{t, i}  -  cost_t

Weights are *target* weights set the previous close; returns are
realized at next close. Trading cost is a flat per-side rate
multiplied by the L1 turnover at each rebalance.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series          # daily portfolio return, net of costs
    gross_returns: pd.Series    # daily portfolio return, before costs
    weights: pd.DataFrame       # weights actually applied each day
    turnover: pd.Series         # L1 turnover per day (0 between rebalances)


def run(returns: pd.DataFrame, weights: pd.DataFrame, *,
        cost_bps: float = 5.0) -> BacktestResult:
    """Apply `weights` to `returns` with a per-side trading cost.

    `cost_bps` is one-way in basis points: a 5 bps cost means each
    1.0 of turnover is charged 0.0005 in P&L. Conservative default
    for liquid US large-cap.
    """
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")

    # Align: weights are target end-of-day-t, applied to return_{t+1}.
    weights, returns = weights.align(returns, join="inner", axis=None, copy=False)
    weights = weights.fillna(0.0)
    returns = returns.fillna(0.0)

    applied_weights = weights.shift(1).fillna(0.0)
    gross_daily = (applied_weights * returns).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost_daily = turnover * (cost_bps / 10_000.0)
    net_daily = gross_daily - cost_daily

    return BacktestResult(
        returns=net_daily,
        gross_returns=gross_daily,
        weights=applied_weights,
        turnover=turnover,
    )
