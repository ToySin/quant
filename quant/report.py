"""Performance metrics — Sharpe, Sortino, MDD, CAGR, Calmar.

Hand-rolled to keep the dependency surface small. All inputs are
daily simple returns (decimal, e.g. 0.01 = +1%). Annualization uses
the standard 252 trading days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Metrics:
    cagr: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    n_days: int

    def as_table(self) -> str:
        return (
            f"  CAGR              {self.cagr * 100:>7.2f}%\n"
            f"  Volatility (ann)  {self.annualized_volatility * 100:>7.2f}%\n"
            f"  Sharpe            {self.sharpe:>7.2f}\n"
            f"  Sortino           {self.sortino:>7.2f}\n"
            f"  Max drawdown      {self.max_drawdown * 100:>7.2f}%\n"
            f"  Calmar            {self.calmar:>7.2f}\n"
            f"  Hit rate          {self.hit_rate * 100:>7.2f}%\n"
            f"  Trading days      {self.n_days:>7d}"
        )


def metrics(returns: pd.Series, *, risk_free_rate: float = 0.0) -> Metrics:
    """Compute the standard battery of performance metrics.

    `risk_free_rate` is annualized; converted to per-period internally.
    """
    r = returns.dropna()
    if r.empty:
        raise ValueError("returns series is empty")

    n = len(r)
    ann_factor = TRADING_DAYS

    cumulative = (1 + r).prod()
    cagr = cumulative ** (ann_factor / n) - 1 if n > 0 else 0.0
    ann_vol = r.std(ddof=1) * np.sqrt(ann_factor)

    rf_daily = (1 + risk_free_rate) ** (1 / ann_factor) - 1
    excess = r - rf_daily
    sharpe = (excess.mean() * ann_factor) / (r.std(ddof=1) * np.sqrt(ann_factor)) \
        if r.std(ddof=1) > 0 else 0.0

    downside = r[r < 0]
    downside_vol = downside.std(ddof=1) * np.sqrt(ann_factor) if len(downside) > 1 else 0.0
    sortino = (excess.mean() * ann_factor) / downside_vol if downside_vol > 0 else 0.0

    mdd = _max_drawdown(r)
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    hit_rate = float((r > 0).sum()) / n

    return Metrics(
        cagr=float(cagr),
        annualized_volatility=float(ann_vol),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(mdd),
        calmar=float(calmar),
        hit_rate=float(hit_rate),
        n_days=int(n),
    )


def _max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough drawdown, expressed as a negative number."""
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min()) if not dd.empty else 0.0
