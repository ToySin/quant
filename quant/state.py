"""Persistent state for daily/monthly automation.

Lightweight JSON file store under data/state/. Tracks last-seen
equity (for daily PnL diff) and last rebalance month (for monthly
auto-rebalance idempotency).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from quant.cache import cache_dir


def _state_path() -> Path:
    p = cache_dir().parent / "state" / "portfolio.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class PortfolioState:
    last_equity: float | None = None
    last_snapshot_date: str | None = None    # ISO date
    last_rebalance_month: str | None = None  # "2026-05"
    peak_equity: float = 0.0
    history: list[dict] = field(default_factory=list)  # list of {date, equity}

    @classmethod
    def load(cls) -> "PortfolioState":
        path = _state_path()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        return cls(
            last_equity=data.get("last_equity"),
            last_snapshot_date=data.get("last_snapshot_date"),
            last_rebalance_month=data.get("last_rebalance_month"),
            peak_equity=float(data.get("peak_equity", 0.0)),
            history=list(data.get("history", [])),
        )

    def save(self) -> None:
        _state_path().write_text(json.dumps({
            "last_equity": self.last_equity,
            "last_snapshot_date": self.last_snapshot_date,
            "last_rebalance_month": self.last_rebalance_month,
            "peak_equity": self.peak_equity,
            "history": self.history,
        }, indent=2))

    def record_snapshot(self, equity: float, today: date) -> None:
        """Update equity, peak, history. Caller must call save()."""
        self.last_equity = equity
        self.last_snapshot_date = today.isoformat()
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.history.append({"date": today.isoformat(), "equity": equity})
        # Cap history to last 365 entries to prevent unbounded growth
        if len(self.history) > 365:
            self.history = self.history[-365:]

    def drawdown_from_peak(self, equity: float) -> float:
        """Current drawdown from peak (negative if below peak, 0 if at/above)."""
        if self.peak_equity <= 0:
            return 0.0
        return (equity - self.peak_equity) / self.peak_equity
