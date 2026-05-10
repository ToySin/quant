"""Sanity-check Alpaca paper trading credentials + connection.

Usage:
    python -m scripts.check_alpaca
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_workspace_env() -> None:
    """Load ALPACA_* keys from the assistant-hub quant workspace .env
    if present. The quant repo's own .env is also checked as a fallback.
    """
    candidates = [
        Path.home() / "repositories" / "assisthub-ws-quant" / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def main() -> int:
    _load_workspace_env()

    if "ALPACA_PAPER_KEY" not in os.environ or "ALPACA_PAPER_SECRET" not in os.environ:
        print("✗ ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET not set", file=sys.stderr)
        print("  Add them to ~/repositories/assisthub-ws-quant/.env", file=sys.stderr)
        return 1

    from quant.broker import AlpacaBroker
    broker = AlpacaBroker.from_env()
    print(f"[check] connected to {'paper' if broker.is_paper else 'LIVE'} trading")

    snap = broker.snapshot()
    print(f"  Equity:        ${snap.equity:,.2f}")
    print(f"  Cash:          ${snap.cash:,.2f}")
    print(f"  Buying power:  ${snap.buying_power:,.2f}")
    print(f"  Positions:     {len(snap.positions)}")
    for p in snap.positions:
        print(f"    {p.ticker:<7} qty={p.qty:>10.4f} "
              f"value=${p.market_value:>11,.2f} "
              f"PnL=${p.unrealized_pl:>+11,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
