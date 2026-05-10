"""Monthly rebalance of momentum 12-1 strategy on Alpaca paper.

Workflow:
  1. Compute target weights via 12-1 momentum top-N% on filtered SPX.
  2. Pull current Alpaca positions + equity.
  3. Compute the trade list to align (current → target).
  4. Print plan.
  5. With --execute, submit notional market orders.

Default is *dry run* — running without --execute prints what would
happen and exits. Forces you to read the plan before clicking the
trigger.

Usage:
    python -m scripts.paper_rebalance                  # dry-run
    python -m scripts.paper_rebalance --execute        # send orders
    python -m scripts.paper_rebalance --top-pct 0.05   # narrower
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from quant import data, universe
from quant.factors import momentum_12_1
from scripts.check_alpaca import _load_workspace_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("liquid30", "sp500"),
                        default="sp500")
    parser.add_argument("--top-pct", type=float, default=0.10,
                        help="Fraction of universe to hold (default 10%%)")
    parser.add_argument("--listed-before", default="2010-01-01")
    parser.add_argument("--execute", action="store_true",
                        help="Actually submit orders. Without this, dry run.")
    parser.add_argument("--min-trade", type=float, default=10.0,
                        help="Skip rebalances smaller than this dollar amount")
    args = parser.parse_args()

    _load_workspace_env()
    from quant.broker import AlpacaBroker

    broker = AlpacaBroker.from_env()
    if not broker.is_paper:
        print("✗ refusing to run on a live account", file=sys.stderr)
        return 2

    snap = broker.snapshot()
    print(f"[rebal] {'paper' if broker.is_paper else 'LIVE'} account: "
          f"equity ${snap.equity:,.2f}, cash ${snap.cash:,.2f}, "
          f"{len(snap.positions)} positions")

    # ---- Compute target weights ---------------------------------
    tickers = universe.sp500() if args.universe == "sp500" else universe.liquid_30()
    print(f"[rebal] universe={args.universe}, downloading bars (cached)...")
    prices = data.load(tickers, start="2024-01-01")
    if args.listed_before:
        prices = data.filter_by_first_bar(prices, listed_before=args.listed_before)

    scores = momentum_12_1(prices.close)
    today_scores = scores.iloc[-1].dropna()
    if today_scores.empty:
        print("✗ no momentum scores computable today (warmup missing?)", file=sys.stderr)
        return 1

    n_pick = max(3, int(-(-len(today_scores) * args.top_pct // 1)))
    n_pick = min(n_pick, len(today_scores))
    target_tickers = today_scores.nlargest(n_pick).index.tolist()
    target_weight = 1.0 / n_pick
    target_dollars = {t: target_weight * snap.equity for t in target_tickers}
    print(f"[rebal] target: {n_pick} tickers, "
          f"${target_weight * snap.equity:,.2f} each "
          f"({target_weight * 100:.2f}%)")

    # ---- Diff against current positions -------------------------
    current = snap.position_map()
    plan: list[tuple[str, str, float]] = []  # (ticker, side, dollars)

    # Sells / liquidations: in current but not in target
    for ticker, pos in current.items():
        if ticker not in target_dollars and pos.qty > 0:
            plan.append((ticker, "sell-all", pos.market_value))

    # Buys / increases: in target
    for ticker, target_d in target_dollars.items():
        current_d = current[ticker].market_value if ticker in current else 0.0
        delta = target_d - current_d
        if abs(delta) < args.min_trade:
            continue
        if delta > 0:
            plan.append((ticker, "buy", delta))
        else:
            plan.append((ticker, "sell", -delta))

    # ---- Print plan --------------------------------------------
    if not plan:
        print("[rebal] portfolio already aligned with target — nothing to do.")
        return 0

    print(f"\n  {'Ticker':<8} {'Side':<10} {'Dollars':>14}")
    print("  " + "-" * 36)
    for ticker, side, dollars in plan:
        print(f"  {ticker:<8} {side:<10} ${dollars:>12,.2f}")
    total_buy = sum(d for _, s, d in plan if s == "buy")
    total_sell = sum(d for _, s, d in plan if s in ("sell", "sell-all"))
    print(f"\n  Total buy:  ${total_buy:>12,.2f}")
    print(f"  Total sell: ${total_sell:>12,.2f}")

    if not args.execute:
        print("\n[rebal] dry run — pass --execute to actually submit orders.")
        return 0

    # ---- Execute ----------------------------------------------
    print("\n[rebal] submitting orders...")
    submitted = 0
    for ticker, side, dollars in plan:
        try:
            if side == "sell-all":
                oid = broker.close_position(ticker)
            else:
                oid = broker.submit_notional_order(ticker, dollars=dollars, side=side)
            print(f"  ✓ {side:<10} {ticker:<8} ${dollars:>10,.2f}  → {oid}")
            submitted += 1
        except Exception as exc:  # noqa: BLE001 — broker errors are varied
            print(f"  ✗ {side:<10} {ticker:<8} FAILED: {exc}")
    print(f"\n[rebal] submitted {submitted}/{len(plan)} orders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
