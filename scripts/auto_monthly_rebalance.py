"""Cron-friendly monthly rebalance with Discord summary.

Designed to run from cron every weekday morning. The script checks
its own state file:

  - Already rebalanced this month? → exit 0 (nothing to do)
  - Otherwise → run the same logic as paper_rebalance.py and submit
    orders, then post a Discord summary

The state file uses (year, month) as the idempotency key, so the
first weekday cron firing of any new month will rebalance, and
later firings that month are no-ops. This handles month-end weekend
spillover automatically (1st of month on a Sunday → first run is
on Monday the 2nd or 3rd).

Usage from cron:
    0 22 * * 1-5  /Users/donshin/repositories/quant/.venv/bin/python \\
        -m scripts.auto_monthly_rebalance >> /tmp/quant_rebal.log 2>&1
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd

from quant import data, universe
from quant.factors import momentum_12_1
from quant.filters import drop_extreme_momentum
from quant.notify import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_RED,
    DiscordWebhook,
    Field,
)
from quant.state import PortfolioState
from scripts.check_alpaca import _load_workspace_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-pct", type=float, default=0.10)
    parser.add_argument("--max-momentum", type=float, default=4.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--force", action="store_true",
                        help="Run even if already rebalanced this month")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute plan, don't submit orders")
    args = parser.parse_args()

    _load_workspace_env()
    from quant.broker import AlpacaBroker

    today = date.today()
    month_key = f"{today.year}-{today.month:02d}"

    state = PortfolioState.load()
    if state.last_rebalance_month == month_key and not args.force:
        print(f"[auto-rebal] already rebalanced {month_key} — skipping")
        return 0

    broker = AlpacaBroker.from_env()
    if not broker.is_paper:
        print("[auto-rebal] refusing to run on live account", file=sys.stderr)
        return 2

    snap = broker.snapshot()
    print(f"[auto-rebal] {today.isoformat()} — equity ${snap.equity:,.2f}, "
          f"{len(snap.positions)} positions")

    # Compute target
    tickers = universe.sp500()
    print("[auto-rebal] downloading bars...")
    prices = data.load(tickers, start="2024-01-01")
    scores = momentum_12_1(prices.close).iloc[-1]
    scores = drop_extreme_momentum(scores, max_score=args.max_momentum).dropna()
    if scores.empty:
        _post_failure(f"No momentum scores computable on {today.isoformat()}")
        return 1

    n_pick = max(3, int(-(-len(scores) * args.top_pct // 1)))
    n_pick = min(n_pick, len(scores))
    target_tickers = scores.nlargest(n_pick).index.tolist()
    target_dollars = snap.equity / n_pick

    # Diff against current
    current = snap.position_map()
    plan: list[tuple[str, str, float]] = []
    for ticker, pos in current.items():
        if ticker not in target_tickers and pos.qty > 0:
            plan.append((ticker, "sell-all", pos.market_value))
    for ticker in target_tickers:
        current_d = current[ticker].market_value if ticker in current else 0.0
        delta = target_dollars - current_d
        if abs(delta) < 10.0:
            continue
        plan.append((ticker, "buy" if delta > 0 else "sell", abs(delta)))

    if not plan:
        print("[auto-rebal] portfolio already aligned — nothing to trade")
        state.last_rebalance_month = month_key
        state.save()
        _post_summary(today, snap, [], n_pick, args.dry_run, no_op=True)
        return 0

    if args.dry_run:
        print("[auto-rebal] dry-run plan:")
        for t, side, d in plan:
            print(f"  {side:<10} {t:<8} ${d:,.2f}")
        return 0

    # Execute
    print(f"[auto-rebal] submitting {len(plan)} orders...")
    submitted = []
    failed = []
    for ticker, side, dollars in plan:
        try:
            if side == "sell-all":
                broker.close_position(ticker)
            else:
                broker.submit_notional_order(ticker, dollars=dollars, side=side)
            submitted.append((ticker, side, dollars))
        except Exception as exc:  # noqa: BLE001
            failed.append((ticker, side, dollars, str(exc)))
            print(f"  ✗ {side} {ticker} ${dollars:,.2f} FAILED: {exc}",
                  file=sys.stderr)

    state.last_rebalance_month = month_key
    state.save()

    print(f"[auto-rebal] {len(submitted)}/{len(plan)} orders submitted")
    _post_summary(today, snap, submitted, n_pick, args.dry_run,
                  failed=failed)
    return 0 if not failed else 1


def _post_summary(today: date, snap, submitted: list,
                  n_pick: int, dry_run: bool, *,
                  failed: list | None = None,
                  no_op: bool = False) -> None:
    try:
        hook = DiscordWebhook.from_env()
    except RuntimeError as exc:
        print(f"[auto-rebal] {exc}", file=sys.stderr)
        return

    if no_op:
        title = f"♻️ Monthly rebalance — {today.isoformat()} (no-op)"
        description = "Portfolio already aligned with target."
        color = COLOR_BLUE
    elif failed:
        title = f"⚠️ Monthly rebalance — {today.isoformat()} (partial)"
        color = COLOR_RED
        description = (f"Submitted {len(submitted)}/{len(submitted) + len(failed)} "
                       f"orders. {len(failed)} failed.")
    else:
        title = f"✅ Monthly rebalance — {today.isoformat()}"
        color = COLOR_GREEN
        description = f"Submitted {len(submitted)} orders."

    total_buy = sum(d for _, s, d in submitted if s == "buy")
    total_sell = sum(d for _, s, d in submitted if s in ("sell", "sell-all"))

    fields = [
        Field("Equity", f"${snap.equity:,.2f}"),
        Field("Target tickers", str(n_pick)),
        Field("Total buy", f"${total_buy:,.2f}"),
        Field("Total sell", f"${total_sell:,.2f}"),
    ]
    if submitted:
        # Show first 8 trades; Discord field values cap at 1024 chars
        sample = "\n".join(
            f"{s:<10} {t:<6} ${d:,.0f}" for t, s, d in submitted[:8]
        )
        if len(submitted) > 8:
            sample += f"\n... +{len(submitted) - 8} more"
        fields.append(Field("Trades", f"```{sample}```", inline=False))
    if failed:
        sample = "\n".join(f"{s} {t}: {err[:60]}" for t, s, _, err in failed[:5])
        fields.append(Field("Failures", f"```{sample}```", inline=False))

    hook.post_embed(title=title, description=description, color=color,
                    fields=fields, footer="momentum 12-1 paper trading")


def _post_failure(message: str) -> None:
    try:
        hook = DiscordWebhook.from_env()
        hook.post_embed(
            title="❌ Monthly rebalance failed",
            description=message,
            color=COLOR_RED,
        )
    except RuntimeError:
        pass


if __name__ == "__main__":
    sys.exit(main())
