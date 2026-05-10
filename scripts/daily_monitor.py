"""Daily snapshot of paper account → Discord embed.

Run from cron after market close each weekday. Pulls equity / cash /
positions from Alpaca, diffs against yesterday's stored equity to
compute daily PnL, and posts a Discord message.

Idempotent: re-running the same day overwrites the day's history
entry rather than double-counting.

Usage:
    python -m scripts.daily_monitor                 # post to Discord
    python -m scripts.daily_monitor --no-post       # print only, no Discord
    python -m scripts.daily_monitor --quiet         # only post if ALERT condition
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from quant.notify import (
    COLOR_BLUE,
    COLOR_GREY,
    DiscordWebhook,
    Field,
    color_for_pnl,
)
from quant.state import PortfolioState
from scripts.check_alpaca import _load_workspace_env

# Daily move that triggers an alert when --quiet is on
ALERT_MOVE_THRESHOLD = 0.03   # ±3% daily change
ALERT_DRAWDOWN_THRESHOLD = -0.10  # -10% from peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-post", action="store_true",
                        help="Print, don't send to Discord")
    parser.add_argument("--quiet", action="store_true",
                        help="Only post if alert thresholds tripped")
    args = parser.parse_args()

    _load_workspace_env()
    from quant.broker import AlpacaBroker

    broker = AlpacaBroker.from_env()
    snap = broker.snapshot()
    today = date.today()

    state = PortfolioState.load()
    daily_pnl_dollars = 0.0
    daily_pnl_pct = 0.0
    if state.last_equity is not None:
        daily_pnl_dollars = snap.equity - state.last_equity
        if state.last_equity > 0:
            daily_pnl_pct = daily_pnl_dollars / state.last_equity

    drawdown = state.drawdown_from_peak(snap.equity)

    # Decide whether to alert
    alert_triggered = (
        abs(daily_pnl_pct) >= ALERT_MOVE_THRESHOLD
        or drawdown <= ALERT_DRAWDOWN_THRESHOLD
    )

    state.record_snapshot(snap.equity, today)
    state.save()

    print(f"[monitor] {today.isoformat()} equity ${snap.equity:,.2f} "
          f"({daily_pnl_pct * 100:+.2f}% / ${daily_pnl_dollars:+,.2f})")
    print(f"[monitor] cash ${snap.cash:,.2f}, positions {len(snap.positions)}")
    print(f"[monitor] drawdown from peak: {drawdown * 100:+.2f}%")
    if alert_triggered:
        print("[monitor] ALERT thresholds tripped")

    if args.no_post:
        return 0
    if args.quiet and not alert_triggered:
        print("[monitor] quiet mode + no alert → skipping Discord post")
        return 0

    try:
        hook = DiscordWebhook.from_env()
    except RuntimeError as exc:
        print(f"[monitor] {exc}", file=sys.stderr)
        return 1

    color = color_for_pnl(daily_pnl_dollars) if state.last_equity else COLOR_BLUE
    title = f"📊 Daily snapshot — {today.isoformat()}"
    if alert_triggered:
        title = f"⚠️ ALERT — {today.isoformat()}"

    fields = [
        Field("Equity", f"${snap.equity:,.2f}"),
        Field("Daily PnL", f"{daily_pnl_pct * 100:+.2f}%  (${daily_pnl_dollars:+,.2f})"),
        Field("Cash", f"${snap.cash:,.2f}"),
        Field("Positions", str(len(snap.positions))),
        Field("Drawdown from peak", f"{drawdown * 100:+.2f}%"),
        Field("Peak equity", f"${state.peak_equity:,.2f}"),
    ]

    description = ""
    if alert_triggered:
        triggers = []
        if abs(daily_pnl_pct) >= ALERT_MOVE_THRESHOLD:
            triggers.append(f"daily move {daily_pnl_pct * 100:+.2f}% "
                            f"(threshold ±{ALERT_MOVE_THRESHOLD * 100:.0f}%)")
        if drawdown <= ALERT_DRAWDOWN_THRESHOLD:
            triggers.append(f"drawdown {drawdown * 100:+.2f}% "
                            f"(threshold {ALERT_DRAWDOWN_THRESHOLD * 100:.0f}%)")
        description = "Triggers: " + "; ".join(triggers)

    hook.post_embed(title=title, description=description, color=color,
                    fields=fields, footer="momentum 12-1 paper trading")
    print("[monitor] posted to Discord")
    return 0


if __name__ == "__main__":
    sys.exit(main())
