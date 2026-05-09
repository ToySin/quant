"""End-to-end demo: 12-1 momentum on a 30-name large-cap US universe.

Pulls daily bars from yfinance (cached locally), computes the 12-1
momentum signal, builds a monthly-rebalanced top-decile long-only
portfolio, runs the backtest, and prints performance metrics.

Usage:
    python -m scripts.run_momentum
    python -m scripts.run_momentum --start 2018-01-01
    python -m scripts.run_momentum --top-pct 0.2 --cost-bps 10
"""

from __future__ import annotations

import argparse

from quant import data, report, universe
from quant.backtest import run as run_backtest
from quant.factors import momentum_12_1
from quant.portfolio import top_decile_long_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("liquid30", "sp500"),
                        default="liquid30",
                        help="Universe to backtest on (default: liquid30)")
    parser.add_argument("--start", default="2019-01-01",
                        help="Backtest start date (default: 2019-01-01)")
    parser.add_argument("--end", default=None,
                        help="Backtest end date (default: latest available)")
    parser.add_argument("--top-pct", type=float, default=0.3,
                        help="Top fraction by score to hold (default: 0.3)")
    parser.add_argument("--cost-bps", type=float, default=5.0,
                        help="One-way trading cost in bps (default: 5)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download from yfinance, ignore cache")
    args = parser.parse_args()

    tickers = universe.sp500() if args.universe == "sp500" else universe.liquid_30()
    print(f"[run_momentum] universe={args.universe} ({len(tickers)} tickers)")
    print(f"[run_momentum] window=[{args.start}, {args.end or 'latest'}]")

    print("[run_momentum] downloading bars (cached after first run)...")
    prices = data.load(tickers, start=args.start, end=args.end, refresh=args.refresh)
    print(f"[run_momentum] loaded close panel: "
          f"{prices.close.shape[0]} days x {prices.close.shape[1]} tickers")

    print("[run_momentum] computing 12-1 momentum...")
    scores = momentum_12_1(prices.close)

    print("[run_momentum] building top-decile monthly-rebalance weights...")
    weights = top_decile_long_only(scores, top_pct=args.top_pct)

    print("[run_momentum] running backtest...")
    result = run_backtest(prices.returns(), weights, cost_bps=args.cost_bps)

    print("\n=== Strategy: 12-1 Momentum, top-{:.0%} long-only ===".format(args.top_pct))
    print(report.metrics(result.returns).as_table())

    print("\n=== Comparison: equal-weight buy-and-hold (no signal) ===")
    bh_returns = prices.returns().mean(axis=1)
    print(report.metrics(bh_returns.dropna()).as_table())


if __name__ == "__main__":
    main()
