"""Unified strategy comparison — factors + signals + combos.

Runs every strategy currently implemented in the package side by side:

  Cross-sectional factors:
    - 12-1 Momentum, top decile, monthly rebalance

  Time-series signals:
    - MA cross 50/200
    - RSI 14 (30/70 mean reversion)
    - MACD 12/26/9

  Combinations:
    - Trend-filtered MACD (MA50/200 regime + MACD entry)
    - Price > MA200 + MACD

  Benchmark:
    - Equal-weight buy & hold

Use --listed-before to drop tickers without enough history, mitigating
the listing-date bias (a 2007 backtest with stocks that didn't trade
until 2012 will look invariant).

Usage:
    python -m scripts.run_all_compare
    python -m scripts.run_all_compare --universe sp500 --start 2010-01-01
    python -m scripts.run_all_compare --universe sp500 --start 2007-01-01 \\
        --listed-before 2007-01-01
"""

from __future__ import annotations

import argparse

import pandas as pd

from quant import data, report, universe
from quant.backtest import run as run_backtest
from quant.factors import momentum_12_1
from quant.portfolio import top_decile_long_only
from quant.positions import equal_weight_when_long
from quant.signals import (
    ma_cross_signal,
    macd_signal,
    price_above_ma_with_macd,
    rsi_signal,
    trend_filtered_macd,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("liquid30", "sp500"),
                        default="liquid30")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--top-pct", type=float, default=0.1,
                        help="Factor top fraction (default 10%%)")
    parser.add_argument("--listed-before", default=None,
                        help="Drop tickers without data before this date "
                             "(mitigates listing-date bias)")
    args = parser.parse_args()

    tickers = universe.sp500() if args.universe == "sp500" else universe.liquid_30()
    print(f"[compare] universe={args.universe} ({len(tickers)} tickers)")
    print(f"[compare] window=[{args.start}, {args.end or 'latest'}]")
    print("[compare] downloading bars (cached after first run)...")
    prices = data.load(tickers, start=args.start, end=args.end)

    if args.listed_before:
        before_n = prices.close.shape[1]
        prices = data.filter_by_first_bar(prices, listed_before=args.listed_before)
        after_n = prices.close.shape[1]
        print(f"[compare] history filter: kept {after_n}/{before_n} tickers "
              f"with data before {args.listed_before}")

    print(f"[compare] panel: {prices.close.shape[0]} days "
          f"x {prices.close.shape[1]} tickers")

    rets = prices.returns()
    rows: list[tuple[str, "report.Metrics"]] = []

    # --- Factor (cross-sectional) ---
    mom_scores = momentum_12_1(prices.close)
    mom_weights = top_decile_long_only(mom_scores, top_pct=args.top_pct)
    mom_result = run_backtest(rets, mom_weights, cost_bps=args.cost_bps)
    rows.append((f"12-1 Momentum top-{int(args.top_pct * 100)}%",
                 report.metrics(mom_result.returns)))

    # --- Time-series signals ---
    ts_strategies = {
        "MA cross 50/200":      ma_cross_signal(prices.close, fast=50, slow=200),
        "RSI 14 (30/70)":       rsi_signal(prices.close),
        "MACD 12/26/9":         macd_signal(prices.close),
        "Trend-filter + MACD":  trend_filtered_macd(prices.close),
        "Price>MA200 + MACD":   price_above_ma_with_macd(prices.close),
    }
    for name, mask in ts_strategies.items():
        weights = equal_weight_when_long(mask)
        result = run_backtest(rets, weights, cost_bps=args.cost_bps)
        rows.append((name, report.metrics(result.returns)))

    # --- Benchmark ---
    bh = rets.mean(axis=1).dropna()
    rows.append(("Buy & hold (EW)", report.metrics(bh)))

    _print_table(rows)


def _print_table(rows: list[tuple[str, "report.Metrics"]]) -> None:
    headers = ["Strategy", "CAGR", "Vol", "Sharpe", "Sortino", "MDD", "Calmar", "Hit"]
    widths = [24, 8, 8, 8, 8, 9, 8, 8]

    line = " ".join(h.ljust(w) for h, w in zip(headers, widths))
    print()
    print(line)
    print("-" * len(line))
    for name, m in rows:
        cells = [
            name.ljust(widths[0]),
            f"{m.cagr * 100:>6.2f}%".ljust(widths[1]),
            f"{m.annualized_volatility * 100:>6.2f}%".ljust(widths[2]),
            f"{m.sharpe:>6.2f}".ljust(widths[3]),
            f"{m.sortino:>6.2f}".ljust(widths[4]),
            f"{m.max_drawdown * 100:>7.2f}%".ljust(widths[5]),
            f"{m.calmar:>6.2f}".ljust(widths[6]),
            f"{m.hit_rate * 100:>5.1f}%".ljust(widths[7]),
        ]
        print(" ".join(cells))


if __name__ == "__main__":
    main()
