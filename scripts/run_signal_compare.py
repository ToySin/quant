"""Compare technical signal strategies side by side.

Runs the same liquid_30 universe through:
  - MA-cross (50/200, golden cross / death cross)
  - RSI mean-reversion (14, 30/70 thresholds)
  - MACD trend (12/26/9)
  - Buy & hold (benchmark)

Each signal is converted to equal-weight-across-currently-long
positions via `quant.positions.equal_weight_when_long`. Output is a
side-by-side metrics table.

Usage:
    python -m scripts.run_signal_compare
    python -m scripts.run_signal_compare --start 2018-01-01 --cost-bps 10
"""

from __future__ import annotations

import argparse

import pandas as pd

from quant import data, report, universe
from quant.backtest import run as run_backtest
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
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    tickers = universe.liquid_30()
    print(f"[signals] universe=liquid30 ({len(tickers)} tickers)")
    print(f"[signals] window=[{args.start}, {args.end or 'latest'}]")
    print("[signals] downloading bars (cached)...")
    prices = data.load(tickers, start=args.start, end=args.end)

    rets = prices.returns()

    strategies = {
        "MA cross 50/200":      ma_cross_signal(prices.close, fast=50, slow=200),
        "RSI 14 (30/70)":       rsi_signal(prices.close, window=14, lower=30, upper=70),
        "MACD 12/26/9":         macd_signal(prices.close),
        "Trend-filter + MACD":  trend_filtered_macd(prices.close),
        "Price>MA200 + MACD":   price_above_ma_with_macd(prices.close),
    }

    rows = []
    for name, mask in strategies.items():
        weights = equal_weight_when_long(mask)
        result = run_backtest(rets, weights, cost_bps=args.cost_bps)
        rows.append((name, report.metrics(result.returns)))

    bh = rets.mean(axis=1).dropna()
    rows.append(("Buy & hold (EW)", report.metrics(bh)))

    _print_table(rows)


def _print_table(rows: list[tuple[str, "report.Metrics"]]) -> None:
    headers = ["Strategy", "CAGR", "Vol", "Sharpe", "Sortino", "MDD", "Calmar", "Hit"]
    widths = [22, 8, 8, 8, 8, 9, 8, 8]

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
