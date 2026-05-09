"""Multi-factor backtest — momentum + value + quality.

Compares:
  - Single 12-1 momentum (baseline)
  - Single value
  - Single quality
  - Equal-weight blend of all three (z-scored, cross-sectional)
  - Buy & hold benchmark

Note: value and quality come from yfinance.info and are *current
snapshot only* — not point-in-time. Treat results as illustrative;
real research needs Sharadar / Norgate.

Usage:
    python -m scripts.run_multifactor
    python -m scripts.run_multifactor --universe sp500 --start 2015-01-01
"""

from __future__ import annotations

import argparse

from quant import data, fundamentals, report, universe
from quant.backtest import run as run_backtest
from quant.factors import momentum_12_1, quality, value
from quant.portfolio import top_decile_long_only
from quant.score import blend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("liquid30", "sp500"),
                        default="liquid30")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--top-pct", type=float, default=0.1)
    parser.add_argument("--listed-before", default=None)
    args = parser.parse_args()

    tickers = universe.sp500() if args.universe == "sp500" else universe.liquid_30()
    print(f"[mfactor] universe={args.universe} ({len(tickers)} tickers)")
    print(f"[mfactor] window=[{args.start}, {args.end or 'latest'}]")

    print("[mfactor] downloading bars...")
    prices = data.load(tickers, start=args.start, end=args.end)
    if args.listed_before:
        prices = data.filter_by_first_bar(prices, listed_before=args.listed_before)

    panel_tickers = list(prices.close.columns)
    print(f"[mfactor] panel: {prices.close.shape[0]} days "
          f"x {len(panel_tickers)} tickers")

    print("[mfactor] pulling fundamentals (slow first run, cached after)...")
    funds = fundamentals.load(panel_tickers)

    rets = prices.returns()
    dates = prices.close.index

    momentum_panel = momentum_12_1(prices.close)
    value_panel = value.as_panel(value.composite(funds), dates)
    quality_panel = value.as_panel(quality.composite(funds), dates)

    factor_panels = {
        "momentum": momentum_panel,
        "value": value_panel,
        "quality": quality_panel,
    }
    composite_panel = blend(factor_panels)

    rows = []
    for name, scores in [
        (f"Momentum 12-1 (top-{int(args.top_pct * 100)}%)", momentum_panel),
        (f"Value only (top-{int(args.top_pct * 100)}%)", value_panel),
        (f"Quality only (top-{int(args.top_pct * 100)}%)", quality_panel),
        (f"Mom + Val + Qual (top-{int(args.top_pct * 100)}%)", composite_panel),
    ]:
        weights = top_decile_long_only(scores, top_pct=args.top_pct)
        result = run_backtest(rets, weights, cost_bps=args.cost_bps)
        rows.append((name, report.metrics(result.returns)))

    bh = rets.mean(axis=1).dropna()
    rows.append(("Buy & hold (EW)", report.metrics(bh)))

    _print_table(rows)


def _print_table(rows) -> None:
    headers = ["Strategy", "CAGR", "Vol", "Sharpe", "Sortino", "MDD", "Calmar", "Hit"]
    widths = [38, 8, 8, 8, 8, 9, 8, 8]

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
