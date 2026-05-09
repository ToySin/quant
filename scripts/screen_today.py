"""Today's top-N screener — momentum / quality / value / blend.

Backtest results are theater (lookahead bias inflates Quality, Value
is empirically dead post-2010 for US large cap). Where the same
factors *can* still help is as a starting point for human review
RIGHT NOW: "given today's prices and today's fundamentals, which
stocks rank highest on each dimension?"

This script is that. Zero claims about future returns. It just
computes scores and prints lists.

Usage:
    python -m scripts.screen_today
    python -m scripts.screen_today --universe sp500 --top-n 20
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from quant import data, fundamentals, universe
from quant.factors import momentum_12_1, quality, value
from quant.score import blend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("liquid30", "sp500"),
                        default="sp500")
    parser.add_argument("--top-n", type=int, default=15,
                        help="How many tickers to print per factor (default 15)")
    parser.add_argument("--listed-before", default=None,
                        help="Optional history filter (rarely useful for "
                             "screener — current SPX members are all live)")
    args = parser.parse_args()

    tickers = universe.sp500() if args.universe == "sp500" else universe.liquid_30()
    print(f"# Today's Screener — {datetime.now().date().isoformat()}")
    print(f"# Universe: {args.universe} ({len(tickers)} tickers)")

    # Need ~14 months of price history for 12-1 momentum + warmup
    print(f"\n[screener] downloading bars (cached)...")
    prices = data.load(tickers, start="2024-01-01")
    if args.listed_before:
        prices = data.filter_by_first_bar(prices, listed_before=args.listed_before)
    print(f"[screener] panel: {prices.close.shape[0]} days "
          f"x {prices.close.shape[1]} tickers")

    print("[screener] pulling fundamentals (cached)...")
    funds = fundamentals.load(list(prices.close.columns))

    # Today's scores
    mom_panel = momentum_12_1(prices.close)
    today_mom = mom_panel.iloc[-1].dropna()

    val_score = value.composite(funds)
    qual_score = quality.composite(funds)

    # Blend uses panel form
    val_panel = value.as_panel(val_score, prices.close.index)
    qual_panel = value.as_panel(qual_score, prices.close.index)
    blend_panel = blend({
        "momentum": mom_panel,
        "value": val_panel,
        "quality": qual_panel,
    })
    today_blend = blend_panel.iloc[-1].dropna()

    print(f"\n## Top {args.top_n} — 12-1 Momentum (price-based, no lookahead)")
    print("(Recently strong on a 12-month basis, excluding the last month)")
    _print_factor_table(today_mom.nlargest(args.top_n), funds, mom=today_mom,
                        val=val_score, qual=qual_score)

    print(f"\n## Top {args.top_n} — Quality (⚠️ uses today's fundamentals)")
    print("(High ROE, low debt, high margins — current snapshot)")
    _print_factor_table(qual_score.dropna().nlargest(args.top_n), funds,
                        mom=today_mom, val=val_score, qual=qual_score)

    print(f"\n## Top {args.top_n} — Value (⚠️ uses today's fundamentals)")
    print("(Cheap on P/B and P/E — current snapshot)")
    _print_factor_table(val_score.dropna().nlargest(args.top_n), funds,
                        mom=today_mom, val=val_score, qual=qual_score)

    print(f"\n## Top {args.top_n} — Blend (Momentum + Value + Quality, equal-weight z)")
    print("(Composite ranking — best all-around on the three dimensions)")
    _print_factor_table(today_blend.nlargest(args.top_n), funds,
                        mom=today_mom, val=val_score, qual=qual_score)


def _print_factor_table(top: pd.Series, funds: fundamentals.Fundamentals, *,
                         mom: pd.Series, val: pd.Series, qual: pd.Series) -> None:
    print(f"  {'Rank':<5} {'Ticker':<7} {'P/B':>7} {'P/E':>7} {'ROE':>7} "
          f"{'Mom':>7} {'Val':>7} {'Qual':>7}")
    print("  " + "-" * 58)
    for i, (ticker, _) in enumerate(top.items(), 1):
        row = funds.df.loc[ticker] if ticker in funds.df.index else None
        pb = row["priceToBook"] if row is not None else None
        pe = row["trailingPE"] if row is not None else None
        roe = row["returnOnEquity"] if row is not None else None
        m = mom.get(ticker)
        v = val.get(ticker)
        q = qual.get(ticker)
        print(f"  {i:<5} {ticker:<7} "
              f"{_fmt_num(pb):>7} {_fmt_num(pe):>7} "
              f"{_fmt_pct(roe):>7} "
              f"{_fmt_pct(m):>7} {_fmt_num(v):>7} {_fmt_num(q):>7}")


def _fmt_num(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.2f}"


def _fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:.1f}%"


if __name__ == "__main__":
    main()
