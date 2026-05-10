"""Out-of-sample validation of 12-1 momentum on filtered S&P 500.

Splits the 2010-2026 window into train (2010-2018) and test (2019-2026).
Reports two checks designed to surface different validation concerns.

Check A — Fixed-parameter consistency
  Run canonical 12-1 momentum (top-10% long-only, monthly rebalance) on
  each period without touching the parameters. If train and test
  Sharpe are similar, the apparent alpha is robust across regimes. If
  test is dramatically worse, the train-period result was period-
  specific (e.g. a particular market regime that didn't repeat).

Check B — Parameter-tuned shrinkage
  Scan momentum lookbacks {3-1, 6-1, 9-1, 12-1} × top-pct {5%, 10%,
  20%, 30%} on the train period only. Pick the highest-Sharpe combo
  on train, then apply that combo to test. The drop from train Sharpe
  to test Sharpe is the data-snooping cost of having tuned. Big drop
  = the train winner was overfit.

Usage:
    python -m scripts.run_oos
    python -m scripts.run_oos --split 2020-01-01
"""

from __future__ import annotations

import argparse

import pandas as pd

from quant import data, report, universe
from quant.backtest import run as run_backtest
from quant.factors import momentum_lookback
from quant.portfolio import top_decile_long_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("liquid30", "sp500"),
                        default="sp500")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--split", default="2019-01-01",
                        help="Train ends, test starts on this date")
    parser.add_argument("--listed-before", default="2010-01-01")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    tickers = universe.sp500() if args.universe == "sp500" else universe.liquid_30()
    print(f"[oos] universe={args.universe} ({len(tickers)} tickers)")
    print("[oos] downloading bars (cached)...")
    prices = data.load(tickers, start=args.start, end=args.end)
    if args.listed_before:
        prices = data.filter_by_first_bar(prices, listed_before=args.listed_before)
    print(f"[oos] panel: {prices.close.shape[0]} days "
          f"x {prices.close.shape[1]} tickers")

    train_start = pd.Timestamp(args.start)
    test_start = pd.Timestamp(args.split)
    train_end = test_start - pd.Timedelta(days=1)
    test_end = prices.close.index[-1]
    print(f"[oos] train: [{train_start.date()}, {train_end.date()}]")
    print(f"[oos] test:  [{test_start.date()}, {test_end.date()}]")

    rets = prices.returns()
    bh = rets.mean(axis=1)

    # ============================================================
    # Check A — Fixed 12-1, no peeking at test
    # ============================================================
    print("\n## Check A — Fixed 12-1 momentum (top-10%) consistency\n")
    fixed = _run_strategy(prices, lookback=231, skip=21, top_pct=0.10,
                          cost_bps=args.cost_bps)
    train_a = report.metrics(fixed.loc[train_start:train_end].dropna())
    test_a = report.metrics(fixed.loc[test_start:test_end].dropna())
    bh_train = report.metrics(bh.loc[train_start:train_end].dropna())
    bh_test = report.metrics(bh.loc[test_start:test_end].dropna())

    _print_table([
        ("Train  Mom 12-1 top-10%", train_a),
        ("Train  Buy & hold (EW)", bh_train),
        ("Test   Mom 12-1 top-10%", test_a),
        ("Test   Buy & hold (EW)", bh_test),
    ])

    train_alpha = train_a.sharpe - bh_train.sharpe
    test_alpha = test_a.sharpe - bh_test.sharpe
    print(f"\n  Train Sharpe alpha vs BH: {train_alpha:+.2f}")
    print(f"  Test  Sharpe alpha vs BH: {test_alpha:+.2f}")
    print(f"  Sharpe shrinkage train→test: {train_a.sharpe - test_a.sharpe:+.2f}")

    # ============================================================
    # Check B — Parameter scan, train-only tuning, OOS shrinkage
    # ============================================================
    print("\n## Check B — Parameter-tuned shrinkage\n")
    lookbacks = [42, 105, 168, 231]   # roughly 3-1, 6-1, 9-1, 12-1
    top_pcts = [0.05, 0.10, 0.20, 0.30]

    scan_rows = []
    for lb in lookbacks:
        for tp in top_pcts:
            full = _run_strategy(prices, lookback=lb, skip=21, top_pct=tp,
                                 cost_bps=args.cost_bps)
            train_m = report.metrics(full.loc[train_start:train_end].dropna())
            scan_rows.append({
                "lookback": lb,
                "label": _label_for(lb),
                "top_pct": tp,
                "train_sharpe": train_m.sharpe,
                "train_cagr": train_m.cagr,
                "train_mdd": train_m.max_drawdown,
            })

    scan_df = pd.DataFrame(scan_rows).sort_values("train_sharpe", ascending=False)
    print("  Train scan (sorted by Sharpe):")
    print("  " + scan_df.to_string(index=False).replace("\n", "\n  "))

    best = scan_df.iloc[0]
    print(f"\n  Train winner: {best['label']} momentum, top-{int(best['top_pct'] * 100)}% "
          f"(Sharpe {best['train_sharpe']:.2f})")

    full_best = _run_strategy(prices, lookback=int(best["lookback"]), skip=21,
                              top_pct=float(best["top_pct"]),
                              cost_bps=args.cost_bps)
    train_best = report.metrics(full_best.loc[train_start:train_end].dropna())
    test_best = report.metrics(full_best.loc[test_start:test_end].dropna())

    print(f"\n  Applied train-winner to test:")
    _print_table([
        (f"Train  {best['label']} top-{int(best['top_pct']*100)}%", train_best),
        (f"Test   {best['label']} top-{int(best['top_pct']*100)}%", test_best),
    ])
    snooping_cost = train_best.sharpe - test_best.sharpe
    print(f"\n  Data-snooping shrinkage: train Sharpe {train_best.sharpe:.2f} "
          f"→ test Sharpe {test_best.sharpe:.2f} ({snooping_cost:+.2f})")

    # Comparison
    print("\n## Verdict\n")
    print(f"  Fixed 12-1 test Sharpe:         {test_a.sharpe:.2f}")
    print(f"  Train-tuned best test Sharpe:   {test_best.sharpe:.2f}")
    if test_best.sharpe > test_a.sharpe + 0.05:
        print("  → Parameter tuning meaningfully helped on test (real signal in the scan)")
    elif test_best.sharpe < test_a.sharpe - 0.05:
        print("  → Parameter tuning HURT on test (data-snooping was overfitting)")
    else:
        print("  → Parameter tuning was a wash (canonical 12-1 is fine)")

    if test_a.sharpe > 0.7 and test_alpha > 0.0:
        print("  → 12-1 momentum survived OOS with positive alpha vs buy-and-hold ✓")
    elif test_a.sharpe > 0.0 and test_alpha < 0.0:
        print("  → 12-1 momentum has positive Sharpe but lost to buy-and-hold OOS")
    else:
        print("  → 12-1 momentum did NOT generalize — train alpha was period-specific ✗")


def _run_strategy(prices, *, lookback: int, skip: int, top_pct: float,
                  cost_bps: float) -> pd.Series:
    """Run the strategy on the full panel; caller slices return series."""
    scores = momentum_lookback(prices.close, lookback=lookback, skip=skip)
    weights = top_decile_long_only(scores, top_pct=top_pct)
    result = run_backtest(prices.returns(), weights, cost_bps=cost_bps)
    return result.returns


def _label_for(lookback: int) -> str:
    months_back = (lookback + 21) // 21
    return f"{months_back}-1"


def _print_table(rows) -> None:
    print(f"  {'Period / Strategy':<35} {'CAGR':>7} {'Sharpe':>7} {'MDD':>8}")
    print("  " + "-" * 60)
    for name, m in rows:
        print(f"  {name:<35} {m.cagr * 100:>6.2f}% {m.sharpe:>7.2f} "
              f"{m.max_drawdown * 100:>7.2f}%")


if __name__ == "__main__":
    main()
