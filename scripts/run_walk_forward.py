"""Walk-forward analysis of 12-1 momentum on filtered S&P 500.

Two analyses:

A) Per-year stability (canonical 12-1, no tuning)
   For each calendar year in the test panel, compute that year's
   strategy return series, report Sharpe / CAGR / MDD vs equal-weight
   buy-and-hold. Tests whether alpha is distributed across time or
   concentrated in a few standout years.

B) Walk-forward parameter selection
   For each year t, scan momentum parameters on the prior 5 years
   (train), pick the highest-Sharpe combo, apply it to year t alone
   (test). Document the chosen parameter per year. Stable winners
   suggest robust signal; jumping winners suggest noise.

Outputs three PNGs to data/plots/:
  - per_year_sharpe.png       (bar chart)
  - equity_curves.png         (cumulative returns)
  - walk_forward_picks.png    (per-year selected parameters)

Run after run_oos.py for the full validation arc.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant import data, report, universe
from quant.backtest import run as run_backtest
from quant.factors import momentum_lookback
from quant.portfolio import top_decile_long_only
from quant.cache import cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("liquid30", "sp500"),
                        default="sp500")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--listed-before", default="2010-01-01")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--analysis-start-year", type=int, default=2015,
                        help="First year to analyze (need 5y train history)")
    args = parser.parse_args()

    tickers = universe.sp500() if args.universe == "sp500" else universe.liquid_30()
    print(f"[wf] universe={args.universe} ({len(tickers)} tickers)")
    print("[wf] downloading bars (cached)...")
    prices = data.load(tickers, start=args.start, end=args.end)
    if args.listed_before:
        prices = data.filter_by_first_bar(prices, listed_before=args.listed_before)
    print(f"[wf] panel: {prices.close.shape[0]} days "
          f"x {prices.close.shape[1]} tickers")

    rets = prices.returns()
    bh = rets.mean(axis=1)

    # Canonical 12-1, full panel
    fixed = _run_strategy(prices, lookback=231, skip=21, top_pct=0.10,
                          cost_bps=args.cost_bps)

    plot_dir = _plot_dir()

    # ============================================================
    # Analysis A: per-year metrics
    # ============================================================
    print("\n## Per-year metrics (canonical 12-1)\n")
    years = sorted({d.year for d in fixed.index
                    if d.year >= args.analysis_start_year})
    rows = []
    for year in years:
        mask = fixed.index.year == year
        if mask.sum() < 100:        # partial year (e.g., 2026 YTD)
            label = f"{year} (YTD)"
        else:
            label = str(year)
        mom_y = fixed[mask].dropna()
        bh_y = bh[mask].dropna()
        if mom_y.empty or bh_y.empty:
            continue
        m_mom = report.metrics(mom_y)
        m_bh = report.metrics(bh_y)
        rows.append({
            "year": year,
            "label": label,
            "mom_cagr": m_mom.cagr, "mom_sharpe": m_mom.sharpe, "mom_mdd": m_mom.max_drawdown,
            "bh_cagr": m_bh.cagr, "bh_sharpe": m_bh.sharpe, "bh_mdd": m_bh.max_drawdown,
        })

    yearly = pd.DataFrame(rows)
    print(yearly[["label", "mom_cagr", "mom_sharpe", "mom_mdd",
                  "bh_cagr", "bh_sharpe", "bh_mdd"]]
          .to_string(index=False, float_format=lambda x: f"{x*100:.1f}%"
                     if abs(x) < 5 else f"{x:.2f}"))

    n_pos = int((yearly["mom_sharpe"] > 0).sum())
    n_outperf = int((yearly["mom_sharpe"] > yearly["bh_sharpe"]).sum())
    print(f"\n  Years with positive momentum Sharpe: {n_pos}/{len(yearly)}")
    print(f"  Years momentum beat BH Sharpe:        {n_outperf}/{len(yearly)}")

    _plot_per_year_sharpe(yearly, plot_dir / "per_year_sharpe.png")
    _plot_equity_curves(fixed, bh, plot_dir / "equity_curves.png")

    # ============================================================
    # Analysis B: walk-forward parameter selection
    # ============================================================
    print("\n## Walk-forward parameter selection\n")
    print("  For each year, scan params on prior 5 years, pick best Sharpe,")
    print("  apply to that year. Stable winner = robust signal.\n")

    lookbacks = [42, 105, 168, 231]   # 3-1, 6-1, 9-1, 12-1
    top_pcts = [0.05, 0.10, 0.20, 0.30]

    picks = []
    for year in years:
        train_start = pd.Timestamp(f"{year - 5}-01-01")
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")

        # Skip years where train history isn't fully available
        if train_start < pd.Timestamp(args.start):
            continue

        best = None
        for lb in lookbacks:
            for tp in top_pcts:
                full = _run_strategy(prices, lookback=lb, skip=21, top_pct=tp,
                                     cost_bps=args.cost_bps)
                train_ser = full.loc[train_start:train_end].dropna()
                if len(train_ser) < 200:
                    continue
                m = report.metrics(train_ser)
                if best is None or m.sharpe > best["train_sharpe"]:
                    best = {
                        "year": year, "lookback": lb, "top_pct": tp,
                        "label": _label_for(lb),
                        "train_sharpe": m.sharpe,
                    }
        if best is None:
            continue

        # Test on year t
        full_best = _run_strategy(prices, lookback=best["lookback"], skip=21,
                                  top_pct=best["top_pct"],
                                  cost_bps=args.cost_bps)
        test_ser = full_best.loc[test_start:test_end].dropna()
        if test_ser.empty:
            continue
        m_test = report.metrics(test_ser)
        best["test_sharpe"] = m_test.sharpe
        best["test_cagr"] = m_test.cagr
        picks.append(best)

    wf = pd.DataFrame(picks)
    print(wf[["year", "label", "top_pct", "train_sharpe", "test_sharpe", "test_cagr"]]
          .to_string(index=False))

    canonical_count = int((wf["label"] == "12-1").sum())
    print(f"\n  Canonical (12-1) picked as winner: {canonical_count}/{len(wf)} years")
    print(f"  Mean train Sharpe (best per year):  {wf['train_sharpe'].mean():.2f}")
    print(f"  Mean test Sharpe (after pick):      {wf['test_sharpe'].mean():.2f}")
    print(f"  Mean shrinkage train→test:          "
          f"{(wf['train_sharpe'] - wf['test_sharpe']).mean():+.2f}")

    _plot_walk_forward_picks(wf, plot_dir / "walk_forward_picks.png")

    print(f"\n[wf] plots saved to {plot_dir}/")
    print("       - per_year_sharpe.png")
    print("       - equity_curves.png")
    print("       - walk_forward_picks.png")


def _run_strategy(prices, *, lookback: int, skip: int, top_pct: float,
                  cost_bps: float) -> pd.Series:
    scores = momentum_lookback(prices.close, lookback=lookback, skip=skip)
    weights = top_decile_long_only(scores, top_pct=top_pct)
    result = run_backtest(prices.returns(), weights, cost_bps=cost_bps)
    return result.returns


def _label_for(lookback: int) -> str:
    months = (lookback + 21) // 21
    return f"{months}-1"


def _plot_dir() -> Path:
    p = cache_dir().parent / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _plot_per_year_sharpe(yearly: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(yearly))
    width = 0.4

    ax.bar(x - width / 2, yearly["mom_sharpe"], width=width,
           label="12-1 Momentum top-10%", color="#1f77b4")
    ax.bar(x + width / 2, yearly["bh_sharpe"], width=width,
           label="Buy & hold (EW)", color="#888888")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(yearly["label"], rotation=45, ha="right")
    ax.set_ylabel("Annual Sharpe")
    ax.set_title("Per-year Sharpe — Momentum vs Buy & hold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_equity_curves(mom: pd.Series, bh: pd.Series, path: Path) -> None:
    aligned = pd.DataFrame({"momentum": mom, "buy_hold": bh}).dropna()
    eq_mom = (1 + aligned["momentum"]).cumprod()
    eq_bh = (1 + aligned["buy_hold"]).cumprod()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(eq_mom.index, eq_mom, label="12-1 Momentum top-10%", color="#1f77b4")
    ax.plot(eq_bh.index, eq_bh, label="Buy & hold (EW)", color="#888888")
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative return (log scale, $1 start)")
    ax.set_title("Equity curves — full window")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_walk_forward_picks(wf: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    label_to_color = {
        "3-1": "#d62728",   # red
        "6-1": "#ff7f0e",   # orange
        "9-1": "#2ca02c",   # green
        "12-1": "#1f77b4",  # blue
    }
    bar_colors = [label_to_color.get(lbl, "#888") for lbl in wf["label"]]

    # Top: train vs test Sharpe
    x = np.arange(len(wf))
    width = 0.4
    ax1.bar(x - width / 2, wf["train_sharpe"], width=width,
            label="Train Sharpe (best of scan)", color="#aaaaaa")
    ax1.bar(x + width / 2, wf["test_sharpe"], width=width,
            label="Test Sharpe (winner applied)", color=bar_colors)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Sharpe ratio")
    ax1.set_title("Walk-forward — train winner vs out-of-sample test")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Bottom: which lookback picked, color-coded
    for label, color in label_to_color.items():
        mask = wf["label"] == label
        if mask.any():
            ax2.scatter(np.array(x)[mask], wf.loc[mask, "top_pct"],
                        s=140, c=color, label=label, edgecolors="black",
                        linewidths=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(wf["year"], rotation=0)
    ax2.set_ylabel("top-pct picked")
    ax2.set_xlabel("Year (test)")
    ax2.set_title("Per-year parameter pick (color = lookback)")
    ax2.legend(title="Lookback")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
