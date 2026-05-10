# quant

Personal quantitative investment program. US equity factor research +
backtesting + Alpaca paper trading.

Tracking lives in the [assistant-hub `quant` workspace](https://github.com/ToySin/assisthub-ws-quant)
and the linked Notion page (Quant Investment → Strategies / Research Notes).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For Alpaca paper trading, add to `~/repositories/assisthub-ws-quant/.env`:
```
ALPACA_PAPER_KEY=PK...
ALPACA_PAPER_SECRET=...
```
(Generate at [alpaca.markets](https://alpaca.markets/) → Paper Trading → API Keys.)

## Quickstart

```bash
# Tests (no network)
pytest

# Backtests
python -m scripts.run_momentum                                # 12-1 momentum on liquid_30
python -m scripts.run_signal_compare --start 2020-01-01       # MA cross / RSI / MACD
python -m scripts.run_all_compare --universe sp500 \
    --listed-before 2010-01-01                                # everything side by side
python -m scripts.run_multifactor                             # momentum + value + quality

# Validation
python -m scripts.run_oos                                     # train/test split
python -m scripts.run_walk_forward                            # 5y rolling, plots in outputs/

# Live screening (real-time, no lookahead)
python -m scripts.screen_today --universe sp500 --top-n 15

# Paper trading
python -m scripts.check_alpaca                                # connection sanity check
python -m scripts.paper_rebalance                             # dry run — print plan only
python -m scripts.paper_rebalance --execute                   # send orders

# Monitoring + automation (24/7 machine — see deploy/README.md)
python -m scripts.daily_monitor --no-post                     # local snapshot
python -m scripts.daily_monitor                               # snapshot → Discord
python -m scripts.auto_monthly_rebalance --dry-run --force    # test auto-rebalance
```

## Layout

```
quant/
  data.py              yfinance OHLCV + parquet cache + listing-date filter
  universe.py          liquid_30, sp500 (Wikipedia)
  cache.py             cache dir resolution
  factors/
    momentum.py        12-1 (Jegadeesh-Titman) and configurable lookback
    volatility.py      realized vol + inverse-vol score
    value.py           1/P/B + 1/P/E z-blend
    quality.py         ROE + low-leverage + margin z-blend
  signals/
    trend.py           MA cross 50/200
    oscillator.py      RSI(14, Wilder), MACD(12/26/9)
    combo.py           MA + MACD overlays
  fundamentals.py      yfinance.info snapshot (⚠️ not point-in-time)
  portfolio.py         rank → top decile → equal weight, monthly rebalance
  positions.py         signal mask → target weights
  backtest.py          target-weight × forward-return + L1 turnover cost
  report.py            CAGR / Sharpe / Sortino / MDD / Calmar / hit rate
  score.py             multi-factor blend (cross-sectional z-score)
  broker.py            Alpaca paper-trading wrapper (paper-default, double-gated for live)
  notify.py            Discord webhook client
  state.py             Persistent JSON state for daily/monthly automation
  filters.py           Sanity filters (e.g. drop extreme-momentum data errors)

scripts/
  run_momentum.py            single-factor demo
  run_signal_compare.py      MA / RSI / MACD comparison
  run_all_compare.py         factors + signals side by side
  run_multifactor.py         momentum + value + quality blend
  run_oos.py                 in-sample / out-of-sample split
  run_walk_forward.py        rolling 5y windows + 3 PNG plots → outputs/plots/
  screen_today.py            top-N today's tickers per factor
  check_alpaca.py            paper account snapshot
  paper_rebalance.py         momentum 12-1 monthly rebalance via paper trading
  daily_monitor.py           equity / PnL snapshot → Discord
  auto_monthly_rebalance.py  cron-friendly auto-rebalance with state-tracked idempotency

tests/                       62 pytest cases, all on synthetic data (no network)
data/                        gitignored — parquet cache, intermediate outputs
outputs/                     committed — plots that get embedded in Notion
```

## Validation arc

The `run_oos` and `run_walk_forward` scripts together validate that the
12-1 momentum factor has real alpha:

- **OOS test**: train (2010-2018) Sharpe 0.89 → test (2019-2026) Sharpe 1.09. Survives the regime change.
- **Walk-forward**: 12 of 12 calendar years (2015-2026) had positive momentum Sharpe. Notably 2018 (mom +0.17 vs BH -0.09) and 2022 (mom +0.24 vs BH -0.22) — momentum protected capital in bear markets.
- **Parameter tuning**: scanning train winners *underperforms* canonical 12-1 on test by 0.08 Sharpe. Stick with canonical specs.

Plots: `outputs/plots/per_year_sharpe.png`, `equity_curves.png`, `walk_forward_picks.png`.

## Adding a factor

Each factor is a function `(close: DataFrame) -> DataFrame` of cross-sectional
scores (higher = stronger exposure). Drop it in `quant/factors/<name>.py`,
re-export from `factors/__init__.py`, add a smoke test, and pass it to
`top_decile_long_only`.

## Roadmap

- [ ] Corporate-action filter (drop tickers with extreme momentum > 200%, e.g. SNDK spinoff)
- [ ] Point-in-time fundamentals (Sharadar / Norgate, or yfinance quarterly financials with announcement-date lag)
- [ ] Walk-forward parameter selection refinement
- [ ] Daily monitoring script + email/Slack alerts
- [ ] Multi-strategy capital allocation across momentum / quality / value
