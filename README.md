# quant

Personal quantitative investment program. US equity factor research +
backtesting toolkit, pure pandas, no broker.

Tracking lives in [assistant-hub `quant` workspace](https://github.com/ToySin/assisthub-ws-quant)
and the linked Notion page (Quant Investment → Strategies / Research Notes).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Quickstart

```bash
# Run pytest smoke tests (no network)
pytest

# Run end-to-end momentum backtest on a 30-name universe (hits yfinance once,
# caches to data/cache/, then is instant)
python -m scripts.run_momentum

# Run on full S&P 500
python -m scripts.run_momentum --universe sp500 --start 2015-01-01
```

## Layout

```
quant/
  data.py              # yfinance OHLCV loader + parquet cache
  universe.py          # ticker lists (liquid_30, sp500 from Wikipedia)
  cache.py             # cache dir resolution
  factors/
    momentum.py        # n-1 lookback, classic 12-1
    volatility.py      # realized vol, inverse-vol score
  portfolio.py         # rank → top decile → monthly rebalance, long-only or long-short
  backtest.py          # apply weights to returns with t-cost
  report.py            # CAGR / Sharpe / Sortino / MDD / Calmar / hit rate
scripts/
  run_momentum.py      # end-to-end demo
tests/                 # pytest, all-synthetic fixtures
data/cache/            # parquet cache (gitignored)
```

## Adding a factor

Each factor is just a function `(close: DataFrame) -> DataFrame` of
cross-sectional scores (higher = stronger exposure). Drop it in
`quant/factors/<name>.py`, re-export from `factors/__init__.py`,
add a smoke test, and pass it to `top_decile_long_only`.

## Roadmap

- [ ] Add value (book/price) and quality (ROE) factors — needs fundamentals beyond yfinance
- [ ] Multi-factor combination + IR comparison vs. single factor
- [ ] Replace pure-pandas backtest with vectorbt for parameter sweeps
- [ ] Alpaca paper trading hook for forward testing
