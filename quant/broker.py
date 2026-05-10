"""Alpaca broker wrapper.

Thin layer around alpaca-py's TradingClient. Two operations matter
for our momentum rebalance:

  - snapshot(): equity, cash, current positions
  - submit_notional_order(): buy/sell a dollar amount

We use *notional* (dollar-amount) orders rather than share orders
so fractional shares are handled automatically by Alpaca and we
don't have to round-trip through current price.

Safety: defaults to paper endpoint. To trade live, the caller must
pass `paper=False` explicitly AND set `live_trading_consent=True`.
The double-gate is deliberate — making the live switch ergonomically
inconvenient is the simplest defense against accidental real money.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


@dataclass(frozen=True)
class Position:
    ticker: str
    qty: float           # signed (negative = short, but paper accounts default long-only)
    market_value: float
    avg_entry_price: float
    unrealized_pl: float


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    positions: list[Position] = field(default_factory=list)

    def position_map(self) -> dict[str, Position]:
        return {p.ticker: p for p in self.positions}


class AlpacaBroker:
    """Wrapper that defaults to paper trading.

    Construct with explicit credentials, or use `from_env()` to pull
    them from `ALPACA_PAPER_KEY` / `ALPACA_PAPER_SECRET`.
    """

    def __init__(self, key: str, secret: str, *, paper: bool = True,
                 live_trading_consent: bool = False) -> None:
        if not paper and not live_trading_consent:
            raise RuntimeError(
                "Live trading requires paper=False AND live_trading_consent=True. "
                "If you really want this, pass both explicitly."
            )
        self._paper = paper
        self._client = TradingClient(api_key=key, secret_key=secret, paper=paper)

    @classmethod
    def from_env(cls) -> "AlpacaBroker":
        key = os.environ.get("ALPACA_PAPER_KEY")
        secret = os.environ.get("ALPACA_PAPER_SECRET")
        if not key or not secret:
            raise RuntimeError(
                "Set ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET in the workspace .env"
            )
        return cls(key=key, secret=secret, paper=True)

    @property
    def is_paper(self) -> bool:
        return self._paper

    def snapshot(self) -> AccountSnapshot:
        account = self._client.get_account()
        raw_positions = self._client.get_all_positions()
        positions = [
            Position(
                ticker=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                unrealized_pl=float(p.unrealized_pl),
            )
            for p in raw_positions
        ]
        return AccountSnapshot(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            positions=positions,
        )

    def submit_notional_order(self, ticker: str, *, dollars: float,
                              side: str) -> str:
        """Submit a market order denominated in dollars (not shares).

        Returns the Alpaca order ID. Alpaca accepts notional orders
        for most US equities; fractional shares are returned for
        non-divisible amounts. Time in force is 'day' so anything not
        filled by close is cancelled.
        """
        if dollars <= 0:
            raise ValueError(f"dollars must be > 0, got {dollars}")
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        req = MarketOrderRequest(
            symbol=ticker,
            notional=round(dollars, 2),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        return str(order.id)

    def close_position(self, ticker: str) -> str:
        """Liquidate the entire position in `ticker`.

        Faster than computing the exact dollar amount yourself when
        you just want out — Alpaca handles the share count internally.
        """
        order = self._client.close_position(ticker)
        return str(order.id)

    def cancel_all_orders(self) -> int:
        """Cancel all open (unfilled) orders. Returns count cancelled."""
        cancelled = self._client.cancel_orders()
        return len(cancelled or [])
