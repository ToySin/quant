"""Broker wrapper tests — no real network calls.

We mock alpaca-py's TradingClient so the unit tests cover the
wrapper's argument plumbing and safety gates without needing real
credentials.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant.broker import AlpacaBroker


def _make_mock_account(equity=100_000, cash=50_000, buying_power=200_000):
    account = MagicMock()
    account.equity = equity
    account.cash = cash
    account.buying_power = buying_power
    return account


def _make_mock_position(symbol, qty=1.0, market_value=100.0,
                       avg_entry_price=99.0, unrealized_pl=1.0):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = qty
    pos.market_value = market_value
    pos.avg_entry_price = avg_entry_price
    pos.unrealized_pl = unrealized_pl
    return pos


@patch("quant.broker.TradingClient")
def test_paper_default_constructs_paper_client(mock_client_cls):
    AlpacaBroker(key="k", secret="s")
    mock_client_cls.assert_called_once_with(api_key="k", secret_key="s", paper=True)


def test_live_without_consent_raises():
    with pytest.raises(RuntimeError, match="live_trading_consent"):
        AlpacaBroker(key="k", secret="s", paper=False)


@patch("quant.broker.TradingClient")
def test_live_with_consent_passes_paper_false(mock_client_cls):
    AlpacaBroker(key="k", secret="s", paper=False, live_trading_consent=True)
    mock_client_cls.assert_called_once_with(api_key="k", secret_key="s", paper=False)


@patch("quant.broker.TradingClient")
def test_snapshot_translates_alpaca_objects(mock_client_cls):
    instance = mock_client_cls.return_value
    instance.get_account.return_value = _make_mock_account()
    instance.get_all_positions.return_value = [
        _make_mock_position("AAPL", qty=10, market_value=2000, avg_entry_price=190),
        _make_mock_position("MSFT", qty=5, market_value=1500, avg_entry_price=290),
    ]

    broker = AlpacaBroker(key="k", secret="s")
    snap = broker.snapshot()

    assert snap.equity == 100_000
    assert snap.cash == 50_000
    assert len(snap.positions) == 2
    assert snap.positions[0].ticker == "AAPL"
    assert snap.positions[0].qty == 10
    pmap = snap.position_map()
    assert "AAPL" in pmap and "MSFT" in pmap


@patch("quant.broker.TradingClient")
def test_submit_notional_order_validates_inputs(mock_client_cls):
    broker = AlpacaBroker(key="k", secret="s")

    with pytest.raises(ValueError, match="dollars must be"):
        broker.submit_notional_order("AAPL", dollars=0, side="buy")
    with pytest.raises(ValueError, match="dollars must be"):
        broker.submit_notional_order("AAPL", dollars=-50, side="buy")
    with pytest.raises(ValueError, match="side must be"):
        broker.submit_notional_order("AAPL", dollars=100, side="hold")


@patch("quant.broker.TradingClient")
def test_submit_notional_order_passes_correct_request(mock_client_cls):
    instance = mock_client_cls.return_value
    fake_order = MagicMock()
    fake_order.id = "order-123"
    instance.submit_order.return_value = fake_order

    broker = AlpacaBroker(key="k", secret="s")
    order_id = broker.submit_notional_order("AAPL", dollars=500.123, side="buy")

    assert order_id == "order-123"
    submitted_req = instance.submit_order.call_args[0][0]
    assert submitted_req.symbol == "AAPL"
    assert submitted_req.notional == 500.12  # rounded to 2 decimals


@patch("quant.broker.TradingClient")
def test_close_position_returns_order_id(mock_client_cls):
    instance = mock_client_cls.return_value
    fake_order = MagicMock()
    fake_order.id = "close-456"
    instance.close_position.return_value = fake_order

    broker = AlpacaBroker(key="k", secret="s")
    order_id = broker.close_position("AAPL")

    assert order_id == "close-456"
    instance.close_position.assert_called_once_with("AAPL")


@patch("quant.broker.TradingClient")
def test_cancel_all_orders_returns_count(mock_client_cls):
    instance = mock_client_cls.return_value
    instance.cancel_orders.return_value = ["a", "b", "c"]

    broker = AlpacaBroker(key="k", secret="s")
    assert broker.cancel_all_orders() == 3


@patch("quant.broker.TradingClient")
def test_cancel_all_orders_handles_none(mock_client_cls):
    instance = mock_client_cls.return_value
    instance.cancel_orders.return_value = None

    broker = AlpacaBroker(key="k", secret="s")
    assert broker.cancel_all_orders() == 0
