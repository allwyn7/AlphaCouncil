"""Tests for broker adapters: PaperBroker, BrokerAdapter base, cost model."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphacouncil.core.broker.base import BrokerAdapter, BrokerError, OrderRejectedError
from alphacouncil.core.broker.paper import (
    PaperBroker,
    _BROKERAGE_PCT,
    _GST_ON_BROKERAGE_PCT,
    _SEBI_PCT,
    _SLIPPAGE_PCT,
    _STAMP_BUY_PCT,
    _STT_SELL_PCT,
    _calculate_charges,
)
from alphacouncil.core.models import (
    Exchange,
    Order,
    OrderSide,
    OrderType,
    Position,
    TradeRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(
    ticker: str = "INFY",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: int = 10,
    price: float | None = None,
) -> Order:
    return Order(
        order_id=f"TEST-{ticker}-{side.value}",
        ticker=ticker,
        exchange=Exchange.NSE,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        trigger_price=None,
        agent_name="test_agent",
        reasoning="unit test order",
        timestamp=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Cost model tests
# ---------------------------------------------------------------------------


class TestCostModel:
    def test_buy_charges_structure(self):
        total, slippage_factor = _calculate_charges(
            side=OrderSide.BUY,
            turnover=100_000.0,
            is_market_order=True,
        )
        assert total > 0
        assert slippage_factor > 1.0  # buyer pays more

    def test_sell_charges_include_stt(self):
        total_sell, _ = _calculate_charges(OrderSide.SELL, 100_000.0, False)
        total_buy, _ = _calculate_charges(OrderSide.BUY, 100_000.0, False)
        # Sell side has STT; buy side has stamp duty -- different amounts
        assert total_sell != total_buy

    def test_sell_slippage_factor(self):
        _, slippage = _calculate_charges(OrderSide.SELL, 100_000.0, True)
        assert slippage < 1.0  # seller gets less

    def test_no_slippage_for_limit_orders(self):
        _, slippage = _calculate_charges(OrderSide.BUY, 100_000.0, False)
        assert slippage == 1.0

    def test_charges_proportional_to_turnover(self):
        small, _ = _calculate_charges(OrderSide.BUY, 10_000.0, False)
        big, _ = _calculate_charges(OrderSide.BUY, 100_000.0, False)
        assert big == pytest.approx(small * 10, rel=0.01)

    def test_buy_charges_exact_components(self):
        turnover = 100_000.0
        total, _ = _calculate_charges(OrderSide.BUY, turnover, False)
        brokerage = turnover * _BROKERAGE_PCT
        gst = brokerage * _GST_ON_BROKERAGE_PCT
        sebi = turnover * _SEBI_PCT
        stamp = turnover * _STAMP_BUY_PCT
        expected = brokerage + gst + sebi + stamp
        assert total == pytest.approx(expected, rel=1e-9)

    def test_sell_charges_exact_components(self):
        turnover = 100_000.0
        total, _ = _calculate_charges(OrderSide.SELL, turnover, False)
        brokerage = turnover * _BROKERAGE_PCT
        gst = brokerage * _GST_ON_BROKERAGE_PCT
        sebi = turnover * _SEBI_PCT
        stt = turnover * _STT_SELL_PCT
        expected = brokerage + gst + sebi + stt
        assert total == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# PaperBroker tests
# ---------------------------------------------------------------------------


class TestPaperBroker:
    def test_initial_capital(self):
        broker = PaperBroker(initial_capital=500_000.0)
        assert broker.cash == 500_000.0
        assert broker.initial_capital == 500_000.0

    @pytest.mark.asyncio
    async def test_place_market_order_buy(self):
        broker = PaperBroker(initial_capital=1_000_000.0)
        with patch.object(broker, "_fetch_ltp", return_value=1500.0):
            order = _make_order("INFY", OrderSide.BUY, OrderType.MARKET, 10)
            order_id = await broker.place_order(order)
            assert order_id

        # Cash should have decreased
        assert broker.cash < 1_000_000.0
        assert len(broker.trades) == 1
        assert broker.trades[0].side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_place_and_cancel_limit_order(self):
        broker = PaperBroker(initial_capital=1_000_000.0)
        # Limit buy below market -> should pend (not fill)
        with patch.object(broker, "_fetch_ltp", return_value=1500.0):
            order = _make_order("INFY", OrderSide.BUY, OrderType.LIMIT, 10, price=1400.0)
            order_id = await broker.place_order(order)

        # Order should be pending, not filled
        assert len(broker.trades) == 0

        # Cancel it
        cancelled = await broker.cancel_order(order_id)
        assert cancelled is True

        # Cancelling again should fail
        cancelled_again = await broker.cancel_order(order_id)
        assert cancelled_again is False

    @pytest.mark.asyncio
    async def test_buy_then_sell(self):
        broker = PaperBroker(initial_capital=1_000_000.0)
        with patch.object(broker, "_fetch_ltp", return_value=1000.0):
            buy_order = _make_order("TCS", OrderSide.BUY, OrderType.MARKET, 20)
            await broker.place_order(buy_order)

            sell_order = _make_order("TCS", OrderSide.SELL, OrderType.MARKET, 20)
            await broker.place_order(sell_order)

        # Position should be gone
        assert "TCS" not in broker._positions
        assert len(broker.trades) == 2

    @pytest.mark.asyncio
    async def test_sell_without_position_raises(self):
        broker = PaperBroker(initial_capital=1_000_000.0)
        with patch.object(broker, "_fetch_ltp", return_value=500.0):
            sell_order = _make_order("WIPRO", OrderSide.SELL, OrderType.MARKET, 10)
            with pytest.raises(OrderRejectedError):
                await broker.place_order(sell_order)

    @pytest.mark.asyncio
    async def test_insufficient_funds_raises(self):
        broker = PaperBroker(initial_capital=100.0)
        with patch.object(broker, "_fetch_ltp", return_value=500.0):
            order = _make_order("RELIANCE", OrderSide.BUY, OrderType.MARKET, 100)
            with pytest.raises(OrderRejectedError, match="Insufficient funds"):
                await broker.place_order(order)

    @pytest.mark.asyncio
    async def test_get_funds(self):
        broker = PaperBroker(initial_capital=1_000_000.0)
        with patch.object(broker, "_fetch_ltp", return_value=1000.0):
            order = _make_order("INFY", OrderSide.BUY, OrderType.MARKET, 10)
            await broker.place_order(order)

        funds = await broker.get_funds()
        assert "cash" in funds
        assert "deployed" in funds
        assert "total" in funds
        assert funds["total"] == pytest.approx(
            funds["cash"] + funds["deployed"], abs=1.0
        )

    @pytest.mark.asyncio
    async def test_get_positions_returns_position_objects(self):
        broker = PaperBroker(initial_capital=1_000_000.0)
        with patch.object(broker, "_fetch_ltp", return_value=2000.0):
            order = _make_order("RELIANCE", OrderSide.BUY, OrderType.MARKET, 5)
            await broker.place_order(order)
            positions = await broker.get_positions()

        assert len(positions) == 1
        assert isinstance(positions[0], Position)
        assert positions[0].ticker == "RELIANCE"
        assert positions[0].quantity == 5

    def test_order_book_tracking(self):
        broker = PaperBroker()
        assert broker.order_book == []

    def test_summary_structure(self):
        broker = PaperBroker(initial_capital=500_000.0)
        summary = broker.summary()
        assert summary["initial_capital"] == 500_000.0
        assert summary["cash"] == 500_000.0
        assert summary["deployed"] == 0.0
        assert summary["pnl"] == 0.0
        assert summary["num_positions"] == 0
        assert summary["num_trades"] == 0

    def test_yf_ticker_suffix(self):
        assert PaperBroker._yf_ticker("RELIANCE") == "RELIANCE.NS"
        assert PaperBroker._yf_ticker("RELIANCE.NS") == "RELIANCE.NS"
        assert PaperBroker._yf_ticker("RELIANCE.BO") == "RELIANCE.BO"


# ---------------------------------------------------------------------------
# BrokerAdapter abstract interface tests
# ---------------------------------------------------------------------------


class TestBrokerAdapterInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BrokerAdapter()  # type: ignore[abstract]

    def test_repr(self):
        broker = PaperBroker()
        r = repr(broker)
        assert "PaperBroker" in r

    def test_broker_error_hierarchy(self):
        assert issubclass(OrderRejectedError, BrokerError)

    @pytest.mark.asyncio
    async def test_connect_disconnect_are_noop(self):
        broker = PaperBroker()
        await broker.connect()
        await broker.disconnect()
        # Should not raise
