"""Tests for safety rails: PositionLimits, KillSwitch, ValidationGate, AuditTrail.

This is the CRITICAL test file -- every safety check must be verified.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alphacouncil.core.models import (
    AgentSignal,
    AgentStatus,
    Action,
    Exchange,
    Order,
    OrderSide,
    OrderType,
    PortfolioState,
    Position,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(capital: float = 1_000_000.0, **overrides: Any) -> SimpleNamespace:
    defaults = {
        "capital": capital,
        "max_per_stock_pct": 0.05,
        "max_per_sector_pct": 0.25,
        "max_deployed_pct": 0.80,
        "max_open_positions": 15,
        "max_daily_trades": 50,
        "max_order_value": 50_000.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_order(
    ticker: str = "INFY",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: int = 10,
    price: float = 1000.0,
) -> Order:
    return Order(
        order_id=f"ORD-{ticker}",
        ticker=ticker,
        exchange=Exchange.NSE,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        trigger_price=None,
        agent_name="test_agent",
        reasoning="test",
        timestamp=datetime.now(tz=timezone.utc),
    )


def _make_portfolio(
    cash: float = 500_000.0,
    positions: list[Position] | None = None,
    total_value: float = 1_000_000.0,
    deployed_pct: float = 0.50,
    daily_pnl: float = 0.0,
    daily_pnl_pct: float = 0.0,
    drawdown: float = 0.0,
) -> PortfolioState:
    return PortfolioState(
        cash=cash,
        positions=positions or [],
        total_value=total_value,
        deployed_pct=deployed_pct,
        daily_pnl=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        drawdown_from_peak=drawdown,
    )


def _make_position(
    ticker: str = "INFY",
    quantity: int = 10,
    avg_price: float = 1000.0,
    current_price: float = 1050.0,
    sector: str | None = "IT",
) -> Position:
    pnl = (current_price - avg_price) * quantity
    pnl_pct = ((current_price / avg_price) - 1.0) * 100
    return Position(
        ticker=ticker,
        quantity=quantity,
        avg_price=avg_price,
        current_price=current_price,
        pnl=pnl,
        pnl_pct=pnl_pct,
        sector=sector,
    )


# ---------------------------------------------------------------------------
# PositionLimits tests
# ---------------------------------------------------------------------------


class TestPositionLimits:
    @pytest.mark.asyncio
    async def test_order_within_limits_passes(self):
        from alphacouncil.core.safety.limits import PositionLimits

        config = _make_config()
        limits = PositionLimits(config)
        order = _make_order(quantity=10, price=1000.0)  # 10,000 INR
        portfolio = _make_portfolio()

        with patch.object(limits, "_check_trading_hours", return_value=(True, "")):
            passed, reason = await limits.check_order(order, portfolio)
        assert passed is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_max_order_value_exceeded(self):
        from alphacouncil.core.safety.limits import PositionLimits

        config = _make_config(max_order_value=50_000.0)
        limits = PositionLimits(config)
        # 100 shares at 1000 = 100,000 > 50,000 limit
        order = _make_order(quantity=100, price=1000.0)
        portfolio = _make_portfolio()

        passed, reason = await limits.check_order(order, portfolio)
        assert passed is False
        assert "per-order" in reason.lower() or "order value" in reason.lower()

    @pytest.mark.asyncio
    async def test_per_stock_concentration_limit(self):
        from alphacouncil.core.safety.limits import PositionLimits

        config = _make_config(capital=1_000_000.0, max_per_stock_pct=0.05)
        limits = PositionLimits(config)
        # Existing position worth 45,000; new order adds 10,000 -> 55,000 > 50,000 (5% of 1M)
        existing = _make_position("INFY", 30, 1500.0, 1500.0)
        portfolio = _make_portfolio(positions=[existing])
        order = _make_order("INFY", OrderSide.BUY, quantity=10, price=1000.0)

        with patch.object(limits, "_check_trading_hours", return_value=(True, "")):
            passed, reason = await limits.check_order(order, portfolio)
        assert passed is False
        assert "INFY" in reason

    @pytest.mark.asyncio
    async def test_per_sector_concentration_limit(self):
        from alphacouncil.core.safety.limits import PositionLimits

        config = _make_config(capital=1_000_000.0, max_per_sector_pct=0.25)
        limits = PositionLimits(config)
        # Sector "IT" already has 240,000 exposure; adding 20,000 -> 260,000 > 250,000
        positions = [
            _make_position("TCS", 80, 3000.0, 3000.0, "IT"),
        ]
        portfolio = _make_portfolio(positions=positions)
        order = _make_order("INFY", OrderSide.BUY, quantity=20, price=1000.0)

        with patch.object(limits, "_check_trading_hours", return_value=(True, "")):
            passed, reason = await limits.check_order(order, portfolio)
        assert passed is False
        assert "sector" in reason.lower()

    @pytest.mark.asyncio
    async def test_max_deployment_limit(self):
        from alphacouncil.core.safety.limits import PositionLimits

        config = _make_config(max_deployed_pct=0.80)
        limits = PositionLimits(config)
        portfolio = _make_portfolio(deployed_pct=0.79, total_value=1_000_000.0)
        # Adding an order worth 20,000 -> 0.79 + 0.02 = 0.81 > 0.80
        order = _make_order(quantity=20, price=1000.0)

        with patch.object(limits, "_check_trading_hours", return_value=(True, "")):
            passed, reason = await limits.check_order(order, portfolio)
        assert passed is False
        assert "deployment" in reason.lower() or "deployed" in reason.lower()

    @pytest.mark.asyncio
    async def test_max_positions_limit(self):
        from alphacouncil.core.safety.limits import PositionLimits

        config = _make_config(max_open_positions=15)
        limits = PositionLimits(config)
        positions = [
            _make_position(f"STOCK{i}", 10, 100.0, 100.0) for i in range(15)
        ]
        portfolio = _make_portfolio(positions=positions)
        # New ticker -> 16th position
        order = _make_order("NEWSTOCK", OrderSide.BUY, quantity=1, price=100.0)

        with patch.object(limits, "_check_trading_hours", return_value=(True, "")):
            passed, reason = await limits.check_order(order, portfolio)
        assert passed is False
        assert "15" in reason

    @pytest.mark.asyncio
    async def test_daily_trade_limit(self):
        from alphacouncil.core.safety.limits import PositionLimits

        config = _make_config(max_daily_trades=3)
        limits = PositionLimits(config)
        portfolio = _make_portfolio()
        order = _make_order(quantity=1, price=100.0)

        with patch.object(limits, "_check_trading_hours", return_value=(True, "")):
            for _ in range(3):
                passed, _ = await limits.check_order(order, portfolio)
                assert passed is True

            passed, reason = await limits.check_order(order, portfolio)
            assert passed is False
            assert "daily trade limit" in reason.lower()

    def test_reset_daily_counter(self):
        from alphacouncil.core.safety.limits import PositionLimits

        config = _make_config()
        limits = PositionLimits(config)
        limits._daily_trade_count = 42
        limits.reset_daily_counter()
        assert limits._daily_trade_count == 0

    @pytest.mark.asyncio
    async def test_short_selling_blocked_in_nse_cash(self):
        from alphacouncil.core.safety.limits import PositionLimits

        config = _make_config()
        limits = PositionLimits(config)
        portfolio = _make_portfolio(
            positions=[_make_position("INFY", 5, 1000.0, 1000.0)]
        )
        order = _make_order("INFY", OrderSide.SELL, quantity=10, price=1000.0)

        with patch.object(limits, "_check_trading_hours", return_value=(True, "")):
            passed, reason = await limits.check_order(order, portfolio)
        assert passed is False
        assert "short selling" in reason.lower()

    @pytest.mark.asyncio
    async def test_get_utilization(self):
        from alphacouncil.core.safety.limits import PositionLimits

        config = _make_config()
        limits = PositionLimits(config)
        portfolio = _make_portfolio(
            positions=[_make_position("INFY", 10, 1000.0, 1000.0)],
            deployed_pct=0.40,
        )
        util = await limits.get_utilization(portfolio)
        assert "per_stock_pct" in util
        assert "deployed_pct" in util
        assert all(isinstance(v, float) for v in util.values())


# ---------------------------------------------------------------------------
# Trading hours tests
# ---------------------------------------------------------------------------


class TestTradingHours:
    def test_regular_order_during_market_hours(self):
        from alphacouncil.core.safety.limits import PositionLimits

        order = _make_order(order_type=OrderType.MARKET)
        # Patch time to 10:00 IST
        with patch("alphacouncil.core.safety.limits.datetime") as mock_dt:
            from datetime import time as time_cls
            mock_now = MagicMock()
            mock_now.time.return_value = time_cls(10, 0, 0)
            mock_dt.now.return_value = mock_now
            passed, reason = PositionLimits._check_trading_hours(order)
        assert passed is True

    def test_regular_order_outside_market_hours(self):
        from alphacouncil.core.safety.limits import PositionLimits

        order = _make_order(order_type=OrderType.MARKET)
        with patch("alphacouncil.core.safety.limits.datetime") as mock_dt:
            from datetime import time as time_cls
            mock_now = MagicMock()
            mock_now.time.return_value = time_cls(17, 0, 0)
            mock_dt.now.return_value = mock_now
            passed, reason = PositionLimits._check_trading_hours(order)
        assert passed is False
        assert "09:15-15:30" in reason

    def test_amo_order_in_amo_window(self):
        from alphacouncil.core.safety.limits import PositionLimits

        order = _make_order(order_type=OrderType.AMO)
        with patch("alphacouncil.core.safety.limits.datetime") as mock_dt:
            from datetime import time as time_cls
            mock_now = MagicMock()
            mock_now.time.return_value = time_cls(16, 0, 0)  # 16:00 IST
            mock_dt.now.return_value = mock_now
            passed, reason = PositionLimits._check_trading_hours(order)
        assert passed is True


# ---------------------------------------------------------------------------
# KillSwitch tests
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def _make_kill_switch(self, **config_overrides):
        from alphacouncil.core.safety.kill_switch import KillSwitch

        config = SimpleNamespace(
            capital=1_000_000.0,
            daily_loss_pct=0.03,
            single_trade_loss_pct=0.015,
            drawdown_pct=0.08,
            max_errors_per_hour=5,
            max_consecutive_slow=3,
            latency_threshold_ms=5000.0,
            telegram_bot_token="",
            telegram_chat_id="",
            **config_overrides,
        )
        broker = AsyncMock()
        broker.cancel_all_orders = AsyncMock(return_value=0)
        broker.square_off_all = AsyncMock(return_value=0)
        bus = AsyncMock()
        bus.publish = AsyncMock()
        return KillSwitch(broker=broker, config=config, bus=bus)

    @pytest.mark.asyncio
    async def test_daily_loss_trigger(self):
        ks = self._make_kill_switch()
        portfolio = _make_portfolio(daily_pnl=-35_000.0)  # 3.5% > 3%
        await ks.check_auto_triggers(portfolio)
        assert ks.is_active is True
        assert "daily loss" in ks.activation_reason.lower()

    @pytest.mark.asyncio
    async def test_drawdown_trigger(self):
        ks = self._make_kill_switch()
        portfolio = _make_portfolio(drawdown=0.09)  # 9% > 8%
        await ks.check_auto_triggers(portfolio)
        assert ks.is_active is True
        assert "drawdown" in ks.activation_reason.lower()

    @pytest.mark.asyncio
    async def test_error_threshold_trigger(self):
        ks = self._make_kill_switch(max_errors_per_hour=3)
        portfolio = _make_portfolio()
        # Inject 3 errors
        await ks.check_auto_triggers(portfolio, error_count=3)
        assert ks.is_active is True
        assert "error" in ks.activation_reason.lower()

    @pytest.mark.asyncio
    async def test_no_trigger_below_thresholds(self):
        ks = self._make_kill_switch()
        portfolio = _make_portfolio(daily_pnl=-10_000.0, drawdown=0.02)
        await ks.check_auto_triggers(portfolio)
        assert ks.is_active is False

    @pytest.mark.asyncio
    async def test_idempotent_activation(self):
        ks = self._make_kill_switch()
        await ks.activate("first reason")
        assert ks.is_active is True
        # Second activation should be a no-op
        await ks.activate("second reason")
        assert ks.activation_reason == "first reason"

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        ks = self._make_kill_switch()
        await ks.activate("test")
        assert ks.is_active is True
        await ks.reset()
        assert ks.is_active is False
        assert ks.activated_at is None

    @pytest.mark.asyncio
    async def test_consecutive_slow_latency_trigger(self):
        ks = self._make_kill_switch(max_consecutive_slow=3, latency_threshold_ms=100.0)
        portfolio = _make_portfolio()
        # Feed 3 consecutive slow latencies
        await ks.check_auto_triggers(portfolio, latencies=[200.0, 200.0, 200.0])
        assert ks.is_active is True
        assert "latenc" in ks.activation_reason.lower()


# ---------------------------------------------------------------------------
# ValidationGate tests (mocked DB)
# ---------------------------------------------------------------------------


class TestValidationGate:
    """Test promotion / demotion criteria using a mocked DB session."""

    def test_promotion_thresholds_defined(self):
        from alphacouncil.core.safety.validation_gate import (
            _MAX_DRAWDOWN,
            _MIN_PAPER_DAYS,
            _MIN_SHARPE,
            _MIN_WIN_RATE,
        )

        assert _MIN_PAPER_DAYS == 30
        assert _MIN_SHARPE == 0.5
        assert _MAX_DRAWDOWN == 0.15
        assert _MIN_WIN_RATE == 0.40

    def test_demotion_thresholds_defined(self):
        from alphacouncil.core.safety.validation_gate import (
            _DEMOTION_SHARPE_THRESHOLD,
            _DEMOTION_SHARPE_WINDOW,
        )

        assert _DEMOTION_SHARPE_WINDOW == 20
        assert _DEMOTION_SHARPE_THRESHOLD == 0.0

    @pytest.mark.asyncio
    async def test_promotion_criteria_all_met(self):
        from alphacouncil.core.safety.validation_gate import AgentRecord, ValidationGate

        session = AsyncMock()
        gate = ValidationGate(db_session=session)

        record = MagicMock(spec=AgentRecord)
        record.status = AgentStatus.PAPER.value
        record.paper_days = 45
        record.paper_sharpe = 0.8
        record.paper_max_drawdown = 0.10
        record.paper_win_rate = 0.55

        with patch.object(gate, "_get_record", return_value=record):
            eligible, reason = await gate.check_promotion("test_agent")
        assert eligible is True
        assert "met" in reason.lower()

    @pytest.mark.asyncio
    async def test_promotion_fails_insufficient_days(self):
        from alphacouncil.core.safety.validation_gate import AgentRecord, ValidationGate

        session = AsyncMock()
        gate = ValidationGate(db_session=session)

        record = MagicMock(spec=AgentRecord)
        record.status = AgentStatus.PAPER.value
        record.paper_days = 15  # < 30
        record.paper_sharpe = 0.8
        record.paper_max_drawdown = 0.10
        record.paper_win_rate = 0.55

        with patch.object(gate, "_get_record", return_value=record):
            eligible, reason = await gate.check_promotion("test_agent")
        assert eligible is False
        assert "days" in reason.lower()

    @pytest.mark.asyncio
    async def test_promotion_fails_low_sharpe(self):
        from alphacouncil.core.safety.validation_gate import AgentRecord, ValidationGate

        session = AsyncMock()
        gate = ValidationGate(db_session=session)

        record = MagicMock(spec=AgentRecord)
        record.status = AgentStatus.PAPER.value
        record.paper_days = 45
        record.paper_sharpe = 0.3  # < 0.5
        record.paper_max_drawdown = 0.10
        record.paper_win_rate = 0.55

        with patch.object(gate, "_get_record", return_value=record):
            eligible, reason = await gate.check_promotion("test_agent")
        assert eligible is False
        assert "sharpe" in reason.lower()

    @pytest.mark.asyncio
    async def test_promotion_fails_high_drawdown(self):
        from alphacouncil.core.safety.validation_gate import AgentRecord, ValidationGate

        session = AsyncMock()
        gate = ValidationGate(db_session=session)

        record = MagicMock(spec=AgentRecord)
        record.status = AgentStatus.PAPER.value
        record.paper_days = 45
        record.paper_sharpe = 0.8
        record.paper_max_drawdown = 0.20  # >= 0.15
        record.paper_win_rate = 0.55

        with patch.object(gate, "_get_record", return_value=record):
            eligible, reason = await gate.check_promotion("test_agent")
        assert eligible is False
        assert "drawdown" in reason.lower()

    @pytest.mark.asyncio
    async def test_promotion_fails_low_win_rate(self):
        from alphacouncil.core.safety.validation_gate import AgentRecord, ValidationGate

        session = AsyncMock()
        gate = ValidationGate(db_session=session)

        record = MagicMock(spec=AgentRecord)
        record.status = AgentStatus.PAPER.value
        record.paper_days = 45
        record.paper_sharpe = 0.8
        record.paper_max_drawdown = 0.10
        record.paper_win_rate = 0.30  # <= 0.40

        with patch.object(gate, "_get_record", return_value=record):
            eligible, reason = await gate.check_promotion("test_agent")
        assert eligible is False
        assert "win rate" in reason.lower()

    @pytest.mark.asyncio
    async def test_demotion_negative_sharpe(self):
        from alphacouncil.core.safety.validation_gate import AgentRecord, ValidationGate

        session = AsyncMock()
        session.commit = AsyncMock()
        gate = ValidationGate(db_session=session)

        record = MagicMock(spec=AgentRecord)
        record.status = AgentStatus.LIVE.value

        with patch.object(gate, "_get_record", return_value=record):
            demoted = await gate.check_demotion("test_agent", rolling_sharpe=-0.5)
        assert demoted is True
        assert record.status == AgentStatus.DEMOTED.value

    @pytest.mark.asyncio
    async def test_no_demotion_positive_sharpe(self):
        from alphacouncil.core.safety.validation_gate import AgentRecord, ValidationGate

        session = AsyncMock()
        gate = ValidationGate(db_session=session)

        record = MagicMock(spec=AgentRecord)
        record.status = AgentStatus.LIVE.value

        with patch.object(gate, "_get_record", return_value=record):
            demoted = await gate.check_demotion("test_agent", rolling_sharpe=0.5)
        assert demoted is False


# ---------------------------------------------------------------------------
# AuditTrail tests
# ---------------------------------------------------------------------------


class TestAuditTrail:
    @pytest.mark.asyncio
    async def test_log_order_appends(self):
        from alphacouncil.core.safety.audit import AuditTrail

        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        trail = AuditTrail(engine)

        order = _make_order()
        now = datetime.now(tz=timezone.utc)
        signal = AgentSignal(
            ticker="INFY", action=Action.BUY, conviction=75,
            target_weight=0.05, stop_loss=900.0, take_profit=1200.0,
            factor_scores={"alpha": 1.0}, reasoning="test",
            holding_period_days=10, agent_name="test_agent",
            timestamp=now,
        )
        await trail.log_order(order, signal, (True, ""))

        entries = await trail.get_recent(limit=10)
        assert len(entries) == 1
        assert entries[0]["event_type"] == "ORDER"

    @pytest.mark.asyncio
    async def test_log_kill_switch_event(self):
        from alphacouncil.core.safety.audit import AuditTrail

        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        trail = AuditTrail(engine)
        await trail.log_kill_switch("daily loss exceeded", 3, 5)

        entries = await trail.get_recent(limit=10)
        assert len(entries) == 1
        assert entries[0]["event_type"] == "KILL_SWITCH"
        assert entries[0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_audit_is_append_only(self):
        """Verify multiple appends produce cumulative results."""
        from alphacouncil.core.safety.audit import AuditTrail

        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        trail = AuditTrail(engine)

        await trail.log_kill_switch("reason 1", 0, 0)
        await trail.log_kill_switch("reason 2", 0, 0)
        await trail.log_kill_switch("reason 3", 0, 0)

        entries = await trail.get_recent(limit=50)
        assert len(entries) == 3

    @pytest.mark.asyncio
    async def test_log_promotion_event(self):
        from alphacouncil.core.safety.audit import AuditTrail

        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        trail = AuditTrail(engine)
        await trail.log_promotion(
            "growth_momentum",
            AgentStatus.PAPER,
            AgentStatus.LIVE,
            {"sharpe": 0.8, "win_rate": 0.55},
        )

        entries = await trail.get_recent(limit=10)
        assert len(entries) == 1
        assert entries[0]["event_type"] == "PROMOTION"

    @pytest.mark.asyncio
    async def test_get_agent_audit_filters(self):
        from alphacouncil.core.safety.audit import AuditTrail

        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        trail = AuditTrail(engine)

        await trail.log_kill_switch("system event", 0, 0)

        now = datetime.now(tz=timezone.utc)
        order = _make_order("TCS")
        signal = AgentSignal(
            ticker="TCS", action=Action.BUY, conviction=60,
            target_weight=0.04, stop_loss=3000.0, take_profit=4000.0,
            factor_scores={}, reasoning="test", holding_period_days=5,
            agent_name="growth_momentum", timestamp=now,
        )
        await trail.log_order(order, signal, (True, ""))

        agent_entries = await trail.get_agent_audit("test_agent", limit=50)
        assert len(agent_entries) == 1
        assert agent_entries[0]["ticker"] == "TCS"
