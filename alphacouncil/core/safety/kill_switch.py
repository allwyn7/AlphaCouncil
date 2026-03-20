"""Emergency kill switch -- the last line of defence.

When activated the kill switch will:

1. Cancel every pending order via the broker adapter.
2. Square-off (flatten) all open positions at market.
3. Broadcast a ``KILL_SWITCH_ACTIVATED`` message on the bus so every agent
   disables itself immediately.
4. Optionally push a Telegram alert to the operator.
5. Log the event to the audit trail.

The switch latches *on* once activated and can only be reset by an explicit
call to :meth:`KillSwitch.reset`, which is intended to be a manual action
performed through the dashboard or CLI after the operator has investigated.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx
import structlog

from alphacouncil.core.models import (
    Order,
    OrderSide,
    OrderType,
    Exchange,
    PortfolioState,
)

if TYPE_CHECKING:
    from alphacouncil.core.safety.audit import AuditTrail

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight protocols so this module stays decoupled from concrete impls
# ---------------------------------------------------------------------------


@runtime_checkable
class BrokerAdapter(Protocol):
    """Minimal broker interface required by the kill switch."""

    async def cancel_all_orders(self) -> int:
        """Cancel every pending order.  Return the number cancelled."""
        ...

    async def square_off_all(self) -> int:
        """Close every open position at market.  Return positions closed."""
        ...


@runtime_checkable
class MessageBus(Protocol):
    """Minimal message-bus interface required by the kill switch."""

    async def publish(self, topic: str, payload: dict[str, Any]) -> None: ...


@runtime_checkable
class Config(Protocol):
    """Configuration surface consumed by the kill switch.

    Any object exposing these attributes (e.g. a Pydantic ``Settings`` model,
    a plain dataclass, or even a ``SimpleNamespace``) satisfies the protocol.
    """

    # Capital base in INR -- used for percentage calculations
    capital: float

    # Kill-switch threshold overrides (all optional at the protocol level;
    # the class uses sensible defaults when values are missing).
    daily_loss_pct: float          # default 0.03
    single_trade_loss_pct: float   # default 0.015
    drawdown_pct: float            # default 0.08
    max_errors_per_hour: int       # default 5
    max_consecutive_slow: int      # default 3
    latency_threshold_ms: float    # default 5000.0

    # Telegram (empty string / None means disabled)
    telegram_bot_token: str
    telegram_chat_id: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ERROR_WINDOW_SECONDS: int = 3600  # 1 hour sliding window for error tracking


class KillSwitch:
    """Latching emergency circuit breaker for the AlphaCouncil system.

    Parameters
    ----------
    broker:
        Adapter used to cancel orders and flatten positions.
    config:
        System configuration providing capital base and threshold overrides.
    bus:
        Message bus for broadcasting disable messages to all agents.
    audit:
        Optional audit trail instance.  When provided, every activation is
        written to the immutable audit log.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        broker: BrokerAdapter,
        config: Config,
        bus: MessageBus,
        audit: AuditTrail | None = None,
    ) -> None:
        self._broker = broker
        self._config = config
        self._bus = bus
        self._audit = audit

        self._active: bool = False
        self._activated_at: datetime | None = None
        self._activation_reason: str | None = None

        # Sliding window of error timestamps (epoch floats)
        self._error_timestamps: deque[float] = deque()

        # Recent API latencies for consecutive-slow detection
        self._recent_latencies: deque[float] = deque(maxlen=50)

        # Extract thresholds with safe defaults
        self._daily_loss_pct: float = getattr(config, "daily_loss_pct", 0.03)
        self._single_trade_loss_pct: float = getattr(config, "single_trade_loss_pct", 0.015)
        self._drawdown_pct: float = getattr(config, "drawdown_pct", 0.08)
        self._max_errors: int = getattr(config, "max_errors_per_hour", 5)
        self._max_consecutive_slow: int = getattr(config, "max_consecutive_slow", 3)
        self._latency_threshold_ms: float = getattr(config, "latency_threshold_ms", 5000.0)
        self._capital: float = config.capital

        logger.info(
            "kill_switch.initialised",
            capital=self._capital,
            daily_loss_pct=self._daily_loss_pct,
            drawdown_pct=self._drawdown_pct,
        )

    # -------------------------------------------------------------- property
    @property
    def is_active(self) -> bool:
        """``True`` when the kill switch has been activated and not yet reset."""
        return self._active

    @property
    def activated_at(self) -> datetime | None:
        """Timestamp of last activation, or ``None``."""
        return self._activated_at

    @property
    def activation_reason(self) -> str | None:
        """Human-readable reason for the last activation."""
        return self._activation_reason

    # -------------------------------------------------------------- activate
    async def activate(self, reason: str) -> None:
        """Activate the kill switch.

        This is an **idempotent** operation -- calling it when already active
        logs a warning but does not re-execute broker actions.

        Steps executed in order:

        1. Cancel all pending orders.
        2. Square-off all open positions at market.
        3. Disable all agents via the message bus.
        4. Send a Telegram alert (if configured).
        5. Write an audit-trail entry.
        """
        if self._active:
            logger.warning("kill_switch.already_active", reason=reason)
            return

        self._active = True
        self._activated_at = datetime.now(timezone.utc)
        self._activation_reason = reason

        logger.critical(
            "kill_switch.ACTIVATED",
            reason=reason,
            timestamp=self._activated_at.isoformat(),
        )

        # 1. Cancel pending orders
        orders_cancelled: int = 0
        try:
            orders_cancelled = await self._broker.cancel_all_orders()
            logger.info("kill_switch.orders_cancelled", count=orders_cancelled)
        except Exception:
            logger.exception("kill_switch.cancel_orders_failed")

        # 2. Square-off positions
        positions_closed: int = 0
        try:
            positions_closed = await self._broker.square_off_all()
            logger.info("kill_switch.positions_closed", count=positions_closed)
        except Exception:
            logger.exception("kill_switch.square_off_failed")

        # 3. Broadcast disable
        try:
            await self._bus.publish(
                "KILL_SWITCH_ACTIVATED",
                {
                    "reason": reason,
                    "timestamp": self._activated_at.isoformat(),
                    "orders_cancelled": orders_cancelled,
                    "positions_closed": positions_closed,
                },
            )
            logger.info("kill_switch.agents_disabled")
        except Exception:
            logger.exception("kill_switch.bus_publish_failed")

        # 4. Telegram alert (fire-and-forget, never block on it)
        asyncio.create_task(
            self._send_telegram(
                f"KILL SWITCH ACTIVATED\n"
                f"Reason: {reason}\n"
                f"Orders cancelled: {orders_cancelled}\n"
                f"Positions closed: {positions_closed}\n"
                f"Time (UTC): {self._activated_at:%Y-%m-%d %H:%M:%S}"
            )
        )

        # 5. Audit
        if self._audit is not None:
            try:
                await self._audit.log_kill_switch(
                    reason=reason,
                    positions_closed=positions_closed,
                    orders_cancelled=orders_cancelled,
                )
            except Exception:
                logger.exception("kill_switch.audit_log_failed")

    # -------------------------------------------------- automatic triggers
    async def check_auto_triggers(
        self,
        portfolio: PortfolioState,
        error_count: int = 0,
        latencies: list[float] | None = None,
    ) -> None:
        """Evaluate automatic kill-switch trigger conditions.

        Parameters
        ----------
        portfolio:
            Current portfolio snapshot.
        error_count:
            Number of new errors to record in the sliding window.
            Typically ``1`` each time a pipeline error occurs.
        latencies:
            Recent API response latencies in **milliseconds**.  Appended to
            the internal ring-buffer for consecutive-slow detection.
        """
        if self._active:
            return

        latencies = latencies or []
        now = time.monotonic()

        # -- Daily loss trigger --
        daily_loss_pct = abs(portfolio.daily_pnl) / self._capital if portfolio.daily_pnl < 0 else 0.0
        if daily_loss_pct >= self._daily_loss_pct:
            await self.activate(
                f"Daily loss {daily_loss_pct:.2%} exceeds threshold "
                f"{self._daily_loss_pct:.2%} of capital"
            )
            return

        # -- Single-trade loss trigger --
        for pos in portfolio.positions:
            if pos.pnl < 0:
                single_loss_pct = abs(pos.pnl) / self._capital
                if single_loss_pct >= self._single_trade_loss_pct:
                    await self.activate(
                        f"Single-position loss on {pos.ticker} "
                        f"({single_loss_pct:.2%}) exceeds threshold "
                        f"{self._single_trade_loss_pct:.2%}"
                    )
                    return

        # -- Drawdown from peak trigger --
        if portfolio.drawdown_from_peak >= self._drawdown_pct:
            await self.activate(
                f"Portfolio drawdown {portfolio.drawdown_from_peak:.2%} "
                f"exceeds threshold {self._drawdown_pct:.2%}"
            )
            return

        # -- Error-rate trigger (sliding 1-hour window) --
        for _ in range(error_count):
            self._error_timestamps.append(now)

        # Prune stale entries outside the 1-hour window
        cutoff = now - _ERROR_WINDOW_SECONDS
        while self._error_timestamps and self._error_timestamps[0] < cutoff:
            self._error_timestamps.popleft()

        if len(self._error_timestamps) >= self._max_errors:
            await self.activate(
                f"{len(self._error_timestamps)} errors in the last hour "
                f"(threshold: {self._max_errors})"
            )
            return

        # -- Consecutive high-latency trigger --
        for lat in latencies:
            self._recent_latencies.append(lat)

        if len(self._recent_latencies) >= self._max_consecutive_slow:
            tail = list(self._recent_latencies)[-self._max_consecutive_slow:]
            if all(lat > self._latency_threshold_ms for lat in tail):
                await self.activate(
                    f"Last {self._max_consecutive_slow} API latencies "
                    f"exceeded {self._latency_threshold_ms}ms: {tail}"
                )
                return

    # ----------------------------------------------------------------- reset
    async def reset(self) -> None:
        """Manually reset the kill switch, allowing trading to resume.

        This is intentionally a manual action -- the operator must verify
        that the root cause has been addressed before calling this.
        """
        if not self._active:
            logger.info("kill_switch.reset_noop", detail="not currently active")
            return

        previous_reason = self._activation_reason
        self._active = False
        self._activated_at = None
        self._activation_reason = None

        # Clear sliding windows so stale data doesn't immediately re-trigger
        self._error_timestamps.clear()
        self._recent_latencies.clear()

        logger.warning(
            "kill_switch.RESET",
            previous_reason=previous_reason,
        )

        try:
            await self._bus.publish(
                "KILL_SWITCH_RESET",
                {
                    "previous_reason": previous_reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception:
            logger.exception("kill_switch.bus_publish_failed_on_reset")

        await self._send_telegram(
            f"Kill switch RESET.\n"
            f"Previous reason: {previous_reason}\n"
            f"Time (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}"
        )

    # ------------------------------------------------------------- telegram
    async def _send_telegram(self, message: str) -> None:
        """Send a Telegram message via the Bot API.

        Silently returns if credentials are not configured.  Errors are
        logged but never propagated -- alerting must never block trading
        operations.
        """
        token = getattr(self._config, "telegram_bot_token", None) or ""
        chat_id = getattr(self._config, "telegram_chat_id", None) or ""

        if not token or not chat_id:
            logger.debug("kill_switch.telegram_not_configured")
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    logger.warning(
                        "kill_switch.telegram_send_failed",
                        status=resp.status_code,
                        body=resp.text[:500],
                    )
                else:
                    logger.info("kill_switch.telegram_sent")
        except httpx.HTTPError as exc:
            logger.warning("kill_switch.telegram_http_error", error=str(exc))
        except Exception:
            logger.exception("kill_switch.telegram_unexpected_error")
