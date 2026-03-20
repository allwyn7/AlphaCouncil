"""Immutable, append-only audit trail.

Every significant event in the AlphaCouncil pipeline -- orders submitted,
kill-switch activations, agent promotions / demotions -- is recorded in a
single SQLite table with monotonically increasing IDs and UTC timestamps.

Design principles:

* **Append-only** -- no ``UPDATE`` or ``DELETE`` operations are exposed.
* **Async-first** -- all writes use ``aiosqlite`` via SQLAlchemy's async
  engine so the audit path never blocks the trading hot loop.
* **Self-contained schema** -- :meth:`AuditTrail.__init__` creates the table
  if it does not exist, so there is no external migration step.
* **Structured payloads** -- event-specific data is serialised as JSON in the
  ``detail`` column, keeping the schema stable while accommodating evolving
  event types.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from alphacouncil.core.models import (
    AgentSignal,
    AgentStatus,
    Order,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

EVENT_ORDER = "ORDER"
EVENT_KILL_SWITCH = "KILL_SWITCH"
EVENT_PROMOTION = "PROMOTION"

# ---------------------------------------------------------------------------
# Table definition (SQLAlchemy Core -- no ORM needed for append-only writes)
# ---------------------------------------------------------------------------

_AUDIT_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS safety_audit (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    event_type    TEXT    NOT NULL,
    agent         TEXT,
    ticker        TEXT,
    action        TEXT,
    quantity      INTEGER,
    price         REAL,
    factor_scores TEXT,
    reasoning     TEXT,
    risk_passed   INTEGER,
    risk_reason   TEXT,
    detail        TEXT,
    severity      TEXT    NOT NULL DEFAULT 'info'
);
"""

_CREATE_INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_safety_audit_ts ON safety_audit(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_safety_audit_agent ON safety_audit(agent);",
    "CREATE INDEX IF NOT EXISTS idx_safety_audit_event ON safety_audit(event_type);",
]


# ---------------------------------------------------------------------------
# AuditTrail
# ---------------------------------------------------------------------------


class AuditTrail:
    """Append-only audit logger backed by SQLite.

    Parameters
    ----------
    db_engine:
        A SQLAlchemy *async* engine (``AsyncEngine``) or a sync ``Engine``.
        When a sync engine is provided, a lightweight async wrapper is
        created internally.  For best performance, pass an ``AsyncEngine``
        created with ``aiosqlite``.
    """

    def __init__(self, db_engine: AsyncEngine | Engine) -> None:
        if isinstance(db_engine, AsyncEngine):
            self._engine: AsyncEngine = db_engine
        else:
            # Wrap a sync engine URL with aiosqlite for async access.
            url = str(db_engine.url).replace("sqlite:///", "sqlite+aiosqlite:///")
            self._engine = create_async_engine(url, echo=False)

        self._initialised: bool = False

    async def _ensure_table(self) -> None:
        """Idempotently create the ``safety_audit`` table and indices."""
        if self._initialised:
            return
        async with self._engine.begin() as conn:
            await conn.execute(text(_AUDIT_TABLE_DDL))
            for idx_ddl in _CREATE_INDEX_DDL:
                await conn.execute(text(idx_ddl))
        self._initialised = True
        logger.info("audit_trail.table_ensured")

    # ----------------------------------------------------------- log_order

    async def log_order(
        self,
        order: Order,
        agent_signal: AgentSignal,
        risk_check: tuple[bool, str],
    ) -> None:
        """Record an order event with full provenance.

        Parameters
        ----------
        order:
            The order ticket (may or may not have been sent to the broker).
        agent_signal:
            The agent signal that originated the order.
        risk_check:
            ``(passed, reason)`` tuple from :class:`PositionLimits`.
        """
        await self._ensure_table()

        passed, reason = risk_check
        detail = {
            "order_id": order.order_id,
            "order_type": order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type),
            "exchange": order.exchange.value if hasattr(order.exchange, "value") else str(order.exchange),
            "trigger_price": order.trigger_price,
            "conviction": agent_signal.conviction,
            "target_weight": agent_signal.target_weight,
            "stop_loss": agent_signal.stop_loss,
            "take_profit": agent_signal.take_profit,
            "holding_period_days": agent_signal.holding_period_days,
        }

        factor_scores_json = json.dumps(agent_signal.factor_scores, default=str)
        detail_json = json.dumps(detail, default=str)

        await self._insert(
            event_type=EVENT_ORDER,
            agent=order.agent_name,
            ticker=order.ticker,
            action=order.side.value if hasattr(order.side, "value") else str(order.side),
            quantity=order.quantity,
            price=order.price,
            factor_scores=factor_scores_json,
            reasoning=agent_signal.reasoning,
            risk_passed=1 if passed else 0,
            risk_reason=reason if reason else None,
            detail=detail_json,
            severity="info" if passed else "warning",
        )

        logger.info(
            "audit_trail.order_logged",
            order_id=order.order_id,
            ticker=order.ticker,
            risk_passed=passed,
        )

    # ----------------------------------------------------- log_kill_switch

    async def log_kill_switch(
        self,
        reason: str,
        positions_closed: int,
        orders_cancelled: int,
    ) -> None:
        """Record a kill-switch activation event."""
        await self._ensure_table()

        detail = {
            "positions_closed": positions_closed,
            "orders_cancelled": orders_cancelled,
        }

        await self._insert(
            event_type=EVENT_KILL_SWITCH,
            agent="system",
            ticker=None,
            action="KILL_SWITCH",
            quantity=None,
            price=None,
            factor_scores=None,
            reasoning=reason,
            risk_passed=0,
            risk_reason=reason,
            detail=json.dumps(detail),
            severity="critical",
        )

        logger.warning(
            "audit_trail.kill_switch_logged",
            reason=reason,
            positions_closed=positions_closed,
            orders_cancelled=orders_cancelled,
        )

    # ------------------------------------------------------ log_promotion

    async def log_promotion(
        self,
        agent_name: str,
        from_status: AgentStatus,
        to_status: AgentStatus,
        evidence: dict[str, Any],
    ) -> None:
        """Record an agent promotion or demotion event."""
        await self._ensure_table()

        detail = {
            "from_status": from_status.value,
            "to_status": to_status.value,
            "evidence": evidence,
        }

        action = "PROMOTION" if _status_rank(to_status) > _status_rank(from_status) else "DEMOTION"
        severity = "info" if action == "PROMOTION" else "warning"

        await self._insert(
            event_type=EVENT_PROMOTION,
            agent=agent_name,
            ticker=None,
            action=action,
            quantity=None,
            price=None,
            factor_scores=None,
            reasoning=f"{from_status.value} -> {to_status.value}",
            risk_passed=1,
            risk_reason=None,
            detail=json.dumps(detail, default=str),
            severity=severity,
        )

        logger.info(
            "audit_trail.promotion_logged",
            agent=agent_name,
            from_status=from_status.value,
            to_status=to_status.value,
        )

    # ---------------------------------------------------------- queries

    async def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the *limit* most recent audit entries, newest first."""
        await self._ensure_table()

        query = text(
            "SELECT * FROM safety_audit ORDER BY id DESC LIMIT :limit"
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(query, {"limit": limit})
            rows = result.mappings().all()
        return [dict(row) for row in rows]

    async def get_agent_audit(
        self,
        agent_name: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return audit entries for a specific *agent_name*, newest first."""
        await self._ensure_table()

        query = text(
            "SELECT * FROM safety_audit "
            "WHERE agent = :agent "
            "ORDER BY id DESC LIMIT :limit"
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(
                query, {"agent": agent_name, "limit": limit}
            )
            rows = result.mappings().all()
        return [dict(row) for row in rows]

    # ---------------------------------------------------------- internals

    async def _insert(
        self,
        *,
        event_type: str,
        agent: str | None,
        ticker: str | None,
        action: str | None,
        quantity: int | None,
        price: float | None,
        factor_scores: str | None,
        reasoning: str | None,
        risk_passed: int | None,
        risk_reason: str | None,
        detail: str | None,
        severity: str,
    ) -> None:
        """Low-level INSERT into the ``safety_audit`` table."""
        stmt = text(
            "INSERT INTO safety_audit "
            "(timestamp, event_type, agent, ticker, action, quantity, price, "
            " factor_scores, reasoning, risk_passed, risk_reason, detail, severity) "
            "VALUES (:ts, :event_type, :agent, :ticker, :action, :quantity, "
            " :price, :factor_scores, :reasoning, :risk_passed, "
            " :risk_reason, :detail, :severity)"
        )
        params = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "agent": agent,
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "price": price,
            "factor_scores": factor_scores,
            "reasoning": reasoning,
            "risk_passed": risk_passed,
            "risk_reason": risk_reason,
            "detail": detail,
            "severity": severity,
        }
        async with self._engine.begin() as conn:
            await conn.execute(stmt, params)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_rank(status: AgentStatus) -> int:
    """Numeric rank for lifecycle statuses to determine promotion vs demotion."""
    _ranks = {
        AgentStatus.BACKTEST: 0,
        AgentStatus.DEMOTED: 1,
        AgentStatus.PAPER: 2,
        AgentStatus.LIVE: 3,
    }
    return _ranks.get(status, -1)
