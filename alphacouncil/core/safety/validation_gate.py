"""Agent lifecycle validation gate.

Governs the promotion pipeline:

    BACKTEST -> PAPER -> LIVE -> DEMOTED

Agents must demonstrate consistent, statistically significant performance in
paper trading before they are eligible for live capital.  Once live, an agent
that under-performs is automatically demoted back to paper trading.

The gate uses an async-compatible SQLAlchemy session to persist agent status
and paper-trading statistics so decisions survive restarts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import DeclarativeBase

from alphacouncil.core.models import AgentStatus

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Promotion thresholds (PAPER -> LIVE)
# ---------------------------------------------------------------------------

_MIN_PAPER_DAYS: int = 30
_MIN_SHARPE: float = 0.5
_MAX_DRAWDOWN: float = 0.15       # 15%
_MIN_WIN_RATE: float = 0.40       # 40%

# Demotion threshold (LIVE -> PAPER)
_DEMOTION_SHARPE_WINDOW: int = 20  # rolling 20-day
_DEMOTION_SHARPE_THRESHOLD: float = 0.0


# ---------------------------------------------------------------------------
# ORM model for agent status persistence
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class AgentRecord(_Base):
    """Persistent record of an agent's lifecycle status and paper-trading stats."""

    __tablename__ = "agent_lifecycle"

    id: int = Column(Integer, primary_key=True, autoincrement=True)  # type: ignore[assignment]
    agent_name: str = Column(String(128), nullable=False, unique=True, index=True)  # type: ignore[assignment]
    status: str = Column(String(32), nullable=False, default=AgentStatus.BACKTEST.value)  # type: ignore[assignment]
    updated_at: datetime = Column(  # type: ignore[assignment]
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Paper-trading statistics (accumulated)
    paper_days: int = Column(Integer, nullable=False, default=0)  # type: ignore[assignment]
    paper_sharpe: float = Column(Float, nullable=True)  # type: ignore[assignment]
    paper_max_drawdown: float = Column(Float, nullable=True)  # type: ignore[assignment]
    paper_win_rate: float = Column(Float, nullable=True)  # type: ignore[assignment]
    paper_total_trades: int = Column(Integer, nullable=False, default=0)  # type: ignore[assignment]

    # Free-form JSON for extended evidence / notes
    notes: str | None = Column(Text, nullable=True)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# ValidationGate
# ---------------------------------------------------------------------------


class ValidationGate:
    """Agent lifecycle gate governing promotion and demotion decisions.

    Parameters
    ----------
    db_session:
        An *async* SQLAlchemy session.  The gate performs reads and writes
        through this session and calls ``commit`` when mutating state.
    """

    def __init__(self, db_session: AsyncSession) -> None:
        self._session = db_session

    # ------------------------------------------------------ table bootstrap
    async def ensure_tables(self) -> None:
        """Create the ``agent_lifecycle`` table if it does not exist.

        Call once during application startup.
        """
        async with self._session.bind.begin() as conn:  # type: ignore[union-attr]
            await conn.run_sync(_Base.metadata.create_all)
        logger.info("validation_gate.tables_ensured")

    # -------------------------------------------------------- status CRUD

    async def get_agent_status(self, agent_name: str) -> AgentStatus:
        """Return the current lifecycle status of *agent_name*.

        Returns ``AgentStatus.BACKTEST`` for unknown agents (first contact).
        """
        record = await self._get_record(agent_name)
        if record is None:
            return AgentStatus.BACKTEST
        return AgentStatus(record.status)

    async def set_agent_status(
        self,
        agent_name: str,
        status: AgentStatus,
    ) -> None:
        """Explicitly set the lifecycle status of *agent_name*."""
        record = await self._get_or_create(agent_name)
        record.status = status.value
        record.updated_at = datetime.now(timezone.utc)
        await self._session.commit()
        logger.info(
            "validation_gate.status_set",
            agent=agent_name,
            status=status.value,
        )

    # -------------------------------------------------------- paper stats

    async def get_paper_stats(self, agent_name: str) -> dict[str, Any]:
        """Return accumulated paper-trading statistics for *agent_name*.

        Keys: ``sharpe``, ``max_drawdown``, ``win_rate``, ``days``,
        ``total_trades``.
        """
        record = await self._get_record(agent_name)
        if record is None:
            return {
                "sharpe": None,
                "max_drawdown": None,
                "win_rate": None,
                "days": 0,
                "total_trades": 0,
            }
        return {
            "sharpe": record.paper_sharpe,
            "max_drawdown": record.paper_max_drawdown,
            "win_rate": record.paper_win_rate,
            "days": record.paper_days,
            "total_trades": record.paper_total_trades,
        }

    async def update_paper_stats(
        self,
        agent_name: str,
        *,
        days: int | None = None,
        sharpe: float | None = None,
        max_drawdown: float | None = None,
        win_rate: float | None = None,
        total_trades: int | None = None,
    ) -> None:
        """Update paper-trading statistics for *agent_name*.

        Only non-``None`` arguments are written, so callers can update a
        single metric without overwriting the rest.
        """
        record = await self._get_or_create(agent_name)
        if days is not None:
            record.paper_days = days
        if sharpe is not None:
            record.paper_sharpe = sharpe
        if max_drawdown is not None:
            record.paper_max_drawdown = max_drawdown
        if win_rate is not None:
            record.paper_win_rate = win_rate
        if total_trades is not None:
            record.paper_total_trades = total_trades
        record.updated_at = datetime.now(timezone.utc)
        await self._session.commit()
        logger.debug("validation_gate.paper_stats_updated", agent=agent_name)

    # ------------------------------------------------------- promotion check

    async def check_promotion(
        self,
        agent_name: str,
    ) -> tuple[bool, str]:
        """Evaluate whether *agent_name* is eligible for promotion.

        Currently implements the **PAPER -> LIVE** gate.  Returns a tuple of
        ``(eligible, reason)`` where *reason* explains the decision.

        Promotion criteria (all must be satisfied):

        * >= 30 paper-trading days
        * Sharpe ratio > 0.5
        * Max drawdown < 15%
        * Win rate > 40%
        """
        record = await self._get_record(agent_name)

        if record is None:
            return False, f"Agent '{agent_name}' has no lifecycle record"

        current = AgentStatus(record.status)
        if current != AgentStatus.PAPER:
            return False, (
                f"Promotion check only applies to PAPER agents; "
                f"'{agent_name}' is currently {current.value}"
            )

        failures: list[str] = []

        if record.paper_days < _MIN_PAPER_DAYS:
            failures.append(
                f"Insufficient paper days: {record.paper_days}/{_MIN_PAPER_DAYS}"
            )

        if record.paper_sharpe is None or record.paper_sharpe <= _MIN_SHARPE:
            sharpe_str = f"{record.paper_sharpe:.2f}" if record.paper_sharpe is not None else "N/A"
            failures.append(
                f"Sharpe ratio too low: {sharpe_str} (required > {_MIN_SHARPE})"
            )

        if record.paper_max_drawdown is None or record.paper_max_drawdown >= _MAX_DRAWDOWN:
            dd_str = (
                f"{record.paper_max_drawdown:.2%}"
                if record.paper_max_drawdown is not None
                else "N/A"
            )
            failures.append(
                f"Max drawdown too high: {dd_str} (required < {_MAX_DRAWDOWN:.0%})"
            )

        if record.paper_win_rate is None or record.paper_win_rate <= _MIN_WIN_RATE:
            wr_str = (
                f"{record.paper_win_rate:.2%}"
                if record.paper_win_rate is not None
                else "N/A"
            )
            failures.append(
                f"Win rate too low: {wr_str} (required > {_MIN_WIN_RATE:.0%})"
            )

        if failures:
            combined = "; ".join(failures)
            logger.info(
                "validation_gate.promotion_denied",
                agent=agent_name,
                reasons=combined,
            )
            return False, combined

        logger.info("validation_gate.promotion_eligible", agent=agent_name)
        return True, "All promotion criteria met"

    # ------------------------------------------------------- demotion check

    async def check_demotion(
        self,
        agent_name: str,
        rolling_sharpe: float,
    ) -> bool:
        """Check if a LIVE agent should be auto-demoted to PAPER.

        Parameters
        ----------
        agent_name:
            Agent to evaluate.
        rolling_sharpe:
            Rolling 20-day Sharpe ratio computed externally.

        Returns
        -------
        bool
            ``True`` if the agent was demoted.
        """
        record = await self._get_record(agent_name)
        if record is None:
            return False

        current = AgentStatus(record.status)
        if current != AgentStatus.LIVE:
            return False

        if rolling_sharpe < _DEMOTION_SHARPE_THRESHOLD:
            record.status = AgentStatus.DEMOTED.value
            record.updated_at = datetime.now(timezone.utc)
            await self._session.commit()
            logger.warning(
                "validation_gate.agent_DEMOTED",
                agent=agent_name,
                rolling_sharpe=rolling_sharpe,
                threshold=_DEMOTION_SHARPE_THRESHOLD,
            )
            return True

        return False

    # -------------------------------------------------------- internals

    async def _get_record(self, agent_name: str) -> AgentRecord | None:
        """Fetch the ``AgentRecord`` for *agent_name*, or ``None``."""
        stmt = select(AgentRecord).where(AgentRecord.agent_name == agent_name)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_or_create(self, agent_name: str) -> AgentRecord:
        """Return an existing record or create a new one in BACKTEST status."""
        record = await self._get_record(agent_name)
        if record is not None:
            return record

        record = AgentRecord(
            agent_name=agent_name,
            status=AgentStatus.BACKTEST.value,
            updated_at=datetime.now(timezone.utc),
            paper_days=0,
            paper_total_trades=0,
        )
        self._session.add(record)
        await self._session.flush()
        logger.info(
            "validation_gate.agent_created",
            agent=agent_name,
            status=AgentStatus.BACKTEST.value,
        )
        return record
