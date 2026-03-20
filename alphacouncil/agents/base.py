"""Abstract base class for all AlphaCouncil trading agents.

Every quant agent in the system inherits from :class:`BaseAgent`, which
provides lifecycle management (status transitions), a uniform signal-
generation contract, cross-sectional scoring utilities, and automatic
publication of signals onto the shared message bus.

Design decisions
----------------
* **MessageBus protocol** -- The bus dependency is expressed as a
  :class:`typing.Protocol` so that agents never import the concrete
  implementation, making testing trivial (inject a mock / in-memory bus).
* **Async-first** -- Signal generation and the run-cycle are ``async`` to
  accommodate IO-bound data fetches inside concrete agents without blocking
  the event loop.
* **Parameter hot-reload** -- :meth:`set_parameters` allows the orchestrator
  (or a self-tuning loop) to update strategy knobs at runtime without
  restarting the agent.
"""

from __future__ import annotations

import time as _time
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import structlog

from alphacouncil.core.models import Action, AgentSignal, AgentStatus

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight protocol for the message bus dependency
# ---------------------------------------------------------------------------


@runtime_checkable
class MessageBus(Protocol):
    """Minimal contract that the shared message bus must satisfy.

    Concrete implementations live in ``alphacouncil.core.bus``; agents only
    depend on this protocol so they remain fully unit-testable in isolation.
    """

    async def publish(self, topic: str, message: Any) -> None:
        """Publish *message* to *topic*."""
        ...

    async def subscribe(self, topic: str, handler: Any) -> None:
        """Register *handler* for messages on *topic*."""
        ...


# ---------------------------------------------------------------------------
# BaseAgent ABC
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base for every trading agent in AlphaCouncil.

    Parameters
    ----------
    name:
        Canonical, human-readable agent name (e.g. ``"growth_momentum"``).
    config:
        System-wide :class:`~alphacouncil.core.config.Settings` instance.
    cache:
        Shared cache backend (dict-like or Redis wrapper).
    bus:
        Message bus satisfying the :class:`MessageBus` protocol.
    db_engine:
        SQLAlchemy :class:`~sqlalchemy.engine.Engine` for persistence.
    """

    # Topic prefix for signal publication on the bus.
    _SIGNAL_TOPIC: str = "signals"

    def __init__(
        self,
        name: str,
        config: Any,
        cache: Any,
        bus: MessageBus,
        db_engine: Any,
    ) -> None:
        self._name: str = name
        self._config = config
        self._cache = cache
        self._bus = bus
        self._db_engine = db_engine
        self._status: AgentStatus = AgentStatus.BACKTEST
        self._parameters: dict[str, Any] = {}
        self._log = logger.bind(agent=name)
        self._log.info("agent.initialised", status=self._status.value)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Canonical agent name."""
        return self._name

    @property
    def status(self) -> AgentStatus:
        """Current lifecycle status."""
        return self._status

    @property
    def parameters(self) -> dict[str, Any]:
        """Current tunable parameters (read-only view)."""
        return dict(self._parameters)

    # ------------------------------------------------------------------
    # Abstract interface -- must be implemented by every concrete agent
    # ------------------------------------------------------------------

    @abstractmethod
    async def generate_signals(
        self,
        universe: list[str],
        market_data: dict[str, Any],
    ) -> list[AgentSignal]:
        """Analyse *market_data* for every ticker in *universe* and return signals.

        Parameters
        ----------
        universe:
            List of NSE ticker symbols to evaluate (e.g. ``["RELIANCE.NS"]``).
        market_data:
            Dictionary of pre-fetched data keyed by data type.  Expected keys
            depend on the concrete agent (e.g. ``"prices"``, ``"fundamentals"``,
            ``"sentiment"``).

        Returns
        -------
        list[AgentSignal]
            Zero or more actionable signals.
        """
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return the agent's current tunable parameters."""
        ...

    @abstractmethod
    def set_parameters(self, params: dict[str, Any]) -> None:
        """Hot-reload tunable parameters from *params*.

        Implementations should validate incoming values and log changes.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    async def run_cycle(
        self,
        universe: list[str],
        market_data: dict[str, Any],
    ) -> list[AgentSignal]:
        """Execute one full analysis cycle: generate signals and publish them.

        This is the main entry point called by the orchestrator on each
        scheduling tick.

        Returns
        -------
        list[AgentSignal]
            Signals produced during this cycle (also published to the bus).
        """
        cycle_start = _time.perf_counter_ns()
        self._log.info("cycle.start", universe_size=len(universe))

        try:
            signals = await self.generate_signals(universe, market_data)
        except Exception:
            self._log.exception("cycle.generate_signals_failed")
            return []

        # Publish each signal to the message bus.
        topic = f"{self._SIGNAL_TOPIC}.{self._name}"
        for sig in signals:
            try:
                await self._bus.publish(topic, sig)
            except Exception:
                self._log.exception(
                    "cycle.publish_failed",
                    ticker=sig.ticker,
                    action=sig.action.value,
                )

        elapsed_ms = (_time.perf_counter_ns() - cycle_start) / 1_000_000
        self._log.info(
            "cycle.complete",
            signals_count=len(signals),
            elapsed_ms=round(elapsed_ms, 2),
        )
        return signals

    # -- Status management ------------------------------------------------

    def get_status(self) -> AgentStatus:
        """Return current lifecycle status."""
        return self._status

    def set_status(self, status: AgentStatus) -> None:
        """Transition the agent to a new lifecycle *status*.

        Transitions are logged for the audit trail.
        """
        old = self._status
        self._status = status
        self._log.info(
            "agent.status_changed",
            old_status=old.value,
            new_status=status.value,
        )

    # -- Cross-sectional scoring utilities --------------------------------

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """Cross-sectional z-score normalisation.

        Returns a z-scored copy of *series*.  If the standard deviation is
        zero (all identical values), returns a Series of zeros to avoid
        division-by-zero.
        """
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index, dtype=np.float64)
        return (series - series.mean()) / std

    @staticmethod
    def _compute_conviction(
        composite_score: float,
        min_score: float = 0.0,
        max_score: float = 2.0,
    ) -> int:
        """Map a continuous *composite_score* to a 0--100 conviction integer.

        Scores at or below *min_score* map to ``0``; scores at or above
        *max_score* map to ``100``.  Values in between are linearly
        interpolated and clamped.
        """
        if max_score <= min_score:
            return 0
        ratio = (composite_score - min_score) / (max_score - min_score)
        return int(np.clip(ratio * 100, 0, 100))
