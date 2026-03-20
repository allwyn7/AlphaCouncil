"""Performance profiler for AlphaCouncil.

Captures nanosecond-precision latencies via :func:`time.perf_counter_ns` and
exposes them as context managers **and** decorators.  Measurements are buffered
in memory and flushed to the ``latency_logs`` SQLite table periodically (every
*flush_every* entries, default 100).

Usage::

    from alphacouncil.core.profiler import Profiler

    profiler = Profiler()

    # --- as an async context manager ---
    async with profiler.measure("price_fetch"):
        price = await api.get_price("INFY")

    # --- as a decorator (sync or async) ---
    @profiler.track
    async def compute_signal(data):
        ...

    # --- query stats ---
    profiler.get_stats("price_fetch")   # {"p50": ..., "p95": ..., "p99": ...}
    profiler.get_all_stats()            # dict[stage, stats]
"""

from __future__ import annotations

import asyncio
import functools
import logging
import statistics
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, TypeVar, overload

from sqlalchemy import insert

from alphacouncil.core.database import get_engine, latency_logs

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Internal data
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _LatencyRecord:
    """A single latency measurement."""

    stage: str
    latency_ns: int
    timestamp: float = field(default_factory=time.time)
    metadata_json: str | None = None


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class Profiler:
    """Nanosecond-resolution profiler with SQLite persistence.

    Parameters
    ----------
    flush_every:
        Number of records to buffer before flushing to SQLite.
    db_url:
        SQLAlchemy connection string.  Defaults to the project-wide SQLite
        database.  Set to ``None`` to disable SQLite persistence entirely
        (records stay only in-memory).
    """

    def __init__(
        self,
        flush_every: int = 100,
        db_url: str | None = "sqlite:///data/alphacouncil.db",
    ) -> None:
        self._flush_every = flush_every
        self._db_url = db_url

        # In-memory buffer awaiting flush
        self._buffer: list[_LatencyRecord] = []

        # Permanent in-memory store keyed by stage name (for stats queries).
        # Each list holds latency values **in nanoseconds**.
        self._history: defaultdict[str, list[int]] = defaultdict(list)

        # Lock to protect buffer flush operations
        self._flush_lock = asyncio.Lock()

        logger.info(
            "Profiler initialised  flush_every=%d  db_url=%s",
            flush_every,
            db_url,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def measure(
        self,
        stage: str,
        *,
        metadata_json: str | None = None,
    ) -> AsyncIterator[None]:
        """Async context manager that records wall-clock latency.

        Parameters
        ----------
        stage:
            Logical name for the operation being measured (e.g.
            ``"price_fetch"``).
        metadata_json:
            Optional JSON string with extra context to persist.

        Example::

            async with profiler.measure("price_fetch"):
                await fetch_prices()
        """
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = time.perf_counter_ns() - start
            await self._record(stage, elapsed_ns, metadata_json)

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    @overload
    def track(self, func: F) -> F: ...

    @overload
    def track(self, *, stage: str | None = None) -> Callable[[F], F]: ...

    def track(
        self,
        func: F | None = None,
        *,
        stage: str | None = None,
    ) -> F | Callable[[F], F]:
        """Decorator that records latency for every invocation.

        Works with both sync and async callables.  The stage name defaults
        to the qualified function name.

        Usage::

            @profiler.track
            async def compute_signal(data): ...

            @profiler.track(stage="custom_name")
            def sync_helper(x): ...
        """

        def decorator(fn: F) -> F:
            _stage = stage or fn.__qualname__

            if asyncio.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start = time.perf_counter_ns()
                    try:
                        return await fn(*args, **kwargs)
                    finally:
                        elapsed_ns = time.perf_counter_ns() - start
                        await self._record(_stage, elapsed_ns)

                return async_wrapper  # type: ignore[return-value]
            else:
                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start = time.perf_counter_ns()
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        elapsed_ns = time.perf_counter_ns() - start
                        # Schedule recording without blocking
                        self._record_sync(_stage, elapsed_ns)

                return sync_wrapper  # type: ignore[return-value]

        if func is not None:
            # Called as @profiler.track (no parentheses)
            return decorator(func)
        # Called as @profiler.track(stage="...")
        return decorator

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, stage: str) -> dict[str, float]:
        """Return p50 / p95 / p99 latencies (in nanoseconds) for *stage*.

        Returns an empty dict if no measurements have been recorded for
        *stage*.
        """
        values = self._history.get(stage)
        if not values:
            return {}
        return self._compute_percentiles(values)

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Return percentile stats for every recorded stage."""
        return {
            stage: self._compute_percentiles(values)
            for stage, values in sorted(self._history.items())
            if values
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_percentiles(values: list[int]) -> dict[str, float]:
        """Compute p50 / p95 / p99 from a list of nanosecond latencies."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n == 0:
            return {}

        def _percentile(p: float) -> float:
            """Simple nearest-rank percentile."""
            k = max(0, min(int(p / 100.0 * n + 0.5) - 1, n - 1))
            return float(sorted_vals[k])

        return {
            "count": float(n),
            "p50": _percentile(50),
            "p95": _percentile(95),
            "p99": _percentile(99),
            "min": float(sorted_vals[0]),
            "max": float(sorted_vals[-1]),
            "mean": statistics.mean(sorted_vals),
        }

    async def _record(
        self,
        stage: str,
        latency_ns: int,
        metadata_json: str | None = None,
    ) -> None:
        """Buffer a measurement and flush when the buffer is full."""
        record = _LatencyRecord(
            stage=stage,
            latency_ns=latency_ns,
            metadata_json=metadata_json,
        )
        self._history[stage].append(latency_ns)
        self._buffer.append(record)

        if len(self._buffer) >= self._flush_every:
            await self._flush()

    def _record_sync(
        self,
        stage: str,
        latency_ns: int,
        metadata_json: str | None = None,
    ) -> None:
        """Non-async variant used by the sync-function decorator.

        If an event loop is running, the flush is scheduled as a task;
        otherwise records simply accumulate until an explicit
        :meth:`flush` call.
        """
        record = _LatencyRecord(
            stage=stage,
            latency_ns=latency_ns,
            metadata_json=metadata_json,
        )
        self._history[stage].append(latency_ns)
        self._buffer.append(record)

        if len(self._buffer) >= self._flush_every:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._flush())
            except RuntimeError:
                # No running loop -- caller can flush manually later
                pass

    async def _flush(self) -> None:
        """Write buffered records to the ``latency_logs`` SQLite table."""
        async with self._flush_lock:
            if not self._buffer:
                return

            to_flush = self._buffer.copy()
            self._buffer.clear()

            if self._db_url is None:
                return  # persistence disabled

            loop = asyncio.get_running_loop()

            def _write() -> None:
                engine = get_engine(self._db_url)  # type: ignore[arg-type]
                rows = [
                    {
                        "stage": r.stage,
                        "latency_ns": r.latency_ns,
                        "metadata_json": r.metadata_json,
                    }
                    for r in to_flush
                ]
                with engine.begin() as conn:
                    conn.execute(insert(latency_logs), rows)

            try:
                await loop.run_in_executor(None, _write)
                logger.debug("Flushed %d latency records to SQLite", len(to_flush))
            except Exception:
                logger.exception("Failed to flush latency records to SQLite")
                # Put records back so they are not lost
                self._buffer.extend(to_flush)

    async def flush(self) -> None:
        """Manually flush all buffered records to SQLite.

        Useful at shutdown or in tests.
        """
        await self._flush()

    @property
    def pending(self) -> int:
        """Number of records waiting to be flushed."""
        return len(self._buffer)
