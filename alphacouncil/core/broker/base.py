"""Abstract base class for all broker adapters.

Every concrete broker (paper, Angel One, Fyers, ...) must implement every
method defined here.  The interface is fully async so that I/O-bound calls
(HTTP, WebSocket) never block the event loop.
"""

from __future__ import annotations

import abc
from datetime import datetime, time
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
import structlog

if TYPE_CHECKING:
    from alphacouncil.core.models import Order, Position

from alphacouncil.core.config import IST, get_settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class BrokerError(Exception):
    """Base exception for all broker-related errors."""


class OrderRejectedError(BrokerError):
    """Raised when the broker rejects an order."""


class ConnectionError(BrokerError):  # noqa: A001  -- intentionally shadow built-in
    """Raised when the broker connection cannot be established."""


class AuthenticationError(BrokerError):
    """Raised when broker authentication fails."""


class RateLimitError(BrokerError):
    """Raised when the broker rate-limits the client."""


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------


class BrokerAdapter(abc.ABC):
    """Unified async interface that every broker adapter must implement.

    Parameters
    ----------
    name:
        Human-readable name for the adapter (used in logs).
    """

    def __init__(self, name: str = "broker") -> None:
        self.name = name
        self._log = structlog.get_logger(broker=name)

    # -- Order management ----------------------------------------------------

    @abc.abstractmethod
    async def place_order(self, order: Order) -> str:
        """Submit *order* to the exchange and return the broker order-id.

        Raises
        ------
        OrderRejectedError
            If the broker rejects the order (insufficient funds, invalid
            parameters, etc.).
        """

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.  Return ``True`` if successfully cancelled."""

    # -- Portfolio queries ---------------------------------------------------

    @abc.abstractmethod
    async def get_positions(self) -> list[Position]:
        """Return all open intraday + carryforward positions."""

    @abc.abstractmethod
    async def get_holdings(self) -> list[dict[str, Any]]:
        """Return delivery holdings (T+1 settled stocks)."""

    @abc.abstractmethod
    async def get_funds(self) -> dict[str, float]:
        """Return available funds.

        Expected keys: ``cash``, ``margin``, ``deployed``, ``total``.
        """

    # -- Market data ---------------------------------------------------------

    @abc.abstractmethod
    async def stream_prices(
        self,
        tickers: list[str],
        callback: Callable[[str, float], Any],
    ) -> None:
        """Stream live prices for *tickers*, invoking *callback(ticker, price)*
        on each update.

        This method is expected to run indefinitely (or until cancelled).
        """

    @abc.abstractmethod
    async def get_historical(
        self,
        ticker: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for *ticker*.

        Parameters
        ----------
        interval:
            Candle interval string, e.g. ``"1d"``, ``"1h"``, ``"5m"``.
        from_date, to_date:
            Date range for the query (inclusive).

        Returns
        -------
        pd.DataFrame
            Columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
            Index: ``datetime`` (timezone-aware, IST).
        """

    @abc.abstractmethod
    async def get_ltp(self, ticker: str) -> float:
        """Return the last traded price for *ticker*.

        Raises
        ------
        BrokerError
            If the price cannot be retrieved.
        """

    # -- Utility -------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Check whether the Indian equity market is currently open.

        Returns ``True`` during:
        * Regular session: 09:15 -- 15:30 IST
        * AMO window:      15:45 IST (today) -- 08:57 IST (next morning)
        """
        settings = get_settings()
        now_ist = datetime.now(tz=IST).time()

        regular_open: time = settings.MARKET_OPEN   # 09:15
        regular_close: time = settings.MARKET_CLOSE  # 15:30
        amo_start: time = settings.AMO_START         # 15:45
        amo_end: time = settings.AMO_END             # 08:57 (next day)

        # Regular session
        if regular_open <= now_ist <= regular_close:
            return True

        # AMO window spans midnight: 15:45 .. 23:59 and 00:00 .. 08:57
        if now_ist >= amo_start or now_ist <= amo_end:
            return True

        return False

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        """Authenticate and establish connection (no-op by default)."""
        self._log.info("broker.connect", broker=self.name)

    async def disconnect(self) -> None:
        """Tear down connections gracefully (no-op by default)."""
        self._log.info("broker.disconnect", broker=self.name)

    # -- Dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
