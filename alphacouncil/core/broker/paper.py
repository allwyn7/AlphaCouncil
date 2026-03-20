"""Simulated (paper) broker for backtesting and paper-trading validation.

``PaperBroker`` keeps all state in memory and uses *yfinance* for price
discovery.  It models realistic Indian equity transaction costs:

* Brokerage: 0.05 % of turnover
* STT: 0.1 % on sell-side turnover
* GST: 18 % on brokerage
* SEBI turnover charges: 0.0001 %
* Stamp duty: 0.015 % on buy side
* Slippage: 0.1 % on market orders

This allows agents to graduate from paper-trading with high-fidelity P&L
numbers before being promoted to live.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import pandas as pd
import structlog
import yfinance as yf

from alphacouncil.core.broker.base import (
    BrokerAdapter,
    BrokerError,
    OrderRejectedError,
)
from alphacouncil.core.config import IST
from alphacouncil.core.models import (
    Exchange,
    Order,
    OrderSide,
    OrderType,
    Position,
    TradeRecord,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal order-book entry
# ---------------------------------------------------------------------------


class _OrderStatus(str, Enum):
    PLACED = "PLACED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class _OrderEntry:
    """Mutable bookkeeping record for the paper order book."""

    __slots__ = (
        "order_id",
        "order",
        "status",
        "fill_price",
        "charges",
        "filled_at",
    )

    def __init__(self, order_id: str, order: Order) -> None:
        self.order_id = order_id
        self.order = order
        self.status: _OrderStatus = _OrderStatus.PLACED
        self.fill_price: float = 0.0
        self.charges: float = 0.0
        self.filled_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "ticker": self.order.ticker,
            "side": self.order.side.value,
            "order_type": self.order.order_type.value,
            "quantity": self.order.quantity,
            "requested_price": self.order.price,
            "fill_price": self.fill_price,
            "charges": self.charges,
            "status": self.status.value,
            "filled_at": self.filled_at,
        }


# ---------------------------------------------------------------------------
# Charges calculator
# ---------------------------------------------------------------------------

_BROKERAGE_PCT = 0.0005       # 0.05 %
_STT_SELL_PCT = 0.001          # 0.1 % on sell
_GST_ON_BROKERAGE_PCT = 0.18  # 18 % of brokerage
_SEBI_PCT = 0.000001           # 0.0001 %
_STAMP_BUY_PCT = 0.00015      # 0.015 % on buy
_SLIPPAGE_PCT = 0.001          # 0.1 % for market orders


def _calculate_charges(
    side: OrderSide,
    turnover: float,
    is_market_order: bool,
) -> tuple[float, float]:
    """Return ``(total_charges, fill_price_adjustment_factor)``.

    ``fill_price_adjustment_factor`` is a multiplier to apply *before*
    charge calculation so that the effective price includes slippage.
    """
    # Slippage on market orders
    slippage_factor = 1.0
    if is_market_order:
        slippage_factor = 1.0 + _SLIPPAGE_PCT if side == OrderSide.BUY else 1.0 - _SLIPPAGE_PCT

    brokerage = turnover * _BROKERAGE_PCT
    gst = brokerage * _GST_ON_BROKERAGE_PCT
    sebi = turnover * _SEBI_PCT

    stt = 0.0
    stamp = 0.0
    if side == OrderSide.SELL:
        stt = turnover * _STT_SELL_PCT
    else:
        stamp = turnover * _STAMP_BUY_PCT

    total = brokerage + gst + stt + sebi + stamp
    return total, slippage_factor


# ---------------------------------------------------------------------------
# PaperBroker
# ---------------------------------------------------------------------------


class PaperBroker(BrokerAdapter):
    """In-memory simulated broker backed by *yfinance* for price discovery.

    Parameters
    ----------
    initial_capital:
        Starting cash balance in INR.
    """

    def __init__(self, initial_capital: float = 1_000_000.0) -> None:
        super().__init__(name="paper")
        self._initial_capital = initial_capital
        self._cash: float = initial_capital

        # ticker -> {"quantity": int, "avg_price": float, "since": datetime}
        self._positions: dict[str, dict[str, Any]] = {}

        # order_id -> _OrderEntry
        self._order_book: dict[str, _OrderEntry] = {}

        # Completed trades (immutable records)
        self._trades: list[TradeRecord] = []

        # Simple LTP cache: ticker -> (price, fetched_at)
        self._price_cache: dict[str, tuple[float, datetime]] = {}
        self._price_cache_ttl = timedelta(seconds=30)

        self._log = structlog.get_logger(broker="paper")

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    async def _fetch_ltp(self, ticker: str) -> float:
        """Fetch last traded price via yfinance with short-term caching."""
        now = datetime.now(tz=IST)
        cached = self._price_cache.get(ticker)
        if cached is not None:
            price, fetched_at = cached
            if now - fetched_at < self._price_cache_ttl:
                return price

        try:
            loop = asyncio.get_running_loop()
            yticker = self._yf_ticker(ticker)
            info = await loop.run_in_executor(None, lambda: yf.Ticker(yticker).fast_info)
            price = float(info["lastPrice"])
        except Exception as exc:
            self._log.error("paper.ltp_fetch_failed", ticker=ticker, error=str(exc))
            raise BrokerError(f"Cannot fetch LTP for {ticker}: {exc}") from exc

        self._price_cache[ticker] = (price, now)
        return price

    @staticmethod
    def _yf_ticker(ticker: str) -> str:
        """Ensure the ticker has a ``.NS`` suffix for NSE on yfinance."""
        if not ticker.endswith((".NS", ".BO")):
            return f"{ticker}.NS"
        return ticker

    # ------------------------------------------------------------------
    # BrokerAdapter: place_order
    # ------------------------------------------------------------------

    async def place_order(self, order: Order) -> str:
        """Simulate order placement with realistic fills and charges.

        Market orders are filled immediately at LTP +/- slippage.
        Limit orders are filled only if the current price satisfies the limit.
        """
        order_id = order.order_id or f"PAPER-{uuid.uuid4().hex[:12].upper()}"
        entry = _OrderEntry(order_id=order_id, order=order)
        self._order_book[order_id] = entry

        self._log.info(
            "paper.order_placed",
            order_id=order_id,
            ticker=order.ticker,
            side=order.side.value,
            qty=order.quantity,
            order_type=order.order_type.value,
            price=order.price,
        )

        try:
            ltp = await self._fetch_ltp(order.ticker)
        except BrokerError:
            entry.status = _OrderStatus.REJECTED
            raise OrderRejectedError(f"Cannot determine price for {order.ticker}")

        is_market = order.order_type == OrderType.MARKET
        fill_price = ltp

        # For limit orders, check fill-ability
        if order.order_type == OrderType.LIMIT:
            if order.price is None:
                entry.status = _OrderStatus.REJECTED
                raise OrderRejectedError("Limit order requires a price")
            if order.side == OrderSide.BUY and order.price < ltp:
                # Limit buy below market -- pend (simplified: reject for now)
                entry.status = _OrderStatus.PLACED
                self._log.info("paper.limit_pending", order_id=order_id, limit=order.price, ltp=ltp)
                return order_id
            if order.side == OrderSide.SELL and order.price > ltp:
                entry.status = _OrderStatus.PLACED
                self._log.info("paper.limit_pending", order_id=order_id, limit=order.price, ltp=ltp)
                return order_id
            fill_price = order.price

        # Apply slippage for market orders
        charges, slippage_factor = _calculate_charges(
            side=order.side,
            turnover=fill_price * order.quantity,
            is_market_order=is_market,
        )
        fill_price *= slippage_factor
        turnover = fill_price * order.quantity

        # Recalculate charges on slipped turnover
        charges, _ = _calculate_charges(
            side=order.side,
            turnover=turnover,
            is_market_order=False,  # slippage already applied
        )

        # Validate funds
        if order.side == OrderSide.BUY:
            required = turnover + charges
            if required > self._cash:
                entry.status = _OrderStatus.REJECTED
                raise OrderRejectedError(
                    f"Insufficient funds: need {required:.2f}, have {self._cash:.2f}"
                )

        # Execute fill
        self._execute_fill(entry, fill_price, charges)
        return order_id

    def _execute_fill(
        self,
        entry: _OrderEntry,
        fill_price: float,
        charges: float,
    ) -> None:
        """Book a fill: update cash, positions, and trade log."""
        order = entry.order
        qty = order.quantity
        turnover = fill_price * qty
        now = datetime.now(tz=IST)

        if order.side == OrderSide.BUY:
            self._cash -= turnover + charges
            pos = self._positions.get(order.ticker)
            if pos is not None:
                # Average up / down
                total_qty = pos["quantity"] + qty
                pos["avg_price"] = (
                    (pos["avg_price"] * pos["quantity"] + fill_price * qty) / total_qty
                )
                pos["quantity"] = total_qty
            else:
                self._positions[order.ticker] = {
                    "quantity": qty,
                    "avg_price": fill_price,
                    "since": now,
                }
        else:
            # SELL
            pos = self._positions.get(order.ticker)
            if pos is None or pos["quantity"] < qty:
                entry.status = _OrderStatus.REJECTED
                raise OrderRejectedError(
                    f"Cannot sell {qty} of {order.ticker}: "
                    f"holding {pos['quantity'] if pos else 0}"
                )
            self._cash += turnover - charges
            pos["quantity"] -= qty
            if pos["quantity"] == 0:
                del self._positions[order.ticker]

        entry.status = _OrderStatus.FILLED
        entry.fill_price = fill_price
        entry.charges = charges
        entry.filled_at = now

        trade = TradeRecord(
            order_id=entry.order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=qty,
            price=fill_price,
            timestamp=now,
            agent_name=order.agent_name,
            factor_scores={},
            reasoning=order.reasoning,
            risk_check_passed=True,
        )
        self._trades.append(trade)

        self._log.info(
            "paper.order_filled",
            order_id=entry.order_id,
            ticker=order.ticker,
            side=order.side.value,
            qty=qty,
            fill_price=round(fill_price, 2),
            charges=round(charges, 2),
            cash_remaining=round(self._cash, 2),
        )

    # ------------------------------------------------------------------
    # BrokerAdapter: cancel_order
    # ------------------------------------------------------------------

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.  Returns ``True`` if cancelled."""
        entry = self._order_book.get(order_id)
        if entry is None:
            self._log.warning("paper.cancel_unknown", order_id=order_id)
            return False
        if entry.status != _OrderStatus.PLACED:
            self._log.warning(
                "paper.cancel_not_pending",
                order_id=order_id,
                current_status=entry.status.value,
            )
            return False

        entry.status = _OrderStatus.CANCELLED
        self._log.info("paper.order_cancelled", order_id=order_id)
        return True

    # ------------------------------------------------------------------
    # BrokerAdapter: positions / holdings / funds
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Return all current simulated positions with live P&L."""
        positions: list[Position] = []
        for ticker, pos in self._positions.items():
            try:
                ltp = await self._fetch_ltp(ticker)
            except BrokerError:
                ltp = pos["avg_price"]  # fallback

            pnl = (ltp - pos["avg_price"]) * pos["quantity"]
            pnl_pct = ((ltp / pos["avg_price"]) - 1.0) * 100.0 if pos["avg_price"] else 0.0

            positions.append(
                Position(
                    ticker=ticker,
                    quantity=pos["quantity"],
                    avg_price=pos["avg_price"],
                    current_price=ltp,
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                )
            )
        return positions

    async def get_holdings(self) -> list[dict[str, Any]]:
        """Return positions held longer than 1 day (delivery holdings)."""
        now = datetime.now(tz=IST)
        holdings: list[dict[str, Any]] = []
        for ticker, pos in self._positions.items():
            held_since: datetime = pos.get("since", now)
            if (now - held_since) >= timedelta(days=1):
                try:
                    ltp = await self._fetch_ltp(ticker)
                except BrokerError:
                    ltp = pos["avg_price"]

                holdings.append({
                    "ticker": ticker,
                    "quantity": pos["quantity"],
                    "avg_price": round(pos["avg_price"], 2),
                    "last_price": round(ltp, 2),
                    "pnl": round((ltp - pos["avg_price"]) * pos["quantity"], 2),
                    "held_since": held_since.isoformat(),
                })
        return holdings

    async def get_funds(self) -> dict[str, float]:
        """Return current cash, deployed capital, and total portfolio value."""
        deployed = sum(
            pos["avg_price"] * pos["quantity"]
            for pos in self._positions.values()
        )
        total = self._cash + deployed
        return {
            "cash": round(self._cash, 2),
            "margin": round(self._cash, 2),  # no margin in paper mode
            "deployed": round(deployed, 2),
            "total": round(total, 2),
        }

    # ------------------------------------------------------------------
    # BrokerAdapter: market data
    # ------------------------------------------------------------------

    async def stream_prices(
        self,
        tickers: list[str],
        callback: Callable[[str, float], Any],
    ) -> None:
        """Poll yfinance every 5 seconds to simulate streaming prices."""
        self._log.info("paper.stream_start", tickers=tickers)
        try:
            while True:
                for ticker in tickers:
                    try:
                        price = await self._fetch_ltp(ticker)
                        callback(ticker, price)
                    except BrokerError:
                        self._log.warning("paper.stream_tick_failed", ticker=ticker)
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            self._log.info("paper.stream_stopped", tickers=tickers)

    async def get_historical(
        self,
        ticker: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV from yfinance.

        The ticker is automatically suffixed with ``.NS`` if needed.
        """
        yticker = self._yf_ticker(ticker)
        self._log.debug(
            "paper.historical",
            ticker=yticker,
            interval=interval,
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
        )

        loop = asyncio.get_running_loop()
        try:
            df: pd.DataFrame = await loop.run_in_executor(
                None,
                lambda: yf.Ticker(yticker).history(
                    start=from_date.strftime("%Y-%m-%d"),
                    end=to_date.strftime("%Y-%m-%d"),
                    interval=interval,
                ),
            )
        except Exception as exc:
            self._log.error("paper.historical_failed", ticker=yticker, error=str(exc))
            raise BrokerError(f"Historical data fetch failed for {yticker}: {exc}") from exc

        if df.empty:
            self._log.warning("paper.historical_empty", ticker=yticker)
            return df

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        # Keep only OHLCV columns if present
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        return df[keep]

    async def get_ltp(self, ticker: str) -> float:
        """Return the last traded price via yfinance ``fast_info``."""
        return await self._fetch_ltp(ticker)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def trades(self) -> list[TradeRecord]:
        """All executed trades (immutable copies)."""
        return list(self._trades)

    @property
    def order_book(self) -> list[dict[str, Any]]:
        """Full order book as a list of dicts."""
        return [e.to_dict() for e in self._order_book.values()]

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    def summary(self) -> dict[str, Any]:
        """Quick portfolio summary for dashboards / logging."""
        deployed = sum(
            pos["avg_price"] * pos["quantity"]
            for pos in self._positions.values()
        )
        total = self._cash + deployed
        return {
            "initial_capital": self._initial_capital,
            "cash": round(self._cash, 2),
            "deployed": round(deployed, 2),
            "total": round(total, 2),
            "pnl": round(total - self._initial_capital, 2),
            "pnl_pct": round(((total / self._initial_capital) - 1) * 100, 2),
            "num_positions": len(self._positions),
            "num_trades": len(self._trades),
        }

    def __repr__(self) -> str:
        return (
            f"<PaperBroker cash={self._cash:,.2f} "
            f"positions={len(self._positions)} "
            f"trades={len(self._trades)}>"
        )
