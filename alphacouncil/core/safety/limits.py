"""Pre-trade position and risk limits.

Every order flows through :meth:`PositionLimits.check_order` before reaching
the broker.  The gate enforces hard limits designed to prevent catastrophic
concentration, over-deployment, or regulatory violations on the Indian
exchanges (NSE cash segment).

All thresholds are configurable via the ``Config`` object but ship with
conservative defaults suitable for retail / small-fund trading.
"""

from __future__ import annotations

import zoneinfo
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Protocol, runtime_checkable

import structlog

from alphacouncil.core.models import (
    Exchange,
    Order,
    OrderSide,
    OrderType,
    PortfolioState,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# IST timezone
# ---------------------------------------------------------------------------

_IST = zoneinfo.ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Trading-hours boundaries (IST)
# ---------------------------------------------------------------------------

_REGULAR_OPEN = time(9, 15)
_REGULAR_CLOSE = time(15, 30)

# AMO window: 15:45 same day -> 08:57 next morning
_AMO_START = time(15, 45)
_AMO_END = time(8, 57)

# ---------------------------------------------------------------------------
# Config protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Config(Protocol):
    """Configuration surface consumed by position limits.

    All attributes have documented defaults; implementations may omit any
    attribute and the class will fall back to the default.
    """

    capital: float

    max_per_stock_pct: float       # default 0.05  (5%)
    max_per_sector_pct: float      # default 0.25  (25%)
    max_deployed_pct: float        # default 0.80  (80%)
    max_open_positions: int        # default 15
    max_daily_trades: int          # default 50
    max_order_value: float         # default 50_000 INR


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def _pass() -> tuple[bool, str]:
    return True, ""


def _fail(reason: str) -> tuple[bool, str]:
    return False, reason


# ---------------------------------------------------------------------------
# PositionLimits
# ---------------------------------------------------------------------------


class PositionLimits:
    """Pre-trade risk gate enforcing position-level and portfolio-level limits.

    Parameters
    ----------
    config:
        System configuration providing ``capital`` and optional threshold
        overrides.
    """

    def __init__(self, config: Config) -> None:
        self._capital: float = config.capital

        # Configurable thresholds with conservative defaults
        self._max_per_stock_pct: float = getattr(config, "max_per_stock_pct", 0.05)
        self._max_per_sector_pct: float = getattr(config, "max_per_sector_pct", 0.25)
        self._max_deployed_pct: float = getattr(config, "max_deployed_pct", 0.80)
        self._max_open_positions: int = getattr(config, "max_open_positions", 15)
        self._max_daily_trades: int = getattr(config, "max_daily_trades", 50)
        self._max_order_value: float = getattr(config, "max_order_value", 50_000.0)

        # Internal daily-trade counter (reset externally at start of day)
        self._daily_trade_count: int = 0

        logger.info(
            "position_limits.initialised",
            capital=self._capital,
            max_per_stock_pct=self._max_per_stock_pct,
            max_deployed_pct=self._max_deployed_pct,
            max_open_positions=self._max_open_positions,
        )

    # ----------------------------------------------------------- public API

    async def check_order(
        self,
        order: Order,
        portfolio: PortfolioState,
    ) -> tuple[bool, str]:
        """Validate *order* against all pre-trade limits.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` when the order passes all checks.
            ``(False, reason)`` with a human-readable explanation on failure.
        """
        # 1. Order value cap
        order_value = self._estimate_order_value(order)
        if order_value > self._max_order_value:
            return _fail(
                f"Order value Rs {order_value:,.0f} exceeds per-order "
                f"limit of Rs {self._max_order_value:,.0f}"
            )

        # 2. Per-stock concentration
        existing_value = self._ticker_exposure(order.ticker, portfolio)
        projected = existing_value + order_value if order.side == OrderSide.BUY else existing_value
        stock_limit = self._capital * self._max_per_stock_pct
        if projected > stock_limit:
            return _fail(
                f"Projected exposure in {order.ticker} (Rs {projected:,.0f}) "
                f"exceeds {self._max_per_stock_pct:.0%} capital limit "
                f"(Rs {stock_limit:,.0f})"
            )

        # 3. Per-sector concentration
        sector = self._get_sector_for_ticker(order.ticker, portfolio)
        if sector is not None:
            sector_value = self._sector_exposure(sector, portfolio)
            projected_sector = (
                sector_value + order_value if order.side == OrderSide.BUY else sector_value
            )
            sector_limit = self._capital * self._max_per_sector_pct
            if projected_sector > sector_limit:
                return _fail(
                    f"Projected sector '{sector}' exposure "
                    f"(Rs {projected_sector:,.0f}) exceeds "
                    f"{self._max_per_sector_pct:.0%} capital limit "
                    f"(Rs {sector_limit:,.0f})"
                )

        # 4. Maximum deployment (cash buffer)
        if order.side == OrderSide.BUY:
            projected_deployed = portfolio.deployed_pct + (order_value / portfolio.total_value)
            if projected_deployed > self._max_deployed_pct:
                return _fail(
                    f"Projected deployment {projected_deployed:.1%} exceeds "
                    f"max {self._max_deployed_pct:.0%} "
                    f"({1 - self._max_deployed_pct:.0%} cash buffer required)"
                )

        # 5. Maximum open positions
        if order.side == OrderSide.BUY:
            # Check if this ticker already has a position; if not, it's a new one
            existing_tickers = {pos.ticker for pos in portfolio.positions}
            if order.ticker not in existing_tickers:
                if len(portfolio.positions) >= self._max_open_positions:
                    return _fail(
                        f"Already at maximum of {self._max_open_positions} "
                        f"open positions"
                    )

        # 6. Daily trade count
        if self._daily_trade_count >= self._max_daily_trades:
            return _fail(
                f"Daily trade limit of {self._max_daily_trades} reached"
            )

        # 7. No short selling in NSE cash segment
        if order.exchange == Exchange.NSE and order.side == OrderSide.SELL:
            existing_qty = self._ticker_quantity(order.ticker, portfolio)
            if order.quantity > existing_qty:
                return _fail(
                    "Short selling not permitted in NSE cash segment. "
                    f"Order qty={order.quantity}, held qty={existing_qty}"
                )

        # 8. Trading hours check
        hours_ok, hours_reason = self._check_trading_hours(order)
        if not hours_ok:
            return _fail(hours_reason)

        # All checks passed -- increment counter
        self._daily_trade_count += 1
        logger.debug(
            "position_limits.order_passed",
            ticker=order.ticker,
            side=order.side.value,
            order_value=order_value,
            daily_trades=self._daily_trade_count,
        )
        return _pass()

    async def get_utilization(self, portfolio: PortfolioState) -> dict[str, float]:
        """Return percentage utilization of each limit.

        Values are in the range ``[0.0, 1.0+]`` where > 1.0 means the limit
        is already breached.
        """
        max_stock_exposure = 0.0
        for pos in portfolio.positions:
            exposure = abs(pos.quantity * pos.current_price) / self._capital
            max_stock_exposure = max(max_stock_exposure, exposure)

        # Sector utilization: find the most concentrated sector
        sector_totals: dict[str, float] = {}
        for pos in portfolio.positions:
            sector = pos.sector or "Unknown"
            value = abs(pos.quantity * pos.current_price)
            sector_totals[sector] = sector_totals.get(sector, 0.0) + value
        max_sector_exposure = (
            max(sector_totals.values()) / self._capital if sector_totals else 0.0
        )

        return {
            "per_stock_pct": max_stock_exposure / self._max_per_stock_pct if self._max_per_stock_pct else 0.0,
            "per_sector_pct": max_sector_exposure / self._max_per_sector_pct if self._max_per_sector_pct else 0.0,
            "deployed_pct": portfolio.deployed_pct / self._max_deployed_pct if self._max_deployed_pct else 0.0,
            "open_positions": len(portfolio.positions) / self._max_open_positions if self._max_open_positions else 0.0,
            "daily_trades": self._daily_trade_count / self._max_daily_trades if self._max_daily_trades else 0.0,
        }

    def reset_daily_counter(self) -> None:
        """Reset the daily trade counter.  Call at the start of each session."""
        self._daily_trade_count = 0
        logger.info("position_limits.daily_counter_reset")

    # -------------------------------------------------------- internal helpers

    @staticmethod
    def _estimate_order_value(order: Order) -> float:
        """Estimate the notional value of an order in INR."""
        if order.price is not None and order.price > 0:
            return order.quantity * order.price
        # For market orders without a price hint, use trigger_price as fallback
        if order.trigger_price is not None and order.trigger_price > 0:
            return order.quantity * order.trigger_price
        # Cannot reliably estimate -- return 0 and let other checks catch it.
        # In production the orchestrator should always populate a price estimate.
        logger.warning(
            "position_limits.no_price_estimate",
            order_id=order.order_id,
            ticker=order.ticker,
        )
        return 0.0

    @staticmethod
    def _ticker_exposure(ticker: str, portfolio: PortfolioState) -> float:
        """Current notional exposure in *ticker* across all positions."""
        total = 0.0
        for pos in portfolio.positions:
            if pos.ticker == ticker:
                total += abs(pos.quantity * pos.current_price)
        return total

    @staticmethod
    def _ticker_quantity(ticker: str, portfolio: PortfolioState) -> int:
        """Current held quantity in *ticker*."""
        total = 0
        for pos in portfolio.positions:
            if pos.ticker == ticker:
                total += pos.quantity
        return total

    @staticmethod
    def _get_sector_for_ticker(
        ticker: str,
        portfolio: PortfolioState,
    ) -> str | None:
        """Look up the sector for *ticker* from existing positions."""
        for pos in portfolio.positions:
            if pos.ticker == ticker and pos.sector:
                return pos.sector
        return None

    @staticmethod
    def _sector_exposure(sector: str, portfolio: PortfolioState) -> float:
        """Total notional exposure in a given *sector*."""
        total = 0.0
        for pos in portfolio.positions:
            if pos.sector == sector:
                total += abs(pos.quantity * pos.current_price)
        return total

    @staticmethod
    def _check_trading_hours(order: Order) -> tuple[bool, str]:
        """Validate that the order is placed within permitted trading windows.

        Regular orders: 09:15 -- 15:30 IST
        AMO orders:     15:45 -- 08:57 IST (next morning)
        """
        now_ist = datetime.now(_IST).time()

        if order.order_type == OrderType.AMO:
            # AMO is valid from 15:45 to 08:57 (wraps midnight)
            if _AMO_START <= now_ist or now_ist <= _AMO_END:
                return _pass()
            return _fail(
                f"AMO orders permitted 15:45-08:57 IST, "
                f"current time: {now_ist:%H:%M:%S}"
            )

        # Regular orders
        if _REGULAR_OPEN <= now_ist <= _REGULAR_CLOSE:
            return _pass()

        return _fail(
            f"Regular trading hours are 09:15-15:30 IST, "
            f"current time: {now_ist:%H:%M:%S}. "
            f"Use AMO order type for after-market orders."
        )
