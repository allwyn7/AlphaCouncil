"""Risk Manager -- centralised pre- and post-trade risk enforcement.

The :class:`RiskManager` wraps the safety primitives (:class:`PositionLimits`,
:class:`KillSwitch`, :class:`AuditTrail`) into a single facade that the
:class:`~alphacouncil.agents.meta.MetaAgent` calls during every council cycle.

Responsibilities
----------------
1. **Pre-trade validation** -- each order is checked against
   :meth:`PositionLimits.check_order` for concentration, deployment, and
   regulatory limits.
2. **Post-trade monitoring** -- after every batch of executions the
   :class:`KillSwitch` auto-trigger conditions are evaluated.
3. **Risk summary** -- a snapshot of current risk metrics is available for
   dashboards and logging.
"""

from __future__ import annotations

from typing import Any

import structlog

from alphacouncil.core.models import (
    Order,
    PortfolioState,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight protocols for type safety without import coupling
# ---------------------------------------------------------------------------


class _PositionLimitsProtocol:
    """Structural sub-type of PositionLimits used for type hints."""

    async def check_order(
        self, order: Order, portfolio: PortfolioState,
    ) -> tuple[bool, str]: ...

    async def get_utilization(
        self, portfolio: PortfolioState,
    ) -> dict[str, float]: ...


class _KillSwitchProtocol:
    """Structural sub-type of KillSwitch used for type hints."""

    is_active: bool

    async def check_auto_triggers(
        self,
        portfolio: PortfolioState,
        error_count: int = 0,
        latencies: list[float] | None = None,
    ) -> None: ...


class _AuditTrailProtocol:
    """Structural sub-type of AuditTrail used for type hints."""

    async def log_order_validation(
        self,
        order: Order,
        passed: bool,
        reason: str,
    ) -> None: ...


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


class RiskManager:
    """Centralised risk enforcement facade.

    Parameters
    ----------
    limits:
        A :class:`~alphacouncil.core.safety.limits.PositionLimits` instance
        for pre-trade order validation.
    kill_switch:
        A :class:`~alphacouncil.core.safety.kill_switch.KillSwitch` instance
        for post-trade circuit-breaking.
    audit:
        An :class:`~alphacouncil.core.safety.audit.AuditTrail` instance for
        immutable logging (optional; pass ``None`` to skip audit logging).
    config:
        System-wide :class:`~alphacouncil.core.config.Settings`.
    """

    def __init__(
        self,
        limits: Any,
        kill_switch: Any,
        audit: Any | None,
        config: Any,
    ) -> None:
        self._limits = limits
        self._kill_switch = kill_switch
        self._audit = audit
        self._config = config
        self._log = logger.bind(component="risk_manager")

        # Running error counter for kill-switch escalation.
        self._error_count: int = 0

        self._log.info(
            "risk_manager.initialised",
            kill_switch_active=getattr(kill_switch, "is_active", False),
        )

    # ------------------------------------------------------------------
    # Pre-trade validation
    # ------------------------------------------------------------------

    async def validate_orders(
        self,
        orders: list[Order],
        portfolio: PortfolioState,
    ) -> list[tuple[Order, bool, str]]:
        """Validate a batch of orders against all pre-trade risk limits.

        Each order is independently checked through
        :meth:`PositionLimits.check_order`.  If the kill switch is already
        active, **all** orders are rejected immediately.

        Parameters
        ----------
        orders:
            List of :class:`Order` tickets to validate.
        portfolio:
            Current portfolio state used for limit calculations.

        Returns
        -------
        list[tuple[Order, bool, str]]
            For each input order: ``(order, passed, reason)``.
            ``passed`` is ``True`` when the order cleared all checks;
            ``reason`` is an empty string on pass or a human-readable
            explanation on rejection.
        """
        results: list[tuple[Order, bool, str]] = []

        # Fast-path: kill switch active -> reject everything.
        if getattr(self._kill_switch, "is_active", False):
            self._log.warning(
                "risk_manager.kill_switch_active_rejecting_all",
                order_count=len(orders),
            )
            for order in orders:
                reason = "Kill switch is active -- all orders rejected"
                results.append((order, False, reason))
                await self._audit_order(order, False, reason)
            return results

        # Validate each order individually.
        for order in orders:
            try:
                passed, reason = await self._limits.check_order(
                    order, portfolio,
                )
            except Exception as exc:
                self._log.exception(
                    "risk_manager.limit_check_error",
                    ticker=order.ticker,
                    order_id=order.order_id,
                )
                passed = False
                reason = f"Internal risk-check error: {exc}"
                self._error_count += 1

            results.append((order, passed, reason))

            # Audit trail.
            await self._audit_order(order, passed, reason)

            # Log per-order result.
            if passed:
                self._log.debug(
                    "risk_manager.order_passed",
                    ticker=order.ticker,
                    side=order.side.value,
                    order_id=order.order_id,
                )
            else:
                self._log.warning(
                    "risk_manager.order_rejected",
                    ticker=order.ticker,
                    side=order.side.value,
                    order_id=order.order_id,
                    reason=reason,
                )

        self._log.info(
            "risk_manager.batch_validated",
            total=len(orders),
            passed=sum(1 for _, p, _ in results if p),
            rejected=sum(1 for _, p, _ in results if not p),
        )
        return results

    # ------------------------------------------------------------------
    # Post-trade check
    # ------------------------------------------------------------------

    async def post_trade_check(self, portfolio: PortfolioState) -> None:
        """Run post-trade kill-switch auto-trigger evaluation.

        This should be called after every batch of order executions to
        detect breaches in daily loss, drawdown, or error-rate thresholds.

        Parameters
        ----------
        portfolio:
            Updated portfolio state reflecting the latest trades.
        """
        try:
            await self._kill_switch.check_auto_triggers(
                portfolio=portfolio,
                error_count=self._error_count,
            )
        except Exception:
            self._log.exception("risk_manager.post_trade_check_error")

        # Reset error counter after each check cycle.
        self._error_count = 0

        if getattr(self._kill_switch, "is_active", False):
            self._log.critical(
                "risk_manager.kill_switch_triggered_post_trade",
                daily_pnl=portfolio.daily_pnl,
                drawdown=portfolio.drawdown_from_peak,
            )

    # ------------------------------------------------------------------
    # Risk summary
    # ------------------------------------------------------------------

    def get_risk_summary(self, portfolio: PortfolioState) -> dict[str, Any]:
        """Return a snapshot of current risk metrics.

        Intended for dashboards, Telegram alerts, and logging.

        Parameters
        ----------
        portfolio:
            Current portfolio state.

        Returns
        -------
        dict[str, Any]
            Keys:
            - ``deployed_pct``: fraction of capital deployed.
            - ``position_count``: number of open positions.
            - ``daily_pnl``: today's P&L in INR.
            - ``daily_pnl_pct``: today's P&L as percentage.
            - ``drawdown_from_peak``: current drawdown fraction.
            - ``cash``: available cash in INR.
            - ``total_value``: portfolio NAV in INR.
            - ``sector_exposure``: dict of sector -> notional value.
            - ``top_positions``: list of (ticker, notional, pnl_pct) tuples.
            - ``kill_switch_active``: whether the kill switch is latched.
        """
        # Sector exposure breakdown.
        sector_exposure: dict[str, float] = {}
        for pos in portfolio.positions:
            sector = pos.sector or "Unknown"
            value = abs(pos.quantity * pos.current_price)
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + value

        # Top positions by notional value.
        position_details: list[dict[str, Any]] = []
        for pos in portfolio.positions:
            notional = abs(pos.quantity * pos.current_price)
            position_details.append({
                "ticker": pos.ticker,
                "quantity": pos.quantity,
                "notional": round(notional, 2),
                "pnl": round(pos.pnl, 2),
                "pnl_pct": round(pos.pnl_pct, 4),
            })
        # Sort descending by notional.
        position_details.sort(key=lambda p: p["notional"], reverse=True)

        return {
            "deployed_pct": round(portfolio.deployed_pct, 4),
            "position_count": len(portfolio.positions),
            "daily_pnl": round(portfolio.daily_pnl, 2),
            "daily_pnl_pct": round(portfolio.daily_pnl_pct, 4),
            "drawdown_from_peak": round(portfolio.drawdown_from_peak, 4),
            "cash": round(portfolio.cash, 2),
            "total_value": round(portfolio.total_value, 2),
            "sector_exposure": {
                k: round(v, 2) for k, v in sector_exposure.items()
            },
            "top_positions": position_details[:10],
            "kill_switch_active": getattr(
                self._kill_switch, "is_active", False,
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _audit_order(
        self,
        order: Order,
        passed: bool,
        reason: str,
    ) -> None:
        """Write order validation result to the audit trail.

        Silently returns if no audit trail is configured.
        """
        if self._audit is None:
            return

        try:
            # AuditTrail may not have a specialised method; fall back to
            # a generic log call if the specific method is unavailable.
            log_fn = getattr(self._audit, "log_order_validation", None)
            if log_fn is not None:
                await log_fn(
                    order=order,
                    passed=passed,
                    reason=reason,
                )
            else:
                # Generic fallback.
                generic_fn = getattr(self._audit, "log", None)
                if generic_fn is not None:
                    await generic_fn(
                        actor="risk_manager",
                        action="order_validation",
                        resource=order.order_id,
                        detail=(
                            f"ticker={order.ticker} side={order.side.value} "
                            f"passed={passed} reason={reason}"
                        ),
                        severity="info" if passed else "warning",
                    )
        except Exception:
            self._log.exception(
                "risk_manager.audit_log_failed",
                order_id=order.order_id,
            )

    def __repr__(self) -> str:
        ks_status = "ACTIVE" if getattr(
            self._kill_switch, "is_active", False,
        ) else "ARMED"
        return f"<RiskManager kill_switch={ks_status}>"
