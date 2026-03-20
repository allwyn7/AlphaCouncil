"""Safety rails for the AlphaCouncil trading system.

This package provides four complementary safety layers:

1. **KillSwitch** -- emergency circuit breaker that cancels all orders,
   flattens positions, and disables agents when loss / error thresholds are
   breached.
2. **PositionLimits** -- pre-trade gate that enforces per-stock, per-sector,
   portfolio-level, and regulatory limits before every order.
3. **ValidationGate** -- agent lifecycle manager that governs promotion from
   BACKTEST -> PAPER -> LIVE and auto-demotion when performance degrades.
4. **AuditTrail** -- append-only immutable logging of every order, kill-switch
   activation, and agent promotion/demotion for regulatory compliance.
"""

from alphacouncil.core.safety.audit import AuditTrail
from alphacouncil.core.safety.kill_switch import KillSwitch
from alphacouncil.core.safety.limits import PositionLimits
from alphacouncil.core.safety.validation_gate import ValidationGate

__all__ = [
    "AuditTrail",
    "KillSwitch",
    "PositionLimits",
    "ValidationGate",
]
