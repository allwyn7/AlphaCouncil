"""Core domain models and configuration for AlphaCouncil."""

# Config import is optional — pydantic-settings may not be installed
# in lightweight deployments (e.g. Streamlit Cloud advisor-only).
try:
    from alphacouncil.core.config import Settings, get_settings  # noqa: F401
except ImportError:
    Settings = None  # type: ignore[assignment,misc]
    get_settings = None  # type: ignore[assignment]

from alphacouncil.core.models import (  # noqa: F401
    Action,
    AgentSignal,
    AgentStatus,
    Exchange,
    FundamentalSignal,
    LatencyLog,
    MacroSignal,
    MarketRegime,
    Order,
    OrderSide,
    OrderType,
    PortfolioState,
    Position,
    ResearchLog,
    SentimentSignal,
    TechnicalSignal,
    TradeRecord,
    VolatilityRegime,
)

__all__ = [
    # Enums
    "Action",
    "AgentStatus",
    "Exchange",
    "MarketRegime",
    "OrderSide",
    "OrderType",
    "VolatilityRegime",
    # Signals
    "AgentSignal",
    "FundamentalSignal",
    "MacroSignal",
    "SentimentSignal",
    "TechnicalSignal",
    # Trading
    "Order",
    "Position",
    "PortfolioState",
    "TradeRecord",
    # Logging
    "LatencyLog",
    "ResearchLog",
    # Config
    "Settings",
    "get_settings",
]
