"""Pydantic v2 domain models for the AlphaCouncil trading system.

Every model that represents *immutable data flowing through the pipeline* uses
``frozen=True`` so instances are hashable and safe for use as dict keys,
lru_cache arguments, and set members.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AgentStatus(str, enum.Enum):
    """Lifecycle stage of a trading agent."""

    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"
    DEMOTED = "DEMOTED"


class Action(str, enum.Enum):
    """Directional action produced by an agent."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(str, enum.Enum):
    """Supported order types on Angel One / NSE."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"          # Stop-Loss Limit
    SL_M = "SL_M"      # Stop-Loss Market
    AMO = "AMO"         # After-Market Order


class Exchange(str, enum.Enum):
    """Exchange segment."""

    NSE = "NSE"
    NFO = "NFO"


class OrderSide(str, enum.Enum):
    """Side of an order."""

    BUY = "BUY"
    SELL = "SELL"


class MarketRegime(str, enum.Enum):
    """Macro-level market regime classification for India."""

    BULL_LOW_VOL = "BULL_LOW_VOL"
    BULL_HIGH_VOL = "BULL_HIGH_VOL"
    BEAR_LOW_VOL = "BEAR_LOW_VOL"
    BEAR_HIGH_VOL = "BEAR_HIGH_VOL"
    SIDEWAYS = "SIDEWAYS"
    FII_BUYING = "FII_BUYING"
    FII_SELLING = "FII_SELLING"
    PRE_EXPIRY = "PRE_EXPIRY"
    EARNINGS_SEASON = "EARNINGS_SEASON"
    BUDGET_POLICY = "BUDGET_POLICY"


class VolatilityRegime(str, enum.Enum):
    """Short-term volatility classification (India VIX-derived)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    TRANSITION = "TRANSITION"


# ---------------------------------------------------------------------------
# Signal models (frozen / hashable)
# ---------------------------------------------------------------------------


class AgentSignal(BaseModel, frozen=True):
    """Signal emitted by any trading agent after analysis."""

    ticker: str = Field(..., description="NSE ticker symbol, e.g. 'RELIANCE.NS'")
    action: Action
    conviction: int = Field(..., ge=0, le=100, description="Confidence 0-100")
    target_weight: float = Field(
        ..., ge=0.0, le=1.0,
        description="Desired portfolio weight for this ticker",
    )
    stop_loss: float = Field(..., gt=0.0, description="Stop-loss price level")
    take_profit: float = Field(..., gt=0.0, description="Take-profit price level")
    factor_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Named factor scores backing this signal",
    )
    reasoning: str = Field(
        ..., min_length=1,
        description="Human-readable rationale for the signal",
    )
    holding_period_days: int = Field(
        ..., ge=1,
        description="Expected holding period in calendar days",
    )
    agent_name: str = Field(
        ..., min_length=1,
        description="Canonical name of the agent that produced this signal",
    )
    timestamp: datetime


class TechnicalSignal(BaseModel, frozen=True):
    """Snapshot of technical indicators for a single ticker at a point in time."""

    ticker: str

    # Momentum
    rsi: float = Field(..., description="Relative Strength Index (14)")
    macd: float
    macd_signal: float
    macd_hist: float
    roc: float = Field(..., description="Rate of Change")

    # Bollinger Bands
    bollinger_upper: float
    bollinger_lower: float
    bollinger_mid: float

    # Moving averages -- simple
    sma_20: float
    sma_50: float
    sma_200: float

    # Moving averages -- exponential
    ema_20: float
    ema_50: float
    ema_200: float

    # Trend / volatility
    adx: float = Field(..., description="Average Directional Index")
    atr: float = Field(..., description="Average True Range")

    # Volume
    obv: float = Field(..., description="On-Balance Volume")
    vwap: float = Field(..., description="Volume Weighted Average Price")
    volume_ratio: float = Field(
        ..., description="Current volume / 20-day average volume",
    )

    timestamp: datetime


class FundamentalSignal(BaseModel, frozen=True):
    """Fundamental / valuation snapshot for a single ticker."""

    ticker: str

    # Valuation
    pe_ratio: float = Field(..., description="Price-to-Earnings (TTM)")
    peg_ratio: float = Field(..., description="Price/Earnings-to-Growth")
    pb_ratio: float = Field(..., description="Price-to-Book")

    # Profitability
    roe: float = Field(..., description="Return on Equity")
    roa: float = Field(..., description="Return on Assets")

    # Leverage
    debt_to_equity: float

    # Cash flow
    fcf: float = Field(..., description="Free Cash Flow (INR)")

    # Margins
    gross_margin: float
    operating_margin: float
    net_margin: float

    # Growth
    revenue_growth: float
    eps_growth: float

    # Ownership (Indian-market specific)
    promoter_holding: float = Field(
        ..., ge=0.0, le=100.0,
        description="Promoter holding percentage",
    )
    fii_holding: float = Field(
        ..., ge=0.0, le=100.0,
        description="FII holding percentage",
    )
    dii_holding: float = Field(
        ..., ge=0.0, le=100.0,
        description="DII holding percentage",
    )

    # Derived
    intrinsic_value: float = Field(
        ..., description="DCF / EPV intrinsic value estimate (INR)",
    )

    timestamp: datetime


class SentimentSignal(BaseModel, frozen=True):
    """Aggregated sentiment signal from a single source for a ticker."""

    ticker: str
    score: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Normalised sentiment score (-1 bearish .. +1 bullish)",
    )
    volume: int = Field(
        ..., ge=0,
        description="Number of mentions / articles / posts analysed",
    )
    trend: float = Field(
        ..., description="Sentiment momentum (change vs. prior window)",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Top keywords driving the sentiment",
    )
    source: str = Field(
        ..., min_length=1,
        description="Source identifier, e.g. 'reddit', 'moneycontrol', 'twitter'",
    )
    timestamp: datetime


class MacroSignal(BaseModel, frozen=True):
    """Macroeconomic / cross-asset snapshot relevant to Indian equity markets."""

    # Indian monetary / macro
    repo_rate: float = Field(..., description="RBI repo rate (%)")
    india_cpi: float = Field(..., description="India CPI YoY (%)")
    india_iip: float = Field(..., description="Index of Industrial Production YoY (%)")

    # Global
    fed_rate: float = Field(..., description="US Fed Funds rate (%)")
    dxy: float = Field(..., description="US Dollar Index")
    brent_crude: float = Field(..., description="Brent crude price (USD/bbl)")

    # Volatility / equity
    india_vix: float = Field(..., description="India VIX level")
    gold_price: float = Field(..., description="Gold price (INR per 10g)")
    nifty_level: float = Field(..., description="Nifty 50 spot level")

    # Flow
    fii_net_flow: float = Field(
        ..., description="FII net flow for the day (INR crore)",
    )
    dii_net_flow: float = Field(
        ..., description="DII net flow for the day (INR crore)",
    )

    # Derived regime
    regime: MarketRegime

    timestamp: datetime


# ---------------------------------------------------------------------------
# Trading / portfolio models
# ---------------------------------------------------------------------------


class Order(BaseModel, frozen=True):
    """Immutable order ticket ready for submission to the broker."""

    order_id: str = Field(..., min_length=1, description="Unique order identifier")
    ticker: str
    exchange: Exchange
    side: OrderSide
    order_type: OrderType
    quantity: int = Field(..., gt=0)
    price: Optional[float] = Field(
        default=None, description="Limit price (required for LIMIT / SL)",
    )
    trigger_price: Optional[float] = Field(
        default=None, description="Trigger price (required for SL / SL_M)",
    )
    agent_name: str = Field(
        ..., min_length=1,
        description="Agent that originated this order",
    )
    reasoning: str = Field(
        ..., min_length=1,
        description="Audit trail: why the order was placed",
    )
    timestamp: datetime


class Position(BaseModel, frozen=True):
    """Snapshot of a single open position."""

    ticker: str
    quantity: int
    avg_price: float = Field(..., gt=0.0)
    current_price: float = Field(..., gt=0.0)
    pnl: float = Field(..., description="Unrealised PnL (INR)")
    pnl_pct: float = Field(..., description="Unrealised PnL (%)")
    sector: Optional[str] = Field(
        default=None,
        description="GICS / BSE sector classification",
    )


class PortfolioState(BaseModel, frozen=True):
    """Full portfolio snapshot at a point in time."""

    cash: float = Field(..., ge=0.0, description="Free cash (INR)")
    positions: list[Position] = Field(default_factory=list)
    total_value: float = Field(..., gt=0.0, description="NAV (INR)")
    deployed_pct: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of capital deployed",
    )
    daily_pnl: float = Field(..., description="Realised + unrealised PnL today (INR)")
    daily_pnl_pct: float
    drawdown_from_peak: float = Field(
        ..., ge=0.0, le=1.0,
        description="Current drawdown from equity peak",
    )


class TradeRecord(BaseModel, frozen=True):
    """Immutable record of an executed trade for the audit log."""

    order_id: str
    ticker: str
    side: OrderSide
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0.0)
    timestamp: datetime
    agent_name: str
    factor_scores: dict[str, float] = Field(default_factory=dict)
    reasoning: str
    risk_check_passed: bool


# ---------------------------------------------------------------------------
# Observability models
# ---------------------------------------------------------------------------


class LatencyLog(BaseModel, frozen=True):
    """Single latency measurement for pipeline observability."""

    stage: str = Field(
        ..., min_length=1,
        description="Pipeline stage, e.g. 'fetch_prices', 'risk_check'",
    )
    duration_ns: int = Field(
        ..., ge=0, description="Duration in nanoseconds",
    )
    timestamp: datetime


class ResearchLog(BaseModel, frozen=True):
    """Record of an agent self-tuning a parameter (requires approval if AUTO_TUNE is off)."""

    agent_name: str = Field(..., min_length=1)
    parameter: str = Field(
        ..., min_length=1,
        description="Parameter that was changed, e.g. 'rsi_oversold_threshold'",
    )
    old_value: str
    new_value: str
    evidence: str = Field(
        ..., min_length=1,
        description="Back-test or statistical evidence supporting the change",
    )
    timestamp: datetime
