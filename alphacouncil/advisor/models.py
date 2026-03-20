"""Pydantic v2 models for the Investment Advisor module."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AdvisorAction(str, enum.Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class InvestmentHorizon(str, enum.Enum):
    SHORT_TERM = "SHORT_TERM"    # 1-4 weeks
    MID_TERM = "MID_TERM"        # 1-6 months
    LONG_TERM = "LONG_TERM"      # 6-24 months


class RiskAppetite(str, enum.Enum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


class ValuationVerdict(str, enum.Enum):
    UNDERVALUED = "UNDERVALUED"
    FAIR_VALUE = "FAIR_VALUE"
    OVERVALUED = "OVERVALUED"


class TechnicalVerdict(BaseModel, frozen=True):
    """Technical analysis summary for a stock."""
    trend: str = Field(..., description="BULLISH / BEARISH / NEUTRAL")
    rsi: float
    rsi_signal: str = Field(..., description="OVERSOLD / NEUTRAL / OVERBOUGHT")
    macd_signal: str = Field(..., description="BULLISH / BEARISH / NEUTRAL")
    ma_alignment: str = Field(..., description="BULLISH / BEARISH / MIXED")
    adx: float
    adx_signal: str = Field(..., description="STRONG_TREND / WEAK_TREND / NO_TREND")
    support: float = Field(..., description="Nearest support level")
    resistance: float = Field(..., description="Nearest resistance level")
    atr: float
    volume_signal: str = Field(..., description="HIGH / NORMAL / LOW")
    breakout: bool = Field(default=False)
    summary: str = Field(..., description="1-2 sentence technical summary")


class FundamentalVerdict(BaseModel, frozen=True):
    """Fundamental analysis summary for a stock."""
    valuation: ValuationVerdict
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None
    eps_growth: Optional[float] = None
    fcf_positive: bool = True
    growth_quality_score: float = Field(0.0, ge=0, le=100)
    intrinsic_value: Optional[float] = None
    current_price: Optional[float] = None
    margin_of_safety: Optional[float] = Field(None, description="(intrinsic - price) / intrinsic")
    financial_health: str = Field("UNKNOWN", description="STRONG / MODERATE / WEAK / UNKNOWN")
    summary: str = Field(..., description="1-2 sentence fundamental summary")


class SentimentVerdict(BaseModel, frozen=True):
    """Sentiment analysis summary for a stock."""
    score: float = Field(..., ge=-1.0, le=1.0)
    signal: str = Field(..., description="BULLISH / BEARISH / NEUTRAL")
    article_count: int = Field(0, ge=0)
    social_buzz: str = Field("NORMAL", description="HIGH / NORMAL / LOW")
    trend: float = Field(0.0, description="Sentiment momentum")
    top_keywords: list[str] = Field(default_factory=list)
    recent_headlines: list[str] = Field(default_factory=list, description="Top 5 recent headlines")
    summary: str = Field(..., description="1-2 sentence sentiment summary")


class RiskAssessment(BaseModel, frozen=True):
    """Risk metrics for a stock."""
    volatility_regime: str = Field(..., description="LOW / MEDIUM / HIGH")
    atr_pct: float = Field(..., description="ATR as % of price")
    beta: Optional[float] = Field(None, description="Beta vs benchmark")
    max_expected_drawdown: float = Field(..., description="Estimated max DD as decimal")
    risk_reward_ratio: float = Field(..., description="Reward/Risk ratio")
    risk_level: str = Field(..., description="LOW / MODERATE / HIGH / VERY_HIGH")
    summary: str


class HorizonRating(BaseModel, frozen=True):
    """Buy rating for a specific investment horizon."""
    horizon: InvestmentHorizon
    action: AdvisorAction
    conviction: int = Field(..., ge=0, le=100)
    target_price: Optional[float] = None
    expected_return_pct: Optional[float] = None
    reasoning: str = ""


class EntryExitLevels(BaseModel, frozen=True):
    """Suggested entry, stop-loss, and target prices."""
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    target_short_term: Optional[float] = None
    target_mid_term: Optional[float] = None
    target_long_term: Optional[float] = None
    risk_reward_short: Optional[float] = None
    risk_reward_mid: Optional[float] = None
    risk_reward_long: Optional[float] = None


class StockRecommendation(BaseModel, frozen=True):
    """Complete stock recommendation with all analysis verdicts."""
    ticker: str
    name: str = Field("", description="Company name")
    exchange: str = Field("", description="NSE / BSE / NASDAQ / NYSE etc.")
    current_price: float
    currency: str = Field("INR")
    action: AdvisorAction
    horizon: InvestmentHorizon
    conviction: int = Field(..., ge=0, le=100)
    technical: TechnicalVerdict
    fundamental: FundamentalVerdict
    sentiment: SentimentVerdict
    risk: RiskAssessment
    levels: EntryExitLevels
    horizon_ratings: list[HorizonRating] = Field(
        default_factory=list,
        description="Buy ratings for short/mid/long term",
    )
    reasoning: str = Field(..., description="3-5 sentence human-readable explanation")
    timestamp: datetime


class ScreenerFilter(BaseModel):
    """Customizable filter for the stock screener."""
    # Technical filters
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None
    above_sma_200: Optional[bool] = None
    macd_bullish: Optional[bool] = None
    adx_min: Optional[float] = None
    # Fundamental filters
    min_revenue_growth: Optional[float] = None
    max_pe: Optional[float] = None
    min_roe: Optional[float] = None
    max_debt_to_equity: Optional[float] = None
    positive_fcf: Optional[bool] = None
    # Sentiment filters
    min_sentiment_score: Optional[float] = None
    min_article_count: Optional[int] = None


class ScreenerResultItem(BaseModel, frozen=True):
    """Single stock result from the screener."""
    ticker: str
    name: str = ""
    current_price: float = 0.0
    composite_score: float = Field(..., description="Overall score 0-100")
    technical_score: float = Field(0.0, ge=0, le=100)
    fundamental_score: float = Field(0.0, ge=0, le=100)
    sentiment_score: float = Field(0.0, ge=0, le=100)
    action: AdvisorAction = AdvisorAction.HOLD
    conviction: int = Field(0, ge=0, le=100)
    key_factors: list[str] = Field(default_factory=list)


class ScreenerResult(BaseModel, frozen=True):
    """Complete screener result set."""
    universe_name: str
    filter_profile: str = "custom"
    total_screened: int
    results: list[ScreenerResultItem]
    timestamp: datetime


class PortfolioAllocation(BaseModel, frozen=True):
    """Single stock allocation in a portfolio suggestion."""
    ticker: str
    name: str = ""
    weight: float = Field(..., ge=0.0, le=1.0)
    amount: float
    action: AdvisorAction
    conviction: int = Field(0, ge=0, le=100)
    sector: str = ""
    rationale: str = ""


class PortfolioSuggestion(BaseModel, frozen=True):
    """Complete portfolio suggestion."""
    capital: float
    risk_appetite: RiskAppetite
    horizon: InvestmentHorizon
    allocations: list[PortfolioAllocation]
    cash_reserve_pct: float = Field(..., ge=0.0, le=1.0)
    expected_annual_return_low: float
    expected_annual_return_high: float
    expected_max_drawdown: float
    expected_sharpe: float
    sector_breakdown: dict[str, float] = Field(default_factory=dict)
    diversification_score: float = Field(0.0, ge=0, le=100)
    reasoning: str
    timestamp: datetime


class MarketOverview(BaseModel, frozen=True):
    """Broad market conditions snapshot."""
    # Indian market
    nifty50_level: float = 0.0
    nifty50_change_pct: float = 0.0
    sensex_level: float = 0.0
    sensex_change_pct: float = 0.0
    india_vix: float = 0.0
    india_vix_signal: str = "NORMAL"
    fii_net_flow: float = 0.0
    dii_net_flow: float = 0.0
    india_regime: str = "NEUTRAL"
    # US market
    sp500_level: float = 0.0
    sp500_change_pct: float = 0.0
    nasdaq_level: float = 0.0
    nasdaq_change_pct: float = 0.0
    us_vix: float = 0.0
    # Global
    dxy: float = 0.0
    gold_price: float = 0.0
    brent_crude: float = 0.0
    # Sector performance (top 5 / bottom 5)
    sector_performance: dict[str, float] = Field(default_factory=dict)
    # Summary
    india_summary: str = ""
    global_summary: str = ""
    risk_outlook: str = "NEUTRAL"
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
