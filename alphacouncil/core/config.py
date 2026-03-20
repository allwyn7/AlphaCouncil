"""System-wide configuration for AlphaCouncil.

Values are loaded from environment variables (upper-cased, prefixed with
``AC_`` where ambiguous) with sensible defaults.  The singleton is obtained
via :func:`get_settings` which uses :func:`functools.lru_cache` so the
``.env`` file is read at most once per process.

Usage::

    from alphacouncil.core.config import get_settings

    cfg = get_settings()
    print(cfg.INITIAL_CAPITAL)
"""

from __future__ import annotations

import zoneinfo
from datetime import time, timezone
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Timezone helper
# ---------------------------------------------------------------------------

IST = zoneinfo.ZoneInfo("Asia/Kolkata")
"""Indian Standard Time (UTC+05:30) -- use everywhere timestamps are created."""

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Central, validated configuration for the AlphaCouncil system.

    All fields can be overridden via environment variables.  For nested /
    prefixed vars ``pydantic-settings`` will look for the exact field name
    (case-insensitive) in the environment.  A ``.env`` file in the project
    root is loaded automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Freeze after construction so the config object is safe for
        # concurrent reads from multiple agents / threads.
        frozen=True,
    )

    # ------------------------------------------------------------------
    # Broker -- Angel One (SmartAPI)
    # ------------------------------------------------------------------
    ANGEL_ONE_API_KEY: str = Field(
        default="", description="Angel One SmartAPI key",
    )
    ANGEL_ONE_CLIENT_ID: str = Field(
        default="", description="Angel One client / login ID",
    )
    ANGEL_ONE_PASSWORD: str = Field(
        default="", description="Angel One login password",
    )
    ANGEL_ONE_TOTP_SECRET: str = Field(
        default="", description="Base-32 TOTP secret for 2FA",
    )

    # ------------------------------------------------------------------
    # Broker -- Fyers (optional, secondary)
    # ------------------------------------------------------------------
    FYERS_APP_ID: Optional[str] = Field(
        default=None, description="Fyers API v3 app ID",
    )
    FYERS_SECRET_ID: Optional[str] = Field(
        default=None, description="Fyers API v3 secret",
    )

    # ------------------------------------------------------------------
    # Data providers
    # ------------------------------------------------------------------
    FRED_API_KEY: str = Field(
        default="", description="FRED (Federal Reserve) API key for US macro data",
    )

    # ------------------------------------------------------------------
    # Reddit (sentiment)
    # ------------------------------------------------------------------
    REDDIT_CLIENT_ID: str = Field(default="")
    REDDIT_CLIENT_SECRET: str = Field(default="")
    REDDIT_USER_AGENT: str = Field(
        default="alphacouncil:v0.1 (by /u/alphacouncil_bot)",
    )

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(
        default=None,
        description="Telegram Bot token for trade / alert notifications",
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    DATABASE_URL: str = Field(
        default="sqlite:///data/alphacouncil.db",
        description="SQLAlchemy-compatible database URL",
    )
    CACHE_DIR: str = Field(
        default="data/cache",
        description="Directory for on-disk caches (prices, fundamentals, etc.)",
    )

    # ------------------------------------------------------------------
    # Trading hours (IST)
    # ------------------------------------------------------------------
    MARKET_OPEN: time = Field(
        default=time(9, 15),
        description="NSE continuous trading session open (IST)",
    )
    MARKET_CLOSE: time = Field(
        default=time(15, 30),
        description="NSE continuous trading session close (IST)",
    )
    AMO_START: time = Field(
        default=time(15, 45),
        description="After-Market Order window start (IST)",
    )
    AMO_END: time = Field(
        default=time(8, 57),
        description="After-Market Order window end (IST, next morning)",
    )

    # ------------------------------------------------------------------
    # Safety / risk limits
    # ------------------------------------------------------------------
    MAX_CAPITAL_PER_STOCK: float = Field(
        default=0.05,
        description="Max fraction of total capital in a single stock",
    )
    MAX_SECTOR_EXPOSURE: float = Field(
        default=0.25,
        description="Max fraction of total capital in a single sector",
    )
    MAX_DEPLOYED: float = Field(
        default=0.80,
        description="Max fraction of capital deployed (rest stays as cash buffer)",
    )
    MAX_POSITIONS: int = Field(
        default=15,
        description="Maximum number of concurrent open positions",
    )
    MAX_DAILY_TRADES: int = Field(
        default=50,
        description="Hard cap on trades per calendar day",
    )
    MAX_ORDER_VALUE: float = Field(
        default=50_000.0,
        description="Maximum single-order notional value (INR)",
    )
    DAILY_LOSS_LIMIT: float = Field(
        default=0.03,
        description="Halt trading if daily loss exceeds this fraction of NAV",
    )
    SINGLE_TRADE_LOSS_LIMIT: float = Field(
        default=0.015,
        description="Max acceptable loss on a single trade as fraction of NAV",
    )
    MAX_DRAWDOWN: float = Field(
        default=0.08,
        description="System-wide drawdown circuit-breaker (fraction from peak)",
    )
    ERROR_THRESHOLD: int = Field(
        default=5,
        description="Consecutive errors before an agent is auto-demoted",
    )

    # ------------------------------------------------------------------
    # Paper-trading graduation gate
    # ------------------------------------------------------------------
    PAPER_TRADING_DAYS: int = Field(
        default=30,
        description="Minimum paper-trading days before live promotion",
    )
    MIN_SHARPE: float = Field(
        default=0.5,
        description="Minimum annualised Sharpe ratio during paper phase",
    )
    MAX_DRAWDOWN_PAPER: float = Field(
        default=0.15,
        description="Maximum drawdown allowed during paper phase",
    )
    MIN_WIN_RATE: float = Field(
        default=0.40,
        description="Minimum win rate during paper phase",
    )

    # ------------------------------------------------------------------
    # Cache TTLs (seconds)
    # ------------------------------------------------------------------
    PRICE_TTL: int = Field(
        default=60, description="Live / intraday price cache TTL (s)",
    )
    FUNDAMENTAL_TTL: int = Field(
        default=86_400, description="Fundamental data cache TTL (s) -- 1 day",
    )
    MACRO_TTL: int = Field(
        default=21_600, description="Macro data cache TTL (s) -- 6 hours",
    )
    SENTIMENT_TTL: int = Field(
        default=900, description="Sentiment data cache TTL (s) -- 15 min",
    )

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------
    DEFAULT_UNIVERSE: list[str] = Field(
        default=[
            # Large-cap growth / quality
            "RELIANCE.NS",
            "TCS.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "BHARTIARTL.NS",
            "HINDUNILVR.NS",
            "ITC.NS",
            "LT.NS",
            "SBIN.NS",
            # Mid-cap growth tilt
            "TRENT.NS",
            "PERSISTENT.NS",
            "POLYCAB.NS",
            "JINDALSTEL.NS",
            "DIXON.NS",
            "ASTRAL.NS",
            "AUROPHARMA.NS",
            "MUTHOOTFIN.NS",
            "FEDERALBNK.NS",
            "COFORGE.NS",
            # IT / digital
            "HCLTECH.NS",
            "WIPRO.NS",
            "LTIM.NS",
            "TECHM.NS",
            # Financials
            "BAJFINANCE.NS",
            "KOTAKBANK.NS",
            "AXISBANK.NS",
            # Auto / EV
            "M&M.NS",
            "TATAMOTORS.NS",
            "MARUTI.NS",
            # Consumption / FMCG
            "NESTLEIND.NS",
            "DABUR.NS",
            "TITAN.NS",
            # Energy / industrials
            "NTPC.NS",
            "POWERGRID.NS",
            "ADANIENT.NS",
            # Pharma
            "SUNPHARMA.NS",
            "DRREDDY.NS",
        ],
        description="Growth-tilted default stock universe (NSE tickers)",
    )

    ETF_UNIVERSE: list[str] = Field(
        default=[
            "NIFTYBEES.NS",
            "BANKBEES.NS",
            "GOLDBEES.NS",
        ],
        description="ETFs used for hedging / parking",
    )

    # ------------------------------------------------------------------
    # Capital
    # ------------------------------------------------------------------
    INITIAL_CAPITAL: float = Field(
        default=1_000_000.0,
        description="Starting paper / live capital (INR) -- Rs 10 lakh",
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Root logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # ------------------------------------------------------------------
    # Self-tuning
    # ------------------------------------------------------------------
    AUTO_TUNE: bool = Field(
        default=False,
        description=(
            "If True agents may self-tune parameters without human approval.  "
            "Defaults to False as a safety measure."
        ),
    )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    The first call reads environment variables and the ``.env`` file;
    subsequent calls return the cached instance.
    """
    return Settings()
