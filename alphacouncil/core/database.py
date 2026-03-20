"""Database setup and table definitions for AlphaCouncil.

Provides a SQLAlchemy Core schema for SQLite, along with engine / session
factory helpers.  All tables are defined declaratively via ``sqlalchemy.Table``
objects so the rest of the codebase can import them without pulling in an ORM
layer.

Usage::

    from alphacouncil.core.database import init_db, get_engine, get_session

    init_db()                       # creates all tables
    engine  = get_engine()          # raw engine for Core queries
    session = get_session()         # scoped session (autoclose)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_DB_URL: Final[str] = "sqlite:///data/alphacouncil.db"

# Shared metadata instance -- every Table below is registered here.
metadata = MetaData()

# ---------------------------------------------------------------------------
# Table definitions (SQLAlchemy Core)
# ---------------------------------------------------------------------------

trades = Table(
    "trades",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, server_default=func.now(), nullable=False),
    Column("symbol", String(32), nullable=False, index=True),
    Column("side", String(8), nullable=False),       # "buy" / "sell"
    Column("order_type", String(16), nullable=False), # "market" / "limit" / ...
    Column("quantity", Float, nullable=False),
    Column("price", Float, nullable=False),
    Column("fees", Float, nullable=False, server_default="0.0"),
    Column("exchange", String(16), nullable=False),
    Column("agent_id", String(64), nullable=True),
    Column("strategy", String(64), nullable=True),
    Column("notes", Text, nullable=True),
)

positions = Table(
    "positions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, server_default=func.now(), nullable=False),
    Column("symbol", String(32), nullable=False, index=True),
    Column("quantity", Float, nullable=False),
    Column("avg_entry_price", Float, nullable=False),
    Column("current_price", Float, nullable=True),
    Column("unrealised_pnl", Float, nullable=True),
    Column("realised_pnl", Float, nullable=True),
    Column("exchange", String(16), nullable=False),
)

portfolio_snapshots = Table(
    "portfolio_snapshots",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, server_default=func.now(), nullable=False, index=True),
    Column("total_value", Float, nullable=False),
    Column("cash", Float, nullable=False),
    Column("invested", Float, nullable=False),
    Column("unrealised_pnl", Float, nullable=True),
    Column("realised_pnl", Float, nullable=True),
    Column("drawdown", Float, nullable=True),
    Column("sharpe_ratio", Float, nullable=True),
    Column("notes", Text, nullable=True),
)

agent_signals = Table(
    "agent_signals",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, server_default=func.now(), nullable=False, index=True),
    Column("agent_id", String(64), nullable=False, index=True),
    Column("signal_type", String(32), nullable=False),  # technical / fundamental / ...
    Column("symbol", String(32), nullable=True, index=True),
    Column("action", String(16), nullable=False),        # buy / sell / hold
    Column("confidence", Float, nullable=False),
    Column("horizon", String(16), nullable=True),
    Column("payload", Text, nullable=True),              # JSON blob
    Column("notes", Text, nullable=True),
)

latency_logs = Table(
    "latency_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, server_default=func.now(), nullable=False, index=True),
    Column("stage", String(128), nullable=False, index=True),
    Column("latency_ns", Integer, nullable=False),
    Column("metadata_json", Text, nullable=True),        # optional JSON context
)

research_logs = Table(
    "research_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, server_default=func.now(), nullable=False, index=True),
    Column("agent_id", String(64), nullable=False, index=True),
    Column("query", Text, nullable=False),
    Column("result_summary", Text, nullable=True),
    Column("sources", Text, nullable=True),              # JSON list of URLs / refs
    Column("tokens_used", Integer, nullable=True),
    Column("cost_usd", Float, nullable=True),
)

audit_trail = Table(
    "audit_trail",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, server_default=func.now(), nullable=False, index=True),
    Column("actor", String(64), nullable=False),         # agent-id or "system"
    Column("action", String(64), nullable=False),
    Column("resource", String(128), nullable=True),
    Column("detail", Text, nullable=True),               # free-form JSON
    Column("severity", String(16), nullable=False, server_default="info"),
)

sentiment_cache = Table(
    "sentiment_cache",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, server_default=func.now(), nullable=False, index=True),
    Column("symbol", String(32), nullable=True, index=True),
    Column("source", String(64), nullable=False),        # twitter / news / reddit / ...
    Column("score", Float, nullable=False),
    Column("magnitude", Float, nullable=True),
    Column("raw_text", Text, nullable=True),
    Column("expires_at", DateTime, nullable=True),
)

agent_portfolio_snapshots = Table(
    "agent_portfolio_snapshots",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, server_default=func.now(), nullable=False, index=True),
    Column("agent_id", String(64), nullable=False, index=True),
    Column("total_value", Float, nullable=False),
    Column("cash", Float, nullable=False),
    Column("invested", Float, nullable=False),
    Column("unrealised_pnl", Float, nullable=True),
    Column("realised_pnl", Float, nullable=True),
    Column("allocation_json", Text, nullable=True),      # per-symbol breakdown
    Column("notes", Text, nullable=True),
)

# ---------------------------------------------------------------------------
# Advisor tables
# ---------------------------------------------------------------------------

advisor_recommendations = Table(
    "advisor_recommendations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, server_default=func.now(), nullable=False, index=True),
    Column("ticker", String(32), nullable=False, index=True),
    Column("action", String(16), nullable=False),        # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    Column("horizon", String(16), nullable=False),       # SHORT_TERM / MID_TERM / LONG_TERM
    Column("conviction", Integer, nullable=False),
    Column("reasoning", Text, nullable=True),
    Column("technical_json", Text, nullable=True),       # JSON blob of TechnicalVerdict
    Column("fundamental_json", Text, nullable=True),     # JSON blob of FundamentalVerdict
    Column("sentiment_json", Text, nullable=True),       # JSON blob of SentimentVerdict
    Column("levels_json", Text, nullable=True),          # JSON blob of EntryExitLevels
    Column("current_price", Float, nullable=True),
    Column("currency", String(8), nullable=True),
)

advisor_watchlist = Table(
    "advisor_watchlist",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(32), nullable=False, unique=True, index=True),
    Column("added_at", DateTime, server_default=func.now(), nullable=False),
    Column("last_recommendation", String(16), nullable=True),
    Column("last_checked", DateTime, nullable=True),
    Column("notes", Text, nullable=True),
)

advisor_portfolio_suggestions = Table(
    "advisor_portfolio_suggestions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, server_default=func.now(), nullable=False, index=True),
    Column("risk_profile", String(16), nullable=False),
    Column("capital", Float, nullable=False),
    Column("allocations_json", Text, nullable=True),     # JSON blob
    Column("expected_sharpe", Float, nullable=True),
    Column("expected_return_low", Float, nullable=True),
    Column("expected_return_high", Float, nullable=True),
    Column("notes", Text, nullable=True),
)

# ---------------------------------------------------------------------------
# Engine / session singletons
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine(db_url: str = DEFAULT_DB_URL) -> Engine:
    """Return (and lazily create) the singleton SQLAlchemy :class:`Engine`.

    Parameters
    ----------
    db_url:
        SQLAlchemy connection string.  Defaults to
        ``sqlite:///data/alphacouncil.db``.
    """
    global _engine  # noqa: PLW0603
    if _engine is None:
        # Ensure the parent directory exists for the SQLite file.
        if db_url.startswith("sqlite:///"):
            db_path = Path(db_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(
            db_url,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},  # SQLite-specific
        )
        logger.info("Database engine created: %s", db_url)
    return _engine


def get_session(db_url: str = DEFAULT_DB_URL) -> Session:
    """Return a new :class:`Session` bound to the singleton engine.

    The caller is responsible for closing the session (or using it as a
    context manager).
    """
    global _session_factory  # noqa: PLW0603
    engine = get_engine(db_url)
    if _session_factory is None:
        _session_factory = sessionmaker(bind=engine)
        logger.debug("Session factory created")
    return _session_factory()


def init_db(db_url: str = DEFAULT_DB_URL) -> Engine:
    """Create all tables defined in :data:`metadata` and return the engine.

    Safe to call multiple times -- SQLAlchemy's ``create_all`` is
    idempotent.
    """
    engine = get_engine(db_url)
    metadata.create_all(engine)
    logger.info("All database tables created / verified")
    return engine


def reset_engine() -> None:
    """Dispose of the current engine and session factory.

    Useful in tests or when switching databases at runtime.
    """
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        _engine.dispose()
        _engine = None
    _session_factory = None
    logger.info("Database engine reset")
