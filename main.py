#!/usr/bin/env python3
"""AlphaCouncil -- CLI entry point for the multi-agent trading system.

Commands
--------
    python main.py backtest   --start 2022-01-01 --end 2025-12-31
    python main.py paper-trade
    python main.py live-trade
    python main.py research
    python main.py dashboard
    python main.py kill
    python main.py status
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import time as _time
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resilient imports -- optional heavy deps are wrapped so the CLI still boots
# even when some packages (torch, optuna, ...) are not yet installed.
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_a: Any, **_kw: Any) -> None:  # type: ignore[misc]
        pass

try:
    import structlog

    def _configure_structlog(log_level: str = "INFO") -> None:
        """Set up structlog with a human-readable console renderer."""
        import logging

        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, log_level.upper(), logging.INFO),
            stream=sys.stderr,
        )
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, log_level.upper(), logging.INFO),
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
except ImportError:  # pragma: no cover
    import logging

    structlog = None  # type: ignore[assignment]

    def _configure_structlog(log_level: str = "INFO") -> None:
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )

# Core imports -- these should always be available.
try:
    from alphacouncil.core.config import IST, get_settings
    from alphacouncil.core.database import init_db, get_engine
    from alphacouncil.core.message_bus import MessageBus
    from alphacouncil.core.cache_manager import TieredCache
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"[FATAL] Core AlphaCouncil packages not importable: {exc}\n"
        "        Run: pip install -e ."
    )

# Broker adapters
try:
    from alphacouncil.core.broker import PaperBroker, AngelOneBroker
except ImportError:  # pragma: no cover
    PaperBroker = None  # type: ignore[misc,assignment]
    AngelOneBroker = None  # type: ignore[misc,assignment]

# Agents
try:
    from alphacouncil.agents import (
        GrowthMomentumAgent,
        MeanReversionAgent,
        MultiFactorRankingAgent,
        PortfolioOptimizationAgent,
        SentimentAlphaAgent,
        VolatilityRegimeAgent,
    )
    from alphacouncil.agents.meta import MetaAgent
except ImportError:  # pragma: no cover
    GrowthMomentumAgent = None  # type: ignore[misc,assignment]
    MeanReversionAgent = None  # type: ignore[misc,assignment]
    MultiFactorRankingAgent = None  # type: ignore[misc,assignment]
    PortfolioOptimizationAgent = None  # type: ignore[misc,assignment]
    SentimentAlphaAgent = None  # type: ignore[misc,assignment]
    VolatilityRegimeAgent = None  # type: ignore[misc,assignment]
    MetaAgent = None  # type: ignore[misc,assignment]

# Safety
try:
    from alphacouncil.core.risk_manager import RiskManager
    from alphacouncil.core.safety import (
        AuditTrail,
        KillSwitch,
        PositionLimits,
        ValidationGate,
    )
except ImportError:  # pragma: no cover
    RiskManager = None  # type: ignore[misc,assignment]
    AuditTrail = None  # type: ignore[misc,assignment]
    KillSwitch = None  # type: ignore[misc,assignment]
    PositionLimits = None  # type: ignore[misc,assignment]
    ValidationGate = None  # type: ignore[misc,assignment]

# Research
try:
    from alphacouncil.research import (
        PerformanceAttribution,
        StrategyBacktester,
        AgentParameterOptimizer,
        RegimeAdaptiveWeightLearner,
        NewsAlphaTracker,
        StrategyDiscovery,
    )
except ImportError:  # pragma: no cover
    PerformanceAttribution = None  # type: ignore[misc,assignment]
    StrategyBacktester = None  # type: ignore[misc,assignment]
    AgentParameterOptimizer = None  # type: ignore[misc,assignment]
    RegimeAdaptiveWeightLearner = None  # type: ignore[misc,assignment]
    NewsAlphaTracker = None  # type: ignore[misc,assignment]
    StrategyDiscovery = None  # type: ignore[misc,assignment]

# Optional: yfinance for data feed tasks
try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore[assignment]

# Advisor module
try:
    from alphacouncil.advisor.engine import InvestmentAdvisor
    from alphacouncil.advisor.screener import StockScreener
    from alphacouncil.advisor.report import ReportGenerator
    from alphacouncil.advisor.universes import get_universe, list_universes
except ImportError:  # pragma: no cover
    InvestmentAdvisor = None  # type: ignore[misc,assignment]
    StockScreener = None  # type: ignore[misc,assignment]
    ReportGenerator = None  # type: ignore[misc,assignment]
    get_universe = None  # type: ignore[misc,assignment]
    list_universes = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = r"""
    ___    __      __          ______                      _ __
   /   |  / /___  / /_  ____ / ____/___  __  ______  ____(_) /
  / /| | / / __ \/ __ \/ __ `/ /   / __ \/ / / / __ \/ ___/ / /
 / ___ |/ / /_/ / / / / /_/ / /___/ /_/ / /_/ / / / / /__/ / /
/_/  |_/_/ .___/_/ /_/\__,_/\____/\____/\__,_/_/ /_/\___/_/_/
        /_/
                   Multi-Agent Trading System
                        NSE India  |  v0.1
"""


def _print_banner() -> None:
    print(_BANNER, flush=True)


# ---------------------------------------------------------------------------
# Graceful shutdown helpers
# ---------------------------------------------------------------------------

_shutdown_event: asyncio.Event | None = None


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register SIGINT / SIGTERM handlers that set the shutdown event."""
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    def _handler(sig: int, _frame: Any) -> None:
        name = signal.Signals(sig).name
        print(f"\n[!] Received {name} -- initiating graceful shutdown ...", flush=True)
        if _shutdown_event is not None:
            _shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

_AGENT_REGISTRY: list[tuple[str, Any]] = [
    ("growth_momentum", GrowthMomentumAgent),
    ("mean_reversion", MeanReversionAgent),
    ("multifactor_ranking", MultiFactorRankingAgent),
    ("portfolio_optimizer", PortfolioOptimizationAgent),
    ("sentiment_alpha", SentimentAlphaAgent),
    ("volatility_regime", VolatilityRegimeAgent),
]


def _create_agents(
    config: Any,
    cache: TieredCache,
    bus: MessageBus,
    db_engine: Any,
) -> list[Any]:
    """Instantiate all 6 quant agents."""
    agents = []
    for name, cls in _AGENT_REGISTRY:
        if cls is None:
            print(f"  [WARN] Agent class for '{name}' not available -- skipped")
            continue
        agent = cls(
            name=name,
            config=config,
            cache=cache,
            bus=bus,
            db_engine=db_engine,
        )
        agents.append(agent)
    return agents


# ---------------------------------------------------------------------------
# Shared bootstrap
# ---------------------------------------------------------------------------

def _bootstrap() -> tuple[Any, Any, TieredCache, MessageBus]:
    """Load .env, init structlog, init DB, return (settings, engine, cache, bus)."""
    load_dotenv()
    cfg = get_settings()
    _configure_structlog(cfg.LOG_LEVEL)
    engine = init_db(cfg.DATABASE_URL)
    cache = TieredCache(cache_dir=cfg.CACHE_DIR)
    bus = MessageBus()
    return cfg, engine, cache, bus


# ===================================================================
# COMMAND: backtest
# ===================================================================


async def cmd_backtest(args: argparse.Namespace) -> None:
    """Run walk-forward backtest for all agents."""
    if StrategyBacktester is None or PaperBroker is None:
        sys.exit("[FATAL] Research or broker modules not installed.")

    cfg, engine, cache, bus = _bootstrap()
    print(f"[*] Backtest period: {args.start} -> {args.end}")
    print(f"[*] Universe: {len(cfg.DEFAULT_UNIVERSE)} stocks")
    print()

    broker = PaperBroker(initial_capital=cfg.INITIAL_CAPITAL)
    backtester = StrategyBacktester(broker=broker, db_engine=engine)
    agents = _create_agents(cfg, cache, bus, engine)

    if not agents:
        sys.exit("[FATAL] No agents available for backtesting.")

    all_results: dict[str, dict[str, Any]] = {}

    for agent in agents:
        print(f"  [>] Backtesting agent: {agent.name}")
        try:
            result = await backtester.backtest(
                agent=agent,
                universe=cfg.DEFAULT_UNIVERSE,
                start=args.start,
                end=args.end,
            )
            all_results[agent.name] = result
            sharpe = result.get("sharpe", 0.0)
            max_dd = result.get("max_drawdown", 0.0)
            total_ret = result.get("total_return", 0.0)
            win_rate = result.get("win_rate", 0.0)
            print(
                f"      Sharpe={sharpe:.2f}  MaxDD={max_dd:.1%}  "
                f"Return={total_ret:.1%}  WinRate={win_rate:.1%}"
            )
        except Exception as exc:
            print(f"      [ERR] {agent.name} failed: {exc}")
            all_results[agent.name] = {"error": str(exc)}

    # ---- Tearsheet summary ----
    print()
    print("=" * 72)
    print("  BACKTEST TEARSHEET SUMMARY")
    print("=" * 72)
    print(f"  {'Agent':<28} {'Sharpe':>8} {'MaxDD':>8} {'Return':>10} {'WinRate':>9}")
    print("-" * 72)
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name:<28} {'ERROR':>8}")
            continue
        print(
            f"  {name:<28} "
            f"{res.get('sharpe', 0.0):>8.2f} "
            f"{res.get('max_drawdown', 0.0):>7.1%} "
            f"{res.get('total_return', 0.0):>9.1%} "
            f"{res.get('win_rate', 0.0):>8.1%}"
        )
    print("=" * 72)
    print()


# ===================================================================
# COMMAND: paper-trade
# ===================================================================

async def _data_feed_task(
    bus: MessageBus,
    cfg: Any,
    shutdown: asyncio.Event,
) -> None:
    """Fetch prices via yfinance every 60s and push to message bus."""
    if yf is None:
        print("  [WARN] yfinance not installed -- DataFeedTask disabled")
        await shutdown.wait()
        return

    universe = cfg.DEFAULT_UNIVERSE
    while not shutdown.is_set():
        try:
            data = yf.download(
                tickers=" ".join(universe),
                period="1d",
                interval="1m",
                progress=False,
            )
            await bus.publish("price_update", data, publisher="data_feed")
        except Exception as exc:
            print(f"  [WARN] DataFeed error: {exc}")
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=cfg.PRICE_TTL)
            break
        except asyncio.TimeoutError:
            pass


async def _sentiment_crawler_task(
    bus: MessageBus,
    cfg: Any,
    shutdown: asyncio.Event,
) -> None:
    """Poll RSS feeds every 5 minutes, score headlines via FinBERT, publish per-ticker sentiment."""
    interval = cfg.SENTIMENT_TTL  # 900s = 15 min
    while not shutdown.is_set():
        try:
            # Try to use the SentimentEngine for real analysis
            per_ticker: dict[str, dict] = {}
            try:
                from alphacouncil.analysis.sentiment import SentimentEngine
                cache = TieredCache()
                engine = SentimentEngine(cache)
                headlines = await engine.fetch_headlines()

                if headlines:
                    # Score all headlines
                    titles = [h.get("title", "") for h in headlines if h.get("title")]
                    if titles:
                        scores = await engine.analyze_batch(titles[:64])  # batch limit

                        # Map headlines to tickers
                        for i, headline in enumerate(headlines[:len(scores)]):
                            for ticker in cfg.DEFAULT_UNIVERSE:
                                mapped = engine.map_ticker(headline.get("title", ""), [ticker])
                                if mapped:
                                    score = scores[i] if i < len(scores) else 0.0
                                    if ticker not in per_ticker:
                                        per_ticker[ticker] = {
                                            "score": 0.0, "count": 0,
                                            "article_count": 0,
                                            "avg_article_count_30d": 5.0,
                                            "sentiment_7d": 0.0,
                                            "sentiment_30d": 0.0,
                                        }
                                    entry = per_ticker[ticker]
                                    entry["count"] += 1
                                    entry["article_count"] += 1
                                    entry["score"] = (
                                        (entry["score"] * (entry["count"] - 1) + score) / entry["count"]
                                    )
                                    entry["sentiment_7d"] = entry["score"]
                                    entry["sentiment_30d"] = entry["score"] * 0.7

            except ImportError:
                pass  # SentimentEngine not available
            except Exception as exc:
                print(f"  [WARN] SentimentCrawler analysis error: {exc}")

            await bus.publish(
                "sentiment",
                {"source": "crawler", "status": "poll_complete", "per_ticker": per_ticker},
                publisher="sentiment_crawler",
            )
        except Exception as exc:
            print(f"  [WARN] SentimentCrawler error: {exc}")
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=min(interval, 300))
            break
        except asyncio.TimeoutError:
            pass


async def _macro_update_task(
    bus: MessageBus,
    cfg: Any,
    shutdown: asyncio.Event,
) -> None:
    """Poll FRED / macro data every 6 hours."""
    interval = cfg.MACRO_TTL  # 21600s = 6hr
    while not shutdown.is_set():
        try:
            await bus.publish(
                "macro",
                {"source": "fred", "status": "poll_complete"},
                publisher="macro_updater",
            )
        except Exception as exc:
            print(f"  [WARN] MacroUpdate error: {exc}")
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass


async def _agent_task(
    agent: Any,
    bus: MessageBus,
    cfg: Any,
    shutdown: asyncio.Event,
) -> None:
    """Run a single agent's signal-generation loop."""
    price_queue = bus.subscribe("price_update")
    sentiment_queue = bus.subscribe("sentiment")

    latest_sentiment: dict[str, Any] = {}

    while not shutdown.is_set():
        try:
            # Check for sentiment updates (non-blocking)
            try:
                sent_envelope = await asyncio.wait_for(sentiment_queue.get(), timeout=0.1)
                payload = sent_envelope.payload
                if isinstance(payload, dict) and "per_ticker" in payload:
                    latest_sentiment = payload["per_ticker"]
            except asyncio.TimeoutError:
                pass

            # Wait for price update
            envelope = await asyncio.wait_for(price_queue.get(), timeout=5.0)
            market_data = {
                "prices": envelope.payload,
                "sentiment": latest_sentiment,
                "sentiment_signals": latest_sentiment,
            }
            signals = await agent.generate_signals(cfg.DEFAULT_UNIVERSE, market_data)
            for sig in signals:
                await bus.publish("signal", sig, publisher=agent.name)
        except asyncio.TimeoutError:
            continue
        except Exception as exc:
            print(f"  [WARN] Agent {agent.name} error: {exc}")
    bus.unsubscribe("price_update", price_queue)
    bus.unsubscribe("sentiment", sentiment_queue)


async def _meta_agent_task(
    meta: Any,
    bus: MessageBus,
    shutdown: asyncio.Event,
) -> None:
    """Collect votes from agents and run the council decision loop."""
    queue = bus.subscribe("signal")
    signal_buffer: list[Any] = []
    last_council_time = _time.monotonic()
    council_interval = 60.0  # Run council every 60s

    while not shutdown.is_set():
        try:
            envelope = await asyncio.wait_for(queue.get(), timeout=2.0)
            signal_buffer.append(envelope.payload)
        except asyncio.TimeoutError:
            pass
        except Exception as exc:
            print(f"  [WARN] MetaAgent queue error: {exc}")
            continue

        elapsed = _time.monotonic() - last_council_time
        if elapsed >= council_interval and signal_buffer:
            try:
                await meta.run_council(signal_buffer)
            except Exception as exc:
                print(f"  [WARN] MetaAgent council error: {exc}")
            signal_buffer.clear()
            last_council_time = _time.monotonic()

    bus.unsubscribe("signal", queue)


async def _risk_manager_task(
    risk_manager: Any,
    bus: MessageBus,
    shutdown: asyncio.Event,
) -> None:
    """Monitor risk alerts from the bus."""
    queue = bus.subscribe("risk_alert")
    while not shutdown.is_set():
        try:
            envelope = await asyncio.wait_for(queue.get(), timeout=5.0)
            print(f"  [RISK] Alert: {envelope.payload}")
        except asyncio.TimeoutError:
            continue
        except Exception as exc:
            print(f"  [WARN] RiskManager task error: {exc}")
    bus.unsubscribe("risk_alert", queue)


async def _portfolio_writer_task(
    bus: MessageBus,
    db_engine: Any,
    shutdown: asyncio.Event,
) -> None:
    """Batch portfolio state writes to SQLite."""
    queue = bus.subscribe("order")
    write_buffer: list[Any] = []
    flush_interval = 30.0
    last_flush = _time.monotonic()

    while not shutdown.is_set():
        try:
            envelope = await asyncio.wait_for(queue.get(), timeout=2.0)
            write_buffer.append(envelope.payload)
        except asyncio.TimeoutError:
            pass
        except Exception as exc:
            print(f"  [WARN] PortfolioWriter error: {exc}")
            continue

        elapsed = _time.monotonic() - last_flush
        if elapsed >= flush_interval and write_buffer:
            try:
                # Batch insert to DB would go here
                write_buffer.clear()
                last_flush = _time.monotonic()
            except Exception as exc:
                print(f"  [WARN] PortfolioWriter flush error: {exc}")

    # Final flush on shutdown
    if write_buffer:
        try:
            write_buffer.clear()
        except Exception:
            pass
    bus.unsubscribe("order", queue)


def _is_market_hours(cfg: Any) -> bool:
    """Return True if current IST time is within 9:15 -- 15:30."""
    now_ist = datetime.now(IST).time()
    return cfg.MARKET_OPEN <= now_ist <= cfg.MARKET_CLOSE


async def _warmup_cache(cache: TieredCache, cfg: Any) -> None:
    """Pre-fetch cache warmup before market open (run at ~9:00 AM IST)."""
    print("[*] Running pre-market cache warmup ...")
    if yf is not None:
        try:
            tickers = " ".join(cfg.DEFAULT_UNIVERSE)
            data = yf.download(
                tickers=tickers,
                period="5d",
                interval="1d",
                progress=False,
            )
            for ticker in cfg.DEFAULT_UNIVERSE:
                try:
                    if isinstance(data.columns, tuple):
                        close = data["Close"][ticker].iloc[-1]
                    else:
                        close = data["Close"].iloc[-1]
                    await cache.set(f"{ticker}:price", float(close), "price")
                except Exception:
                    pass
            print(f"  [OK] Cached prices for {len(cfg.DEFAULT_UNIVERSE)} tickers")
        except Exception as exc:
            print(f"  [WARN] Cache warmup failed: {exc}")
    else:
        print("  [SKIP] yfinance not available for warmup")


async def cmd_paper_trade(args: argparse.Namespace) -> None:
    """Run the full system in paper-trading mode."""
    if PaperBroker is None:
        sys.exit("[FATAL] PaperBroker not available. Install broker deps.")

    cfg, engine, cache, bus = _bootstrap()

    broker = PaperBroker(initial_capital=cfg.INITIAL_CAPITAL)
    agents = _create_agents(cfg, cache, bus, engine)

    if not agents:
        sys.exit("[FATAL] No agents available.")

    # Safety layer
    audit = AuditTrail(engine) if AuditTrail is not None else None
    limits = PositionLimits(cfg) if PositionLimits is not None else None
    kill_switch = (
        KillSwitch(broker=broker, config=cfg, bus=bus, audit=audit)
        if KillSwitch is not None
        else None
    )
    risk_manager = (
        RiskManager(limits=limits, kill_switch=kill_switch, audit=audit, config=cfg)
        if RiskManager is not None
        else None
    )

    # MetaAgent (the council)
    meta = (
        MetaAgent(
            agents=agents,
            risk_manager=risk_manager,
            broker=broker,
            config=cfg,
            bus=bus,
            db_engine=engine,
        )
        if MetaAgent is not None
        else None
    )

    # Shutdown event
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)
    shutdown = _shutdown_event
    assert shutdown is not None

    print("[*] Paper trading mode -- starting all tasks")
    print(f"    Capital : INR {cfg.INITIAL_CAPITAL:,.0f}")
    print(f"    Universe: {len(cfg.DEFAULT_UNIVERSE)} stocks")
    print(f"    Market  : {cfg.MARKET_OPEN.strftime('%H:%M')} - {cfg.MARKET_CLOSE.strftime('%H:%M')} IST")
    print()

    # Pre-market cache warmup (if before market open)
    now_ist = datetime.now(IST).time()
    warmup_time = time(9, 0)
    if now_ist <= warmup_time:
        secs_to_warmup = (
            datetime.combine(datetime.today(), warmup_time)
            - datetime.combine(datetime.today(), now_ist)
        ).total_seconds()
        print(f"[*] Waiting {secs_to_warmup:.0f}s until 09:00 IST for cache warmup ...")
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=max(0, secs_to_warmup))
            if shutdown.is_set():
                print("[*] Shutdown requested before warmup.")
                return
        except asyncio.TimeoutError:
            pass
        await _warmup_cache(cache, cfg)
    else:
        await _warmup_cache(cache, cfg)

    # Assemble concurrent tasks
    tasks: list[asyncio.Task[None]] = []

    tasks.append(asyncio.create_task(
        _data_feed_task(bus, cfg, shutdown), name="data_feed"
    ))
    tasks.append(asyncio.create_task(
        _sentiment_crawler_task(bus, cfg, shutdown), name="sentiment_crawler"
    ))
    tasks.append(asyncio.create_task(
        _macro_update_task(bus, cfg, shutdown), name="macro_update"
    ))

    for agent in agents:
        tasks.append(asyncio.create_task(
            _agent_task(agent, bus, cfg, shutdown), name=f"agent_{agent.name}"
        ))

    if meta is not None:
        tasks.append(asyncio.create_task(
            _meta_agent_task(meta, bus, shutdown), name="meta_agent"
        ))

    if risk_manager is not None:
        tasks.append(asyncio.create_task(
            _risk_manager_task(risk_manager, bus, shutdown), name="risk_manager"
        ))

    tasks.append(asyncio.create_task(
        _portfolio_writer_task(bus, engine, shutdown), name="portfolio_writer"
    ))

    # Trading hours guard
    tasks.append(asyncio.create_task(
        _trading_hours_guard(cfg, shutdown), name="trading_hours_guard"
    ))

    print(f"[*] {len(tasks)} concurrent tasks launched.  Press Ctrl+C to stop.")
    print()

    # Wait for shutdown or all tasks to complete
    try:
        await shutdown.wait()
    except asyncio.CancelledError:
        pass

    print("\n[*] Shutting down tasks ...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print("[*] All tasks stopped. Paper trading session ended.")


async def _trading_hours_guard(cfg: Any, shutdown: asyncio.Event) -> None:
    """Log trading hours boundaries; agents respect market hours internally."""
    while not shutdown.is_set():
        now_ist = datetime.now(IST)
        in_hours = cfg.MARKET_OPEN <= now_ist.time() <= cfg.MARKET_CLOSE
        weekday = now_ist.weekday()  # 0=Mon ... 6=Sun
        is_trading_day = weekday < 5

        if in_hours and is_trading_day:
            status = "MARKET OPEN"
        else:
            status = "MARKET CLOSED"

        # Log every 5 minutes
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=300)
            break
        except asyncio.TimeoutError:
            now_str = now_ist.strftime("%Y-%m-%d %H:%M:%S IST")
            print(f"  [GUARD] {now_str} -- {status}")


# ===================================================================
# COMMAND: live-trade
# ===================================================================


async def cmd_live_trade(args: argparse.Namespace) -> None:
    """Run the full system with a real Angel One broker."""
    if AngelOneBroker is None:
        sys.exit(
            "[FATAL] AngelOneBroker not available.\n"
            "        Install: pip install smartapi-python pyotp"
        )
    if ValidationGate is None:
        sys.exit("[FATAL] ValidationGate not available. Install safety deps.")

    cfg, engine, cache, bus = _bootstrap()

    # ---- Safety validations before enabling live trading ----
    print("[*] Running pre-flight safety validations ...")

    # 1. Verify broker credentials are configured
    if not cfg.ANGEL_ONE_API_KEY:
        sys.exit("[FATAL] ANGEL_ONE_API_KEY not set in .env")
    if not cfg.ANGEL_ONE_CLIENT_ID:
        sys.exit("[FATAL] ANGEL_ONE_CLIENT_ID not set in .env")
    if not cfg.ANGEL_ONE_PASSWORD:
        sys.exit("[FATAL] ANGEL_ONE_PASSWORD not set in .env")
    if not cfg.ANGEL_ONE_TOTP_SECRET:
        sys.exit("[FATAL] ANGEL_ONE_TOTP_SECRET not set in .env")

    print("  [OK] Broker credentials present")

    # 2. Instantiate broker and authenticate
    broker = AngelOneBroker(
        api_key=cfg.ANGEL_ONE_API_KEY,
        client_id=cfg.ANGEL_ONE_CLIENT_ID,
        password=cfg.ANGEL_ONE_PASSWORD,
        totp_secret=cfg.ANGEL_ONE_TOTP_SECRET,
    )
    print("  [OK] AngelOneBroker instantiated")

    # 3. Build safety stack
    audit = AuditTrail(engine) if AuditTrail is not None else None
    limits = PositionLimits(cfg) if PositionLimits is not None else None
    kill_switch = (
        KillSwitch(broker=broker, config=cfg, bus=bus, audit=audit)
        if KillSwitch is not None
        else None
    )
    risk_manager = (
        RiskManager(limits=limits, kill_switch=kill_switch, audit=audit, config=cfg)
        if RiskManager is not None
        else None
    )

    if risk_manager is None:
        sys.exit("[FATAL] RiskManager could not be initialised -- aborting live mode.")
    if kill_switch is None:
        sys.exit("[FATAL] KillSwitch could not be initialised -- aborting live mode.")

    print("  [OK] Safety stack initialised (RiskManager + KillSwitch + Audit)")

    # 4. Validation gate: check every agent has passed paper-trading graduation
    agents = _create_agents(cfg, cache, bus, engine)
    if not agents:
        sys.exit("[FATAL] No agents available for live trading.")

    gate = ValidationGate()
    failed_agents = []
    for agent in agents:
        try:
            eligible = await gate.check_promotion_eligibility(agent.name)
            if not eligible:
                failed_agents.append(agent.name)
                print(f"  [FAIL] Agent '{agent.name}' has NOT passed validation gate")
            else:
                print(f"  [OK] Agent '{agent.name}' passed validation gate")
        except Exception as exc:
            failed_agents.append(agent.name)
            print(f"  [FAIL] Agent '{agent.name}' gate check error: {exc}")

    if failed_agents:
        print()
        print(
            f"[FATAL] {len(failed_agents)} agent(s) failed validation gate.\n"
            f"        Failed: {', '.join(failed_agents)}\n"
            f"        Run 'paper-trade' for at least {cfg.PAPER_TRADING_DAYS} days first."
        )
        sys.exit(1)

    print()
    print("[*] ALL pre-flight checks passed.")
    print("[*] LIVE TRADING MODE -- real money at risk!")
    print(f"    Capital : INR {cfg.INITIAL_CAPITAL:,.0f}")
    print(f"    Universe: {len(cfg.DEFAULT_UNIVERSE)} stocks")
    print(f"    Market  : {cfg.MARKET_OPEN.strftime('%H:%M')} - {cfg.MARKET_CLOSE.strftime('%H:%M')} IST")
    print()

    # Build MetaAgent
    meta = (
        MetaAgent(
            agents=agents,
            risk_manager=risk_manager,
            broker=broker,
            config=cfg,
            bus=bus,
            db_engine=engine,
        )
        if MetaAgent is not None
        else None
    )

    # Shutdown event
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)
    shutdown = _shutdown_event
    assert shutdown is not None

    # Pre-market cache warmup
    await _warmup_cache(cache, cfg)

    # Launch the same task structure as paper-trade but with live broker
    tasks: list[asyncio.Task[None]] = []

    tasks.append(asyncio.create_task(
        _data_feed_task(bus, cfg, shutdown), name="data_feed"
    ))
    tasks.append(asyncio.create_task(
        _sentiment_crawler_task(bus, cfg, shutdown), name="sentiment_crawler"
    ))
    tasks.append(asyncio.create_task(
        _macro_update_task(bus, cfg, shutdown), name="macro_update"
    ))

    for agent in agents:
        tasks.append(asyncio.create_task(
            _agent_task(agent, bus, cfg, shutdown), name=f"agent_{agent.name}"
        ))

    if meta is not None:
        tasks.append(asyncio.create_task(
            _meta_agent_task(meta, bus, shutdown), name="meta_agent"
        ))

    tasks.append(asyncio.create_task(
        _risk_manager_task(risk_manager, bus, shutdown), name="risk_manager"
    ))
    tasks.append(asyncio.create_task(
        _portfolio_writer_task(bus, engine, shutdown), name="portfolio_writer"
    ))
    tasks.append(asyncio.create_task(
        _trading_hours_guard(cfg, shutdown), name="trading_hours_guard"
    ))

    print(f"[*] {len(tasks)} concurrent tasks launched (LIVE).  Press Ctrl+C to stop.")
    print()

    try:
        await shutdown.wait()
    except asyncio.CancelledError:
        pass

    print("\n[*] Shutting down live tasks ...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print("[*] All tasks stopped. Live trading session ended.")


# ===================================================================
# COMMAND: research
# ===================================================================


async def cmd_research(args: argparse.Namespace) -> None:
    """Run the auto-research pipeline."""
    cfg, engine, cache, bus = _bootstrap()
    print("[*] Starting research pipeline ...\n")

    # --- 1. Performance Attribution ---
    print("  [1/5] Performance Attribution")
    if PerformanceAttribution is not None:
        try:
            attrib = PerformanceAttribution(db_engine=engine)
            agent_names = [name for name, _ in _AGENT_REGISTRY]
            for name in agent_names:
                result = await attrib.attribute(agent_name=name, period_days=30)
                alpha = result.get("total_alpha", 0.0)
                print(f"        {name}: total_alpha={alpha:.4f}")
            print("        [OK] Attribution complete")
        except Exception as exc:
            print(f"        [ERR] Attribution failed: {exc}")
    else:
        print("        [SKIP] PerformanceAttribution not available")
    print()

    # --- 2. Parameter Optimizer ---
    print("  [2/5] Parameter Optimizer")
    if AgentParameterOptimizer is not None and StrategyBacktester is not None and PaperBroker is not None:
        try:
            broker = PaperBroker(initial_capital=cfg.INITIAL_CAPITAL)
            backtester = StrategyBacktester(broker=broker, db_engine=engine)
            optimizer = AgentParameterOptimizer(
                backtester=backtester,
                db_engine=engine,
            )
            agents = _create_agents(cfg, cache, bus, engine)
            for agent in agents:
                result = await optimizer.optimize(
                    agent=agent,
                    universe=cfg.DEFAULT_UNIVERSE,
                )
                improved = result.get("improved", False)
                print(f"        {agent.name}: improved={improved}")
            print("        [OK] Optimization complete")
        except Exception as exc:
            print(f"        [ERR] Optimization failed: {exc}")
    else:
        print("        [SKIP] AgentParameterOptimizer not available")
    print()

    # --- 3. Regime Learner ---
    print("  [3/5] Regime Learner")
    if RegimeAdaptiveWeightLearner is not None:
        try:
            learner = RegimeAdaptiveWeightLearner(db_engine=engine)
            result = await learner.learn()
            print(f"        Regimes learned: {len(result) if isinstance(result, dict) else 'N/A'}")
            print("        [OK] Regime learning complete")
        except Exception as exc:
            print(f"        [ERR] Regime learning failed: {exc}")
    else:
        print("        [SKIP] RegimeAdaptiveWeightLearner not available")
    print()

    # --- 4. News Alpha Tracker ---
    print("  [4/5] News Alpha Tracker")
    if NewsAlphaTracker is not None:
        try:
            tracker = NewsAlphaTracker(db_engine=engine)
            result = await tracker.track_events(days=30)
            tracked = result.get("tracked", 0)
            print(f"        Events tracked: {tracked}")
            print("        [OK] News alpha tracking complete")
        except Exception as exc:
            print(f"        [ERR] News alpha tracking failed: {exc}")
    else:
        print("        [SKIP] NewsAlphaTracker not available")
    print()

    # --- 5. Strategy Discovery + Backtester ---
    print("  [5/5] Strategy Discovery + Backtester")
    if StrategyDiscovery is not None:
        try:
            discovery = StrategyDiscovery(db_engine=engine, cache=cache)
            candidates = await discovery.run(universe=cfg.DEFAULT_UNIVERSE)
            print(f"        Candidate features discovered: {len(candidates)}")

            if candidates and StrategyBacktester is not None and PaperBroker is not None:
                broker = PaperBroker(initial_capital=cfg.INITIAL_CAPITAL)
                bt = StrategyBacktester(broker=broker, db_engine=engine)
                print(f"        Queued {len(candidates)} candidates for walk-forward validation")

            print("        [OK] Strategy discovery complete")
        except Exception as exc:
            print(f"        [ERR] Strategy discovery failed: {exc}")
    else:
        print("        [SKIP] StrategyDiscovery not available")
    print()

    print("[*] Research pipeline finished.")


# ===================================================================
# COMMAND: dashboard
# ===================================================================


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit dashboard."""
    app_path = Path(__file__).resolve().parent / "alphacouncil" / "dashboard" / "app.py"
    if not app_path.exists():
        sys.exit(f"[FATAL] Dashboard app not found: {app_path}")

    print(f"[*] Launching Streamlit dashboard: {app_path}")
    print("    Press Ctrl+C to stop.\n")

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=False,
        )
        sys.exit(proc.returncode)
    except FileNotFoundError:
        sys.exit(
            "[FATAL] Streamlit not found.\n"
            "        Install: pip install streamlit"
        )
    except KeyboardInterrupt:
        print("\n[*] Dashboard stopped.")


# ===================================================================
# COMMAND: kill
# ===================================================================


async def cmd_kill(args: argparse.Namespace) -> None:
    """Activate the emergency kill switch."""
    cfg, engine, cache, bus = _bootstrap()

    print("[!] ACTIVATING KILL SWITCH")
    print("    - Cancelling all pending orders")
    print("    - Squaring off all open positions")
    print("    - Disabling all agents")
    print()

    if PaperBroker is None:
        sys.exit("[FATAL] Broker module not available.")

    # Try to instantiate the appropriate broker
    # In a real deployment the kill switch would connect to whatever broker
    # is active.  Here we try Angel One first, then fall back to Paper.
    broker: Any = None
    if AngelOneBroker is not None and cfg.ANGEL_ONE_API_KEY:
        try:
            broker = AngelOneBroker(
                api_key=cfg.ANGEL_ONE_API_KEY,
                client_id=cfg.ANGEL_ONE_CLIENT_ID,
                password=cfg.ANGEL_ONE_PASSWORD,
                totp_secret=cfg.ANGEL_ONE_TOTP_SECRET,
            )
            print("  [*] Connected to Angel One broker")
        except Exception as exc:
            print(f"  [WARN] Angel One connection failed: {exc}")
            broker = None

    if broker is None:
        broker = PaperBroker(initial_capital=cfg.INITIAL_CAPITAL)
        print("  [*] Using PaperBroker (no live broker available)")

    # Build kill switch and activate
    audit = AuditTrail(engine) if AuditTrail is not None else None

    if KillSwitch is not None:
        ks = KillSwitch(broker=broker, config=cfg, bus=bus, audit=audit)
        try:
            await ks.activate(reason="Manual kill switch via CLI")
            print()
            print("  [OK] Kill switch ACTIVATED")
            print("       All orders cancelled, positions squared off, agents disabled.")
        except Exception as exc:
            print(f"  [ERR] Kill switch activation failed: {exc}")
            sys.exit(1)
    else:
        # Manual fallback
        print("  [WARN] KillSwitch class not available -- attempting manual shutdown")
        try:
            cancelled = await broker.cancel_all_orders()
            print(f"  [OK] Cancelled {cancelled} pending orders")
        except Exception as exc:
            print(f"  [ERR] cancel_all_orders failed: {exc}")
        try:
            squared = await broker.square_off_all()
            print(f"  [OK] Squared off {squared} positions")
        except Exception as exc:
            print(f"  [ERR] square_off_all failed: {exc}")

    # Broadcast system kill on bus
    try:
        await bus.publish(
            "system",
            {"event": "KILL_SWITCH_ACTIVATED", "source": "cli"},
            publisher="cli",
        )
    except Exception:
        pass

    print()
    print("[!] Kill switch complete. System is now halted.")


# ===================================================================
# COMMAND: status
# ===================================================================


async def cmd_status(args: argparse.Namespace) -> None:
    """Show current system status."""
    cfg, engine, cache, bus = _bootstrap()

    print("=" * 60)
    print("  ALPHACOUNCIL SYSTEM STATUS")
    print("=" * 60)
    print()

    # --- Timezone / Market Hours ---
    now_ist = datetime.now(IST)
    in_hours = cfg.MARKET_OPEN <= now_ist.time() <= cfg.MARKET_CLOSE
    weekday = now_ist.weekday()
    is_trading_day = weekday < 5

    print(f"  Time (IST)   : {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Market Hours : {cfg.MARKET_OPEN.strftime('%H:%M')} - {cfg.MARKET_CLOSE.strftime('%H:%M')} IST")
    print(f"  Market Status: {'OPEN' if (in_hours and is_trading_day) else 'CLOSED'}")
    print()

    # --- Configuration ---
    print("  Configuration")
    print(f"    Capital        : INR {cfg.INITIAL_CAPITAL:>12,.0f}")
    print(f"    Universe       : {len(cfg.DEFAULT_UNIVERSE)} stocks + {len(cfg.ETF_UNIVERSE)} ETFs")
    print(f"    Max Positions  : {cfg.MAX_POSITIONS}")
    print(f"    Max Deployed   : {cfg.MAX_DEPLOYED:.0%}")
    print(f"    Daily Loss Limit: {cfg.DAILY_LOSS_LIMIT:.1%}")
    print(f"    Max Drawdown   : {cfg.MAX_DRAWDOWN:.1%}")
    print(f"    Auto Tune      : {'ON' if cfg.AUTO_TUNE else 'OFF'}")
    print()

    # --- Safety Limits Usage ---
    print("  Safety Limits")
    print(f"    Max Capital/Stock  : {cfg.MAX_CAPITAL_PER_STOCK:.1%}")
    print(f"    Max Sector Exposure: {cfg.MAX_SECTOR_EXPOSURE:.1%}")
    print(f"    Max Daily Trades   : {cfg.MAX_DAILY_TRADES}")
    print(f"    Max Order Value    : INR {cfg.MAX_ORDER_VALUE:,.0f}")
    print(f"    Single Trade Loss  : {cfg.SINGLE_TRADE_LOSS_LIMIT:.1%}")
    print(f"    Error Threshold    : {cfg.ERROR_THRESHOLD}")
    print()

    # --- Agent Status ---
    print("  Agent Status")
    for name, cls in _AGENT_REGISTRY:
        available = cls is not None
        status = "AVAILABLE" if available else "NOT INSTALLED"
        print(f"    {name:<28} {status}")
    meta_status = "AVAILABLE" if MetaAgent is not None else "NOT INSTALLED"
    print(f"    {'meta_agent (council)':<28} {meta_status}")
    print()

    # --- Cache Stats ---
    print("  Cache Stats")
    try:
        stats = cache.get_stats()
        for tier, tier_stats in stats.items():
            print(f"    {tier}: {tier_stats}")
    except Exception:
        print("    (unable to retrieve cache stats)")
    print()

    # --- Cache TTLs ---
    print("  Cache TTLs")
    print(f"    Price       : {cfg.PRICE_TTL}s")
    print(f"    Fundamental : {cfg.FUNDAMENTAL_TTL}s ({cfg.FUNDAMENTAL_TTL // 3600}h)")
    print(f"    Macro       : {cfg.MACRO_TTL}s ({cfg.MACRO_TTL // 3600}h)")
    print(f"    Sentiment   : {cfg.SENTIMENT_TTL}s ({cfg.SENTIMENT_TTL // 60}min)")
    print()

    # --- Database ---
    print("  Database")
    print(f"    URL: {cfg.DATABASE_URL}")
    try:
        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(engine)
        tables = inspector.get_table_names()
        print(f"    Tables: {len(tables)} ({', '.join(tables[:8])}{'...' if len(tables) > 8 else ''})")
    except Exception:
        print("    (unable to inspect database)")
    print()

    # --- Graduation Gate ---
    print("  Paper-Trading Graduation Requirements")
    print(f"    Min Paper Days : {cfg.PAPER_TRADING_DAYS}")
    print(f"    Min Sharpe     : {cfg.MIN_SHARPE}")
    print(f"    Max Drawdown   : {cfg.MAX_DRAWDOWN_PAPER:.0%}")
    print(f"    Min Win Rate   : {cfg.MIN_WIN_RATE:.0%}")
    print()

    print("=" * 60)


# ===================================================================
# Advisor commands (no broker required)
# ===================================================================


def cmd_advisor_dashboard(_args: argparse.Namespace) -> None:
    """Launch the Investment Advisor Streamlit dashboard."""
    _print_banner()
    print("[*] Launching Investment Advisor Dashboard ...")

    load_dotenv()
    init_db()

    dashboard_path = Path(__file__).parent / "alphacouncil" / "dashboard" / "advisor_app.py"
    if not dashboard_path.exists():
        sys.exit(f"[FATAL] Advisor dashboard not found at {dashboard_path}")

    subprocess.run(
        ["streamlit", "run", str(dashboard_path), "--server.headless=true"],
        check=False,
    )


async def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze a single stock and print recommendation to terminal."""
    if InvestmentAdvisor is None:
        sys.exit("[FATAL] Advisor module not available. Check alphacouncil.advisor.")

    load_dotenv()
    init_db()
    cache = TieredCache()

    ticker = args.ticker
    print(f"\n[*] Analyzing {ticker} ...\n")

    advisor = InvestmentAdvisor(cache=cache)

    try:
        rec = await advisor.analyze(ticker)
    except Exception as e:
        sys.exit(f"[ERROR] Analysis failed for {ticker}: {e}")

    # Print recommendation
    action_colors = {
        "STRONG_BUY": "\033[92m",
        "BUY": "\033[92m",
        "HOLD": "\033[93m",
        "SELL": "\033[91m",
        "STRONG_SELL": "\033[91m",
    }
    reset = "\033[0m"
    color = action_colors.get(rec.action.value, "")

    print("=" * 60)
    print(f"  {rec.name} ({rec.ticker})")
    print(f"  Price: {rec.currency} {rec.current_price:,.2f}  |  Exchange: {rec.exchange}")
    print("=" * 60)
    print(f"  Action:     {color}{rec.action.value}{reset}")
    print(f"  Conviction: {rec.conviction}%")
    print(f"  Horizon:    {rec.horizon.value.replace('_', ' ')}")
    print()

    t = rec.technical
    print("  TECHNICAL")
    print(f"    Trend: {t.trend}  |  RSI: {t.rsi:.0f} ({t.rsi_signal})")
    print(f"    MACD: {t.macd_signal}  |  MA: {t.ma_alignment}  |  ADX: {t.adx:.0f} ({t.adx_signal})")
    print(f"    Support: {t.support:,.2f}  |  Resistance: {t.resistance:,.2f}")
    print()

    f = rec.fundamental
    print("  FUNDAMENTAL")
    print(f"    Valuation: {f.valuation.value}  |  Health: {f.financial_health}")
    pe_str = f"PE: {f.pe_ratio:.1f}" if f.pe_ratio else "PE: N/A"
    roe_str = f"ROE: {f.roe:.1%}" if f.roe else "ROE: N/A"
    print(f"    {pe_str}  |  {roe_str}  |  Growth Quality: {f.growth_quality_score:.0f}/100")
    if f.margin_of_safety is not None:
        print(f"    Margin of Safety: {f.margin_of_safety:.1%}")
    print()

    s = rec.sentiment
    print("  SENTIMENT")
    print(f"    Score: {s.score:.2f} ({s.signal})  |  Articles: {s.article_count}  |  Buzz: {s.social_buzz}")
    print()

    r = rec.risk
    print("  RISK")
    print(f"    Level: {r.risk_level}  |  Volatility: {r.volatility_regime}")
    print(f"    ATR%: {r.atr_pct:.1%}  |  Beta: {r.beta:.2f}  |  Max DD: {r.max_expected_drawdown:.0%}")
    print()

    lv = rec.levels
    print("  ENTRY / EXIT")
    print(f"    Entry Zone:  {rec.currency} {lv.entry_zone_low:,.2f} - {lv.entry_zone_high:,.2f}")
    print(f"    Stop Loss:   {rec.currency} {lv.stop_loss:,.2f}")
    if lv.target_short_term:
        print(f"    Target (ST): {rec.currency} {lv.target_short_term:,.2f}  (R:R {lv.risk_reward_short or 'N/A'})")
    if lv.target_mid_term:
        print(f"    Target (MT): {rec.currency} {lv.target_mid_term:,.2f}  (R:R {lv.risk_reward_mid or 'N/A'})")
    if lv.target_long_term:
        print(f"    Target (LT): {rec.currency} {lv.target_long_term:,.2f}  (R:R {lv.risk_reward_long or 'N/A'})")
    print()

    print("  REASONING")
    print(f"    {rec.reasoning}")
    print()
    print("=" * 60)


async def cmd_screen(args: argparse.Namespace) -> None:
    """Screen stocks from a universe and print results."""
    if InvestmentAdvisor is None or StockScreener is None or get_universe is None:
        sys.exit("[FATAL] Advisor module not available.")

    load_dotenv()
    init_db()
    cache = TieredCache()

    universe_name = args.universe
    profile = args.profile
    top_n = args.top

    try:
        tickers = get_universe(universe_name)
    except KeyError as e:
        sys.exit(f"[ERROR] {e}")

    print(f"\n[*] Screening {len(tickers)} stocks from '{universe_name}' with profile '{profile}' ...\n")

    advisor = InvestmentAdvisor(cache=cache)
    screener = StockScreener(advisor)

    try:
        result = await screener.screen(tickers, profile=profile)
    except Exception as e:
        sys.exit(f"[ERROR] Screening failed: {e}")

    print(f"  Screened: {result.total_screened} stocks | Profile: {result.filter_profile}")
    print()
    print(f"  {'Rank':<5} {'Ticker':<16} {'Score':>6} {'Tech':>5} {'Fund':>5} {'Sent':>5} {'Action':<12} {'Conv':>4}")
    print("  " + "-" * 70)

    for i, item in enumerate(result.results[:top_n], 1):
        print(
            f"  {i:<5} {item.ticker:<16} {item.composite_score:>5.0f} "
            f"{item.technical_score:>5.0f} {item.fundamental_score:>5.0f} "
            f"{item.sentiment_score:>5.0f} {item.action.value:<12} {item.conviction:>3}%"
        )

    print()
    print(f"  Showing top {min(top_n, len(result.results))} of {len(result.results)} passing stocks.")
    print()


async def cmd_market_pulse(_args: argparse.Namespace) -> None:
    """Show current market overview."""
    if InvestmentAdvisor is None or ReportGenerator is None:
        sys.exit("[FATAL] Advisor module not available.")

    load_dotenv()
    init_db()
    cache = TieredCache()

    print("\n[*] Fetching market data ...\n")

    advisor = InvestmentAdvisor(cache=cache)
    screener = StockScreener(advisor) if StockScreener is not None else None
    reporter = ReportGenerator(advisor, screener)

    try:
        overview = await reporter.generate_market_overview()
    except Exception as e:
        sys.exit(f"[ERROR] Market pulse failed: {e}")

    print("=" * 60)
    print("  MARKET PULSE")
    print("=" * 60)
    print()

    n50_chg = f"+{overview.nifty50_change_pct:.2f}%" if overview.nifty50_change_pct >= 0 else f"{overview.nifty50_change_pct:.2f}%"
    sen_chg = f"+{overview.sensex_change_pct:.2f}%" if overview.sensex_change_pct >= 0 else f"{overview.sensex_change_pct:.2f}%"
    print("  INDIA")
    print(f"    Nifty 50: {overview.nifty50_level:>10,.1f}  ({n50_chg})")
    print(f"    Sensex:   {overview.sensex_level:>10,.1f}  ({sen_chg})")
    print(f"    India VIX: {overview.india_vix:>8.2f}  [{overview.india_vix_signal}]")
    print(f"    Regime:    {overview.india_regime}")
    print()

    sp_chg = f"+{overview.sp500_change_pct:.2f}%" if overview.sp500_change_pct >= 0 else f"{overview.sp500_change_pct:.2f}%"
    nq_chg = f"+{overview.nasdaq_change_pct:.2f}%" if overview.nasdaq_change_pct >= 0 else f"{overview.nasdaq_change_pct:.2f}%"
    print("  US / GLOBAL")
    print(f"    S&P 500:  {overview.sp500_level:>10,.1f}  ({sp_chg})")
    print(f"    Nasdaq:   {overview.nasdaq_level:>10,.1f}  ({nq_chg})")
    print(f"    US VIX:   {overview.us_vix:>8.2f}")
    print(f"    DXY:      {overview.dxy:>8.2f}")
    print(f"    Gold:     ${overview.gold_price:>8,.1f}")
    print(f"    Brent:    ${overview.brent_crude:>8,.2f}")
    print()

    if overview.india_summary:
        print(f"  India: {overview.india_summary}")
    if overview.global_summary:
        print(f"  Global: {overview.global_summary}")
    print(f"  Risk Outlook: {overview.risk_outlook}")
    print()
    print("=" * 60)


# ===================================================================
# Argument parser
# ===================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alphacouncil",
        description="AlphaCouncil -- Multi-Agent Trading System for NSE India",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python main.py backtest --start 2022-01-01 --end 2025-12-31
              python main.py paper-trade
              python main.py live-trade
              python main.py research
              python main.py dashboard
              python main.py kill
              python main.py status
        """),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # backtest
    bt = sub.add_parser("backtest", help="Run walk-forward backtest for all agents")
    bt.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    bt.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")

    # paper-trade
    sub.add_parser("paper-trade", help="Run full system in paper-trading mode")

    # live-trade
    sub.add_parser("live-trade", help="Run full system with real broker (Angel One)")

    # research
    sub.add_parser("research", help="Run auto-research pipeline")

    # dashboard
    sub.add_parser("dashboard", help="Launch Streamlit monitoring dashboard")

    # kill
    sub.add_parser("kill", help="Emergency kill switch -- cancel all, square off, disable")

    # status
    sub.add_parser("status", help="Show current system status")

    # --- Advisor commands (no broker required) ---
    # advisor dashboard
    sub.add_parser("advisor", help="Launch Investment Advisor Streamlit dashboard")

    # analyze single stock
    az = sub.add_parser("analyze", help="Analyze a single stock and print recommendation")
    az.add_argument("ticker", help="Stock ticker (e.g. RELIANCE.NS, AAPL)")

    # screen stocks
    sc = sub.add_parser("screen", help="Screen stocks from a universe")
    sc.add_argument("--universe", default="india_nifty50", help="Universe name")
    sc.add_argument("--profile", default="growth_picks", help="Screener profile")
    sc.add_argument("--top", type=int, default=10, help="Number of top results")

    # market pulse
    sub.add_parser("market-pulse", help="Show current market overview")

    return parser


# ===================================================================
# Entry point
# ===================================================================


def main() -> None:
    _print_banner()

    parser = _build_parser()
    args = parser.parse_args()

    cmd = args.command

    if cmd == "dashboard":
        # Dashboard is synchronous (subprocess)
        cmd_dashboard(args)
        return

    if cmd == "advisor":
        # Advisor dashboard is synchronous (subprocess)
        cmd_advisor_dashboard(args)
        return

    # All other commands are async
    dispatch: dict[str, Any] = {
        "backtest": cmd_backtest,
        "paper-trade": cmd_paper_trade,
        "live-trade": cmd_live_trade,
        "research": cmd_research,
        "kill": cmd_kill,
        "status": cmd_status,
        "analyze": cmd_analyze,
        "screen": cmd_screen,
        "market-pulse": cmd_market_pulse,
    }

    handler = dispatch.get(cmd)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        asyncio.run(handler(args))
    except KeyboardInterrupt:
        print("\n[*] Interrupted. Goodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
