"""Agent Portfolio Tracker -- virtual portfolio management for all agents.

Tracks individual virtual (paper) portfolios for each of the 6 alpha agents,
the Council (MetaAgent), and the Nifty 50 benchmark.  Each portfolio
independently simulates trades, computes NAV, and produces performance metrics
so the system can rank agents, detect degradation, and feed the RL weight
learner with accurate attribution data.

All data is persisted to the SQLite ``agent_portfolio_snapshots`` table
defined in :mod:`alphacouncil.core.database`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.engine import Engine

from alphacouncil.core.models import Action, AgentSignal

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical portfolio identifiers.
PORTFOLIO_NAMES: list[str] = [
    "growth_momentum",
    "mean_reversion",
    "sentiment_alpha",
    "fundamental_value",
    "volatility_regime",
    "macro_flow",
    "council",
    "nifty50_benchmark",
]

#: Annualisation factor (252 trading days).
_TRADING_DAYS: int = 252

#: Minimum observations required for metric computation.
_MIN_OBSERVATIONS: int = 5

#: Risk-free rate (annualised) for Sharpe / Sortino -- India 10Y benchmark.
_RISK_FREE_RATE: float = 0.07


# ---------------------------------------------------------------------------
# Internal virtual portfolio representation
# ---------------------------------------------------------------------------


class _VirtualPortfolio:
    """In-memory representation of a single agent's virtual portfolio.

    This is intentionally a simple, mutable container -- not a Pydantic model
    -- because it is an internal bookkeeping structure, not a domain object
    that crosses module boundaries.
    """

    __slots__ = (
        "name",
        "initial_capital",
        "cash",
        "positions",
        "nav_history",
        "trade_log",
        "_peak_nav",
    )

    def __init__(self, name: str, initial_capital: float) -> None:
        self.name: str = name
        self.initial_capital: float = initial_capital
        self.cash: float = initial_capital
        self.positions: dict[str, _VirtualPosition] = {}
        self.nav_history: list[dict[str, Any]] = []
        self.trade_log: list[dict[str, Any]] = []
        self._peak_nav: float = initial_capital

    def nav(self, prices: dict[str, float]) -> float:
        """Compute net asset value given current prices."""
        invested = sum(
            pos.quantity * prices.get(pos.ticker, pos.avg_price)
            for pos in self.positions.values()
        )
        return self.cash + invested

    def invested_value(self, prices: dict[str, float]) -> float:
        """Compute total invested (non-cash) value."""
        return sum(
            pos.quantity * prices.get(pos.ticker, pos.avg_price)
            for pos in self.positions.values()
        )

    def update_peak(self, current_nav: float) -> None:
        """Track high-water mark for drawdown calculation."""
        if current_nav > self._peak_nav:
            self._peak_nav = current_nav

    @property
    def peak_nav(self) -> float:
        return self._peak_nav


class _VirtualPosition:
    """Single-ticker position inside a virtual portfolio."""

    __slots__ = ("ticker", "quantity", "avg_price", "entry_date")

    def __init__(
        self,
        ticker: str,
        quantity: int,
        avg_price: float,
        entry_date: datetime,
    ) -> None:
        self.ticker: str = ticker
        self.quantity: int = quantity
        self.avg_price: float = avg_price
        self.entry_date: datetime = entry_date


# ---------------------------------------------------------------------------
# AgentPortfolioTracker
# ---------------------------------------------------------------------------


class AgentPortfolioTracker:
    """Tracks virtual portfolios for all agents, council, and benchmark.

    Parameters
    ----------
    db_engine:
        SQLAlchemy :class:`Engine` for persisting snapshots to the
        ``agent_portfolio_snapshots`` table.
    initial_capital:
        Starting capital (INR) for each virtual portfolio.
    """

    def __init__(
        self,
        db_engine: Engine,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        self._db_engine = db_engine
        self._initial_capital = initial_capital
        self._log = logger.bind(component="agent_portfolio_tracker")

        # Initialise all 8 virtual portfolios.
        self._portfolios: dict[str, _VirtualPortfolio] = {
            name: _VirtualPortfolio(name, initial_capital)
            for name in PORTFOLIO_NAMES
        }

        self._log.info(
            "agent_portfolio_tracker.initialised",
            portfolio_count=len(self._portfolios),
            initial_capital=initial_capital,
        )

    # ------------------------------------------------------------------
    # Signal recording (simulated trade execution)
    # ------------------------------------------------------------------

    async def record_signal(
        self,
        agent_name: str,
        signal: AgentSignal,
        current_prices: dict[str, float],
    ) -> None:
        """Simulate execution of a signal in the agent's virtual portfolio.

        For BUY signals the tracker allocates ``target_weight`` of the
        portfolio to the ticker.  For SELL signals it liquidates the
        existing position.  HOLD signals are logged but not acted upon.

        Parameters
        ----------
        agent_name:
            The portfolio to record the trade against.
        signal:
            The :class:`AgentSignal` to simulate.
        current_prices:
            Dict of ticker -> current market price.
        """
        portfolio = self._portfolios.get(agent_name)
        if portfolio is None:
            self._log.warning(
                "agent_portfolio.unknown_agent",
                agent_name=agent_name,
            )
            return

        ticker = signal.ticker
        price = current_prices.get(ticker)
        if price is None or price <= 0:
            self._log.warning(
                "agent_portfolio.no_price",
                agent_name=agent_name,
                ticker=ticker,
            )
            return

        now = datetime.now(timezone.utc)

        if signal.action == Action.BUY:
            await self._execute_buy(portfolio, ticker, price, signal, now)
        elif signal.action == Action.SELL:
            await self._execute_sell(portfolio, ticker, price, signal, now)
        else:
            # HOLD -- log only.
            self._log.debug(
                "agent_portfolio.hold_signal",
                agent_name=agent_name,
                ticker=ticker,
                conviction=signal.conviction,
            )

    async def _execute_buy(
        self,
        portfolio: _VirtualPortfolio,
        ticker: str,
        price: float,
        signal: AgentSignal,
        timestamp: datetime,
    ) -> None:
        """Simulate a BUY trade in a virtual portfolio."""
        current_nav = portfolio.nav({
            t: pos.avg_price for t, pos in portfolio.positions.items()
        })
        # The real NAV should use market prices, but for allocation
        # purposes the target_weight against current NAV is sufficient.
        alloc = signal.target_weight * max(current_nav, portfolio.cash)
        alloc = min(alloc, portfolio.cash)  # cannot exceed available cash

        if alloc < price:
            self._log.debug(
                "agent_portfolio.insufficient_cash",
                agent_name=portfolio.name,
                ticker=ticker,
                required=price,
                available=portfolio.cash,
            )
            return

        quantity = int(alloc / price)
        if quantity <= 0:
            return

        cost = quantity * price
        portfolio.cash -= cost

        # Update or create position.
        if ticker in portfolio.positions:
            existing = portfolio.positions[ticker]
            total_qty = existing.quantity + quantity
            existing.avg_price = (
                (existing.avg_price * existing.quantity + cost) / total_qty
            )
            existing.quantity = total_qty
        else:
            portfolio.positions[ticker] = _VirtualPosition(
                ticker=ticker,
                quantity=quantity,
                avg_price=price,
                entry_date=timestamp,
            )

        portfolio.trade_log.append({
            "timestamp": timestamp.isoformat(),
            "ticker": ticker,
            "side": "BUY",
            "quantity": quantity,
            "price": price,
            "conviction": signal.conviction,
        })

        self._log.info(
            "agent_portfolio.buy_executed",
            agent_name=portfolio.name,
            ticker=ticker,
            quantity=quantity,
            price=price,
        )

    async def _execute_sell(
        self,
        portfolio: _VirtualPortfolio,
        ticker: str,
        price: float,
        signal: AgentSignal,
        timestamp: datetime,
    ) -> None:
        """Simulate a SELL trade in a virtual portfolio."""
        if ticker not in portfolio.positions:
            self._log.debug(
                "agent_portfolio.no_position_to_sell",
                agent_name=portfolio.name,
                ticker=ticker,
            )
            return

        existing = portfolio.positions[ticker]
        quantity = existing.quantity
        proceeds = quantity * price
        portfolio.cash += proceeds

        pnl = (price - existing.avg_price) * quantity
        holding_days = (timestamp - existing.entry_date).days

        del portfolio.positions[ticker]

        portfolio.trade_log.append({
            "timestamp": timestamp.isoformat(),
            "ticker": ticker,
            "side": "SELL",
            "quantity": quantity,
            "price": price,
            "pnl": round(pnl, 2),
            "holding_days": holding_days,
            "conviction": signal.conviction,
        })

        self._log.info(
            "agent_portfolio.sell_executed",
            agent_name=portfolio.name,
            ticker=ticker,
            quantity=quantity,
            price=price,
            pnl=round(pnl, 2),
        )

    # ------------------------------------------------------------------
    # NAV update
    # ------------------------------------------------------------------

    async def update_nav(
        self,
        agent_name: str,
        current_prices: dict[str, float],
    ) -> None:
        """Recompute and record NAV for a single agent's portfolio.

        Parameters
        ----------
        agent_name:
            Portfolio identifier.
        current_prices:
            Dict of ticker -> current market price.
        """
        portfolio = self._portfolios.get(agent_name)
        if portfolio is None:
            return

        nav = portfolio.nav(current_prices)
        portfolio.update_peak(nav)

        portfolio.nav_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nav": nav,
        })

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    async def get_metrics(self, agent_name: str) -> dict[str, Any]:
        """Compute comprehensive performance metrics for a portfolio.

        Parameters
        ----------
        agent_name:
            Portfolio identifier.

        Returns
        -------
        dict[str, Any]
            Keys: ``nav``, ``cumulative_return``, ``cagr``, ``sharpe``,
            ``sortino``, ``calmar``, ``max_drawdown``, ``win_rate``,
            ``avg_holding_period``, ``turnover``, ``profit_factor``.
        """
        portfolio = self._portfolios.get(agent_name)
        if portfolio is None:
            return {"error": f"Unknown portfolio: {agent_name}"}

        # Build NAV series.
        if len(portfolio.nav_history) < _MIN_OBSERVATIONS:
            nav = portfolio.nav({
                t: pos.avg_price
                for t, pos in portfolio.positions.items()
            })
            return {
                "nav": round(nav, 2),
                "cumulative_return": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "calmar": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "avg_holding_period": 0.0,
                "turnover": 0.0,
                "profit_factor": 0.0,
                "note": "Insufficient history for metric computation",
            }

        nav_series = pd.Series(
            [h["nav"] for h in portfolio.nav_history],
            dtype=np.float64,
        )
        returns = nav_series.pct_change().dropna()

        current_nav = float(nav_series.iloc[-1])
        cumulative_return = (current_nav / portfolio.initial_capital) - 1.0

        # CAGR.
        n_days = len(portfolio.nav_history)
        years = max(n_days / _TRADING_DAYS, 1 / _TRADING_DAYS)
        cagr = (
            (current_nav / portfolio.initial_capital) ** (1.0 / years) - 1.0
            if portfolio.initial_capital > 0
            else 0.0
        )

        # Drawdown.
        max_dd = self._compute_max_drawdown(nav_series)

        # Risk-adjusted metrics.
        sharpe = self._compute_sharpe(returns)
        sortino = self._compute_sortino(returns)
        calmar = self._compute_calmar(returns, max_dd)

        # Trade-level metrics.
        trades = portfolio.trade_log
        sell_trades = [t for t in trades if t["side"] == "SELL"]
        winning = [t for t in sell_trades if t.get("pnl", 0) > 0]
        losing = [t for t in sell_trades if t.get("pnl", 0) < 0]

        win_rate = len(winning) / len(sell_trades) if sell_trades else 0.0

        avg_holding = (
            np.mean([t.get("holding_days", 0) for t in sell_trades])
            if sell_trades
            else 0.0
        )

        # Turnover: total trade value / average NAV.
        total_trade_value = sum(
            t["quantity"] * t["price"] for t in trades
        )
        avg_nav = float(nav_series.mean()) if len(nav_series) > 0 else 1.0
        turnover = total_trade_value / avg_nav if avg_nav > 0 else 0.0

        # Profit factor: gross profits / gross losses.
        gross_profit = sum(t.get("pnl", 0) for t in winning)
        gross_loss = abs(sum(t.get("pnl", 0) for t in losing))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        return {
            "nav": round(current_nav, 2),
            "cumulative_return": round(cumulative_return, 4),
            "cagr": round(cagr, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "avg_holding_period": round(float(avg_holding), 1),
            "turnover": round(turnover, 4),
            "profit_factor": round(profit_factor, 4),
        }

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    async def get_leaderboard(self) -> list[dict[str, Any]]:
        """Return all 8 portfolios ranked by Sharpe ratio (descending).

        Returns
        -------
        list[dict[str, Any]]
            Each entry contains ``agent_name`` plus all keys from
            :meth:`get_metrics`.
        """
        entries: list[dict[str, Any]] = []

        for name in PORTFOLIO_NAMES:
            metrics = await self.get_metrics(name)
            metrics["agent_name"] = name
            entries.append(metrics)

        # Sort by Sharpe (descending), using cumulative_return as tiebreaker.
        entries.sort(
            key=lambda e: (
                e.get("sharpe", 0.0),
                e.get("cumulative_return", 0.0),
            ),
            reverse=True,
        )

        # Add rank.
        for idx, entry in enumerate(entries, start=1):
            entry["rank"] = idx

        return entries

    # ------------------------------------------------------------------
    # Daily snapshot (persistence)
    # ------------------------------------------------------------------

    async def snapshot_all(self, current_prices: dict[str, float]) -> None:
        """Persist a daily snapshot of all portfolios to SQLite.

        Updates NAVs first, then writes one row per portfolio to the
        ``agent_portfolio_snapshots`` table.

        Parameters
        ----------
        current_prices:
            Dict of ticker -> current market price.
        """
        now = datetime.now(timezone.utc)

        for name in PORTFOLIO_NAMES:
            await self.update_nav(name, current_prices)

        rows: list[dict[str, Any]] = []
        for name, portfolio in self._portfolios.items():
            nav = portfolio.nav(current_prices)
            invested = portfolio.invested_value(current_prices)
            unrealised_pnl = sum(
                (current_prices.get(t, pos.avg_price) - pos.avg_price) * pos.quantity
                for t, pos in portfolio.positions.items()
            )
            realised_pnl = sum(
                t.get("pnl", 0.0)
                for t in portfolio.trade_log
                if t["side"] == "SELL"
            )

            # Allocation breakdown.
            allocation: dict[str, float] = {}
            for ticker, pos in portfolio.positions.items():
                val = pos.quantity * current_prices.get(ticker, pos.avg_price)
                allocation[ticker] = round(val, 2)

            rows.append({
                "timestamp": now,
                "agent_id": name,
                "total_value": round(nav, 2),
                "cash": round(portfolio.cash, 2),
                "invested": round(invested, 2),
                "unrealised_pnl": round(unrealised_pnl, 2),
                "realised_pnl": round(realised_pnl, 2),
                "allocation_json": json.dumps(allocation),
                "notes": None,
            })

        # Bulk insert.
        try:
            with self._db_engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO agent_portfolio_snapshots "
                        "(timestamp, agent_id, total_value, cash, invested, "
                        "unrealised_pnl, realised_pnl, allocation_json, notes) "
                        "VALUES "
                        "(:timestamp, :agent_id, :total_value, :cash, :invested, "
                        ":unrealised_pnl, :realised_pnl, :allocation_json, :notes)"
                    ),
                    rows,
                )
            self._log.info(
                "agent_portfolio.snapshot_persisted",
                portfolio_count=len(rows),
            )
        except Exception:
            self._log.exception("agent_portfolio.snapshot_persist_failed")

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sharpe(returns: pd.Series) -> float:
        """Compute annualised Sharpe ratio.

        Parameters
        ----------
        returns:
            Daily return series.

        Returns
        -------
        float
            Annualised Sharpe ratio.  Returns 0.0 when data is
            insufficient or volatility is zero.
        """
        if len(returns) < _MIN_OBSERVATIONS:
            return 0.0

        excess = returns - (_RISK_FREE_RATE / _TRADING_DAYS)
        mean = excess.mean()
        std = excess.std(ddof=1)

        if std == 0 or np.isnan(std):
            return 0.0

        return float(mean / std * np.sqrt(_TRADING_DAYS))

    @staticmethod
    def _compute_sortino(returns: pd.Series) -> float:
        """Compute annualised Sortino ratio.

        Uses only downside deviation (negative returns below the risk-free
        rate) in the denominator.

        Parameters
        ----------
        returns:
            Daily return series.

        Returns
        -------
        float
            Annualised Sortino ratio.
        """
        if len(returns) < _MIN_OBSERVATIONS:
            return 0.0

        daily_rf = _RISK_FREE_RATE / _TRADING_DAYS
        excess = returns - daily_rf
        downside = excess[excess < 0]

        if len(downside) == 0:
            # No downside -- infinite Sortino; cap at a large number.
            return 10.0 if excess.mean() > 0 else 0.0

        downside_std = float(np.sqrt(np.mean(downside ** 2)))
        if downside_std == 0:
            return 0.0

        return float(excess.mean() / downside_std * np.sqrt(_TRADING_DAYS))

    @staticmethod
    def _compute_calmar(returns: pd.Series, max_dd: float) -> float:
        """Compute Calmar ratio (annualised return / max drawdown).

        Parameters
        ----------
        returns:
            Daily return series.
        max_dd:
            Maximum drawdown as a positive fraction (e.g. 0.15 = 15%).

        Returns
        -------
        float
            Calmar ratio.  Returns 0.0 when drawdown is zero.
        """
        if max_dd <= 0 or len(returns) < _MIN_OBSERVATIONS:
            return 0.0

        ann_return = float(returns.mean() * _TRADING_DAYS)
        return ann_return / max_dd

    @staticmethod
    def _compute_max_drawdown(nav_series: pd.Series) -> float:
        """Compute maximum drawdown from a NAV time series.

        Parameters
        ----------
        nav_series:
            Series of portfolio NAV values (chronologically ordered).

        Returns
        -------
        float
            Maximum drawdown as a positive fraction in [0.0, 1.0].
        """
        if len(nav_series) < 2:
            return 0.0

        peak = nav_series.expanding().max()
        drawdown = (nav_series - peak) / peak
        max_dd = float(drawdown.min())

        # Return as positive fraction.
        return abs(max_dd)

    # ------------------------------------------------------------------
    # Benchmark helpers
    # ------------------------------------------------------------------

    async def update_benchmark(
        self,
        nifty_price: float,
    ) -> None:
        """Update the Nifty 50 benchmark portfolio.

        The benchmark buys-and-holds with the full initial capital allocated
        on first call, then simply tracks NAV.

        Parameters
        ----------
        nifty_price:
            Current Nifty 50 index level (or NIFTYBEES ETF price).
        """
        portfolio = self._portfolios["nifty50_benchmark"]

        # On first call, simulate a full allocation.
        if not portfolio.positions and nifty_price > 0:
            quantity = int(portfolio.cash / nifty_price)
            if quantity > 0:
                cost = quantity * nifty_price
                portfolio.cash -= cost
                portfolio.positions["NIFTY50"] = _VirtualPosition(
                    ticker="NIFTY50",
                    quantity=quantity,
                    avg_price=nifty_price,
                    entry_date=datetime.now(timezone.utc),
                )
                self._log.info(
                    "agent_portfolio.benchmark_initialised",
                    quantity=quantity,
                    price=nifty_price,
                )

        # Update NAV.
        await self.update_nav("nifty50_benchmark", {"NIFTY50": nifty_price})

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<AgentPortfolioTracker "
            f"portfolios={len(self._portfolios)} "
            f"capital={self._initial_capital:,.0f}>"
        )
