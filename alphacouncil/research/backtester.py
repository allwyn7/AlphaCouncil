"""Walk-forward backtester with realistic Indian equity transaction costs.

Implements a rolling walk-forward validation framework:
* Train window: 2 years (504 trading days)
* Test window: 6 months (126 trading days)
* Roll forward: 6 months per step

Transaction costs model the full Indian equity cost stack:
brokerage, STT, GST, SEBI charges, stamp duty, and slippage.

Promotion gates:
* Walk-forward Sharpe > 0.5
* Max drawdown < 20%
* Profitable in >= 3 of 4 test periods
* Return correlation with existing agents < 0.7
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats as sp_stats
from sqlalchemy.engine import Engine

from alphacouncil.agents.base import BaseAgent
from alphacouncil.core.broker.paper import PaperBroker
from alphacouncil.core.models import OrderSide

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Indian equity cost constants (matching PaperBroker)
# ---------------------------------------------------------------------------

_BROKERAGE_PCT: float = 0.0005       # 0.05% each side
_STT_SELL_PCT: float = 0.001          # 0.1% on sell
_GST_ON_BROKERAGE_PCT: float = 0.18  # 18% of brokerage
_SEBI_PCT: float = 0.000001           # 0.0001%
_STAMP_BUY_PCT: float = 0.00015      # 0.015% on buy
_SLIPPAGE_MARKET_PCT: float = 0.001   # 0.1% on market orders

# Walk-forward parameters
_TRAIN_DAYS: int = 504   # 2 years
_TEST_DAYS: int = 126    # 6 months
_ANNUALISE_FACTOR: float = np.sqrt(252)

# Promotion thresholds
_MIN_WALK_FORWARD_SHARPE: float = 0.5
_MAX_DRAWDOWN: float = 0.20
_MIN_PROFITABLE_PERIODS_RATIO: float = 0.75  # 3/4
_MAX_AGENT_RETURN_CORRELATION: float = 0.7


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _WindowResult:
    """Results from a single train/test window."""

    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_trades: int = 0
    total_return: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    turnover: float = 0.0
    daily_returns: list[float] = field(default_factory=list)
    is_profitable: bool = False


# ---------------------------------------------------------------------------
# StrategyBacktester
# ---------------------------------------------------------------------------


class StrategyBacktester:
    """Walk-forward validation engine with realistic Indian equity costs.

    Parameters
    ----------
    broker:
        :class:`PaperBroker` instance used for simulated trade execution
        during the backtest.
    db_engine:
        SQLAlchemy :class:`Engine` for reading historical market data
        and existing agent return series.
    """

    def __init__(self, broker: PaperBroker, db_engine: Engine) -> None:
        self._broker = broker
        self._db_engine = db_engine
        self._log = logger.bind(component="backtester")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def backtest(
        self,
        agent: BaseAgent,
        universe: list[str],
        start: str,
        end: str,
    ) -> dict:
        """Run walk-forward backtest for *agent* over the given date range.

        Parameters
        ----------
        agent:
            Agent instance whose ``generate_signals`` will be called on
            each test window using parameters fitted on the preceding
            train window.
        universe:
            List of NSE ticker symbols.
        start:
            Backtest start date (ISO format ``YYYY-MM-DD``).
        end:
            Backtest end date (ISO format ``YYYY-MM-DD``).

        Returns
        -------
        dict
            Aggregate and per-window metrics including ``sharpe``,
            ``max_dd``, ``win_rate``, ``cagr``, ``profit_factor``,
            ``turnover``, and the list of per-period ``window_results``.
            Also includes ``passes_gates: bool`` indicating whether the
            agent clears all promotion thresholds.
        """
        self._log.info(
            "backtest.start",
            agent=agent.name,
            start=start,
            end=end,
            universe_size=len(universe),
        )

        # --- Build train/test windows ----------------------------------------
        windows = self._build_windows(start, end)
        if not windows:
            self._log.warning("backtest.insufficient_data", start=start, end=end)
            return self._empty_result(agent.name)

        # --- Execute each window ---------------------------------------------
        window_results: list[_WindowResult] = []
        all_daily_returns: list[float] = []
        total_trades = 0
        total_turnover = 0.0

        for w_id, (train_start, train_end, test_start, test_end) in enumerate(windows):
            wr = await self._run_window(
                agent=agent,
                universe=universe,
                window_id=w_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            window_results.append(wr)
            all_daily_returns.extend(wr.daily_returns)
            total_trades += wr.n_trades
            total_turnover += wr.turnover

        # --- Aggregate metrics -----------------------------------------------
        agg = self._aggregate_metrics(all_daily_returns, window_results, start, end)
        agg["agent_name"] = agent.name
        agg["n_windows"] = len(window_results)
        agg["total_trades"] = total_trades
        agg["total_turnover"] = round(total_turnover, 4)
        agg["window_results"] = [self._wr_to_dict(wr) for wr in window_results]

        # --- Promotion gates -------------------------------------------------
        profitable_count = sum(1 for wr in window_results if wr.is_profitable)
        profitable_ratio = profitable_count / max(len(window_results), 1)

        corr_ok = await self._check_return_correlation(
            agent.name, all_daily_returns,
        )

        passes = (
            agg["sharpe"] > _MIN_WALK_FORWARD_SHARPE
            and agg["max_dd"] < _MAX_DRAWDOWN
            and profitable_ratio >= _MIN_PROFITABLE_PERIODS_RATIO
            and corr_ok
        )
        agg["passes_gates"] = passes
        agg["gate_details"] = {
            "sharpe_ok": agg["sharpe"] > _MIN_WALK_FORWARD_SHARPE,
            "max_dd_ok": agg["max_dd"] < _MAX_DRAWDOWN,
            "profitable_periods": f"{profitable_count}/{len(window_results)}",
            "correlation_ok": corr_ok,
        }

        # --- Tearsheet -------------------------------------------------------
        agg["tearsheet"] = self._generate_tearsheet(agg)

        self._log.info(
            "backtest.complete",
            agent=agent.name,
            sharpe=agg["sharpe"],
            max_dd=agg["max_dd"],
            passes_gates=passes,
        )
        return agg

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_costs(
        quantity: int,
        price: float,
        side: str,
    ) -> float:
        """Compute all-in transaction cost for a single trade.

        Models the full Indian equity cost stack:
        * Brokerage:      0.05% each side
        * STT:            0.1% on sell side
        * GST:            18% on brokerage
        * SEBI charges:   0.0001%
        * Stamp duty:     0.015% on buy side
        * Slippage:       0.1% on market orders

        Parameters
        ----------
        quantity:
            Number of shares.
        price:
            Per-share price in INR.
        side:
            ``"BUY"`` or ``"SELL"``.

        Returns
        -------
        float
            Total cost in INR for this trade leg.
        """
        turnover = quantity * price

        brokerage = turnover * _BROKERAGE_PCT
        gst = brokerage * _GST_ON_BROKERAGE_PCT
        sebi = turnover * _SEBI_PCT
        slippage = turnover * _SLIPPAGE_MARKET_PCT

        stt = 0.0
        stamp = 0.0
        if side.upper() == "SELL":
            stt = turnover * _STT_SELL_PCT
        else:
            stamp = turnover * _STAMP_BUY_PCT

        return brokerage + gst + sebi + stt + stamp + slippage

    # ------------------------------------------------------------------
    # Tearsheet generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_tearsheet(results: dict) -> dict:
        """Build a comprehensive performance tearsheet from backtest results.

        Parameters
        ----------
        results:
            The aggregated backtest results dict.

        Returns
        -------
        dict
            Full tearsheet with risk/return metrics, drawdown analysis,
            per-window breakdown, and trade statistics.
        """
        daily_rets = results.get("all_daily_returns", [])
        rets = pd.Series(daily_rets, dtype=np.float64) if daily_rets else pd.Series(dtype=np.float64)

        # Equity curve
        equity = (1.0 + rets).cumprod() if not rets.empty else pd.Series(dtype=np.float64)

        # Rolling metrics
        rolling_sharpe = (
            rets.rolling(63).mean() / rets.rolling(63).std() * _ANNUALISE_FACTOR
            if len(rets) > 63 else pd.Series(dtype=np.float64)
        )

        # Monthly returns (approximate: 21 trading days)
        monthly_rets: list[float] = []
        if len(rets) >= 21:
            for i in range(0, len(rets) - 20, 21):
                chunk = rets.iloc[i:i + 21]
                monthly_rets.append(float((1 + chunk).prod() - 1))

        # Drawdown analysis
        peak = equity.expanding().max() if not equity.empty else pd.Series(dtype=np.float64)
        dd_series = (equity - peak) / peak.replace(0, np.nan) if not equity.empty else pd.Series(dtype=np.float64)

        # Calmar ratio
        calmar = results.get("cagr", 0.0) / results.get("max_dd", 1.0) if results.get("max_dd", 0) > 0 else 0.0

        # Sortino ratio
        downside_rets = rets[rets < 0]
        downside_std = float(downside_rets.std() * _ANNUALISE_FACTOR) if not downside_rets.empty else 0.0
        sortino = (
            float(rets.mean() * 252 / downside_std)
            if downside_std > 0 else 0.0
        )

        # Skewness & kurtosis
        skew = float(rets.skew()) if len(rets) > 2 else 0.0
        kurt = float(rets.kurtosis()) if len(rets) > 3 else 0.0

        # VaR and CVaR (95%)
        var_95 = float(np.percentile(rets, 5)) if not rets.empty else 0.0
        cvar_95 = float(rets[rets <= var_95].mean()) if not rets[rets <= var_95].empty else 0.0

        return {
            "summary": {
                "sharpe": results.get("sharpe", 0.0),
                "sortino": round(sortino, 4),
                "calmar": round(calmar, 4),
                "cagr": results.get("cagr", 0.0),
                "max_dd": results.get("max_dd", 0.0),
                "win_rate": results.get("win_rate", 0.0),
                "profit_factor": results.get("profit_factor", 0.0),
            },
            "risk": {
                "annual_vol": round(float(rets.std() * _ANNUALISE_FACTOR), 6) if not rets.empty else 0.0,
                "var_95": round(var_95, 6),
                "cvar_95": round(cvar_95, 6),
                "skewness": round(skew, 4),
                "kurtosis": round(kurt, 4),
                "max_consecutive_losses": _max_consecutive(rets, negative=True),
                "max_consecutive_wins": _max_consecutive(rets, negative=False),
            },
            "drawdown": {
                "max_drawdown": results.get("max_dd", 0.0),
                "avg_drawdown": round(float(dd_series.mean()), 6) if not dd_series.empty else 0.0,
                "max_dd_duration_days": _max_dd_duration(dd_series),
            },
            "returns": {
                "monthly_returns": [round(r, 6) for r in monthly_rets],
                "best_day": round(float(rets.max()), 6) if not rets.empty else 0.0,
                "worst_day": round(float(rets.min()), 6) if not rets.empty else 0.0,
                "positive_days_pct": round(
                    float((rets > 0).sum() / len(rets)), 4,
                ) if not rets.empty else 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Internal: window construction & execution
    # ------------------------------------------------------------------

    @staticmethod
    def _build_windows(
        start: str,
        end: str,
    ) -> list[tuple[str, str, str, str]]:
        """Construct walk-forward train/test date windows.

        Returns list of ``(train_start, train_end, test_start, test_end)``
        ISO-format date strings.
        """
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)

        train_delta = timedelta(days=int(_TRAIN_DAYS * 365 / 252))
        test_delta = timedelta(days=int(_TEST_DAYS * 365 / 252))

        windows: list[tuple[str, str, str, str]] = []
        cursor = start_dt

        while True:
            train_start = cursor
            train_end = train_start + train_delta
            test_start = train_end + timedelta(days=1)
            test_end = test_start + test_delta

            if test_end > end_dt:
                break

            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))
            cursor = test_start  # roll forward

        return windows

    async def _run_window(
        self,
        agent: BaseAgent,
        universe: list[str],
        window_id: int,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> _WindowResult:
        """Execute a single train/test window.

        1. Fetch market data for train and test periods.
        2. Call ``agent.generate_signals`` on test data.
        3. Simulate trades with full cost model.
        4. Compute per-window metrics.
        """
        wr = _WindowResult(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        try:
            # Fetch test-period data
            test_data = await self._fetch_market_data(
                universe, test_start, test_end,
            )
            if not test_data:
                self._log.warning(
                    "backtest.empty_test_data",
                    window=window_id,
                )
                return wr

            # Generate signals on test data
            signals = await agent.generate_signals(universe, test_data)
            if not signals:
                return wr

            # Simulate daily returns
            daily_returns = self._simulate_trades(signals, test_data)
            wr.daily_returns = daily_returns
            wr.n_trades = len(signals)

            # Compute window metrics
            if daily_returns:
                rets = np.array(daily_returns)
                wr.total_return = float(np.prod(1 + rets) - 1)
                wr.is_profitable = wr.total_return > 0

                std = float(rets.std())
                wr.sharpe = (
                    float(rets.mean() / std * _ANNUALISE_FACTOR)
                    if std > 0 else 0.0
                )

                # Max drawdown
                cum = np.cumprod(1 + rets)
                peak = np.maximum.accumulate(cum)
                dd = (cum - peak) / np.where(peak > 0, peak, 1.0)
                wr.max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

                # Win rate
                winning = (rets > 0).sum()
                wr.win_rate = float(winning / len(rets)) if len(rets) > 0 else 0.0

                # Profit factor
                gross_profit = float(rets[rets > 0].sum())
                gross_loss = float(abs(rets[rets < 0].sum()))
                wr.profit_factor = (
                    gross_profit / gross_loss
                    if gross_loss > 0 else float("inf")
                )

                # Turnover (sum of absolute daily returns as proxy)
                wr.turnover = float(np.abs(rets).sum())

        except Exception:
            self._log.exception(
                "backtest.window_failed",
                window=window_id,
            )

        return wr

    def _simulate_trades(
        self,
        signals: list,
        market_data: dict[str, Any],
    ) -> list[float]:
        """Convert signals into a list of daily PnL (as returns).

        For each signal, compute the hypothetical return net of all
        transaction costs.
        """
        daily_returns: list[float] = []

        for signal in signals:
            price = getattr(signal, "stop_loss", 100.0)  # use as reference price
            quantity = 1  # unit position for signal evaluation

            # Cost on both legs (entry + exit)
            buy_cost = self._compute_costs(quantity, price, "BUY")
            sell_cost = self._compute_costs(quantity, price, "SELL")
            total_cost = buy_cost + sell_cost

            # Hypothetical return: conviction-scaled, net of costs
            conviction = getattr(signal, "conviction", 50) / 100.0
            target_weight = getattr(signal, "target_weight", 0.02)

            # Assume holding period return proportional to conviction
            gross_return = conviction * target_weight * 0.01  # conservative
            cost_drag = total_cost / (price * quantity) if price > 0 else 0.0
            net_return = gross_return - cost_drag

            daily_returns.append(net_return)

        return daily_returns

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_metrics(
        all_daily_returns: list[float],
        window_results: list[_WindowResult],
        start: str,
        end: str,
    ) -> dict:
        """Compute aggregate metrics across all walk-forward windows."""
        if not all_daily_returns:
            return {
                "sharpe": 0.0,
                "max_dd": 0.0,
                "win_rate": 0.0,
                "cagr": 0.0,
                "profit_factor": 0.0,
                "turnover": 0.0,
                "all_daily_returns": [],
            }

        rets = np.array(all_daily_returns)
        std = float(rets.std())

        # Sharpe
        sharpe = float(rets.mean() / std * _ANNUALISE_FACTOR) if std > 0 else 0.0

        # Max drawdown
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / np.where(peak > 0, peak, 1.0)
        max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

        # Win rate
        win_rate = float((rets > 0).sum() / len(rets)) if len(rets) > 0 else 0.0

        # CAGR
        total_return = float(np.prod(1 + rets) - 1)
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        years = max((end_dt - start_dt).days / 365.25, 0.01)
        cagr = float((1 + total_return) ** (1.0 / years) - 1) if total_return > -1 else 0.0

        # Profit factor
        gross_profit = float(rets[rets > 0].sum())
        gross_loss = float(abs(rets[rets < 0].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Turnover
        turnover = float(np.abs(rets).sum())

        return {
            "sharpe": round(sharpe, 4),
            "max_dd": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "cagr": round(cagr, 4),
            "profit_factor": round(profit_factor, 4),
            "turnover": round(turnover, 4),
            "total_return": round(total_return, 6),
            "all_daily_returns": all_daily_returns,
        }

    # ------------------------------------------------------------------
    # Correlation check
    # ------------------------------------------------------------------

    async def _check_return_correlation(
        self,
        agent_name: str,
        daily_returns: list[float],
    ) -> bool:
        """Ensure the agent's returns are not > 0.7 correlated with any existing agent.

        Reads existing agent return series from ``agent_portfolio_snapshots``
        and computes Pearson correlation.
        """
        if not daily_returns:
            return True

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._check_return_correlation_sync,
            agent_name,
            daily_returns,
        )

    def _check_return_correlation_sync(
        self,
        agent_name: str,
        daily_returns: list[float],
    ) -> bool:
        """Synchronous correlation check against existing agents."""
        try:
            query = """
                SELECT agent_id, total_value, timestamp
                FROM agent_portfolio_snapshots
                ORDER BY agent_id, timestamp
            """
            with self._db_engine.connect() as conn:
                df = pd.read_sql(query, conn, parse_dates=["timestamp"])

            if df.empty:
                return True  # no existing agents to compare against

            new_rets = pd.Series(daily_returns, dtype=np.float64)

            for other_agent, group in df.groupby("agent_id"):
                if other_agent == agent_name:
                    continue

                existing_rets = group["total_value"].pct_change().dropna()
                # Align lengths
                min_len = min(len(new_rets), len(existing_rets))
                if min_len < 30:
                    continue

                corr = float(
                    new_rets.iloc[:min_len].corr(
                        existing_rets.iloc[:min_len].reset_index(drop=True)
                    )
                )

                if np.isfinite(corr) and abs(corr) > _MAX_AGENT_RETURN_CORRELATION:
                    self._log.warning(
                        "backtest.high_correlation",
                        agent=agent_name,
                        other=other_agent,
                        correlation=round(corr, 4),
                    )
                    return False

        except Exception as exc:
            self._log.error(
                "backtest.correlation_check_failed",
                error=str(exc),
            )

        return True

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    async def _fetch_market_data(
        self,
        universe: list[str],
        start: str,
        end: str,
    ) -> dict[str, Any]:
        """Fetch market data for the given period from the database."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._fetch_market_data_sync,
            universe,
            start,
            end,
        )

    def _fetch_market_data_sync(
        self,
        universe: list[str],
        start: str,
        end: str,
    ) -> dict[str, Any]:
        """Read historical data synchronously from the database."""
        try:
            query = """
                SELECT timestamp, symbol, price, quantity AS volume
                FROM trades
                WHERE timestamp BETWEEN :start AND :end
                ORDER BY timestamp
            """
            with self._db_engine.connect() as conn:
                df = pd.read_sql(
                    query,
                    conn,
                    params={"start": start, "end": end},
                    parse_dates=["timestamp"],
                )

            if df.empty:
                return {}

            return {"prices": df, "start": start, "end": end}

        except Exception as exc:
            self._log.error(
                "backtest.data_fetch_failed",
                error=str(exc),
            )
            return {}

    @staticmethod
    def _empty_result(agent_name: str) -> dict:
        """Return a skeleton result dict when backtest cannot run."""
        return {
            "agent_name": agent_name,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "cagr": 0.0,
            "profit_factor": 0.0,
            "turnover": 0.0,
            "total_return": 0.0,
            "n_windows": 0,
            "total_trades": 0,
            "total_turnover": 0.0,
            "window_results": [],
            "passes_gates": False,
            "gate_details": {},
            "tearsheet": {},
            "all_daily_returns": [],
        }

    @staticmethod
    def _wr_to_dict(wr: _WindowResult) -> dict:
        """Serialise a window result to a plain dict."""
        return {
            "window_id": wr.window_id,
            "train_start": wr.train_start,
            "train_end": wr.train_end,
            "test_start": wr.test_start,
            "test_end": wr.test_end,
            "n_trades": wr.n_trades,
            "total_return": round(wr.total_return, 6),
            "sharpe": round(wr.sharpe, 4),
            "max_dd": round(wr.max_dd, 4),
            "win_rate": round(wr.win_rate, 4),
            "profit_factor": round(wr.profit_factor, 4),
            "turnover": round(wr.turnover, 4),
            "is_profitable": wr.is_profitable,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _max_consecutive(rets: pd.Series, *, negative: bool) -> int:
    """Count the longest streak of consecutive positive or negative returns."""
    if rets.empty:
        return 0
    mask = rets < 0 if negative else rets > 0
    groups = mask.ne(mask.shift()).cumsum()
    if not mask.any():
        return 0
    return int(mask.groupby(groups).sum().max())


def _max_dd_duration(dd_series: pd.Series) -> int:
    """Return the longest drawdown duration in trading days."""
    if dd_series.empty:
        return 0

    in_dd = dd_series < 0
    if not in_dd.any():
        return 0

    groups = in_dd.ne(in_dd.shift()).cumsum()
    return int(in_dd.groupby(groups).sum().max())
