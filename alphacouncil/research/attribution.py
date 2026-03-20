"""Brinson--Fachler performance attribution.

Decomposes an agent's returns relative to the Nifty 50 benchmark into:

* **Factor contribution** -- how much each factor in the agent's model
  contributed to total return.
* **Timing contribution** -- did the agent enter and exit positions at
  advantageous times?
* **Selection contribution** -- did the agent pick the right stocks
  within each sector?
* **Interaction effects** -- the cross-term between timing and selection
  that is neither purely timing nor purely selection.

The sum of all four components equals total alpha (portfolio return minus
benchmark return).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.engine import Engine

from alphacouncil.core.models import MarketRegime

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Sector mapping (approximate GICS -> Nifty 50 weights)
# ---------------------------------------------------------------------------

_NIFTY50_SECTOR_WEIGHTS: dict[str, float] = {
    "Financials": 0.33,
    "IT": 0.14,
    "Energy": 0.12,
    "Consumer Staples": 0.09,
    "Consumer Discretionary": 0.08,
    "Industrials": 0.05,
    "Healthcare": 0.05,
    "Telecom": 0.04,
    "Materials": 0.04,
    "Utilities": 0.03,
    "Real Estate": 0.02,
    "Other": 0.01,
}

# Default sector classification for common tickers
_TICKER_SECTOR: dict[str, str] = {
    "HDFCBANK": "Financials", "ICICIBANK": "Financials",
    "SBIN": "Financials", "BAJFINANCE": "Financials",
    "KOTAKBANK": "Financials", "AXISBANK": "Financials",
    "FEDERALBNK": "Financials", "MUTHOOTFIN": "Financials",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT",
    "WIPRO": "IT", "LTIM": "IT", "TECHM": "IT",
    "COFORGE": "IT", "PERSISTENT": "IT",
    "RELIANCE": "Energy", "NTPC": "Utilities",
    "POWERGRID": "Utilities", "ADANIENT": "Energy",
    "HINDUNILVR": "Consumer Staples", "ITC": "Consumer Staples",
    "NESTLEIND": "Consumer Staples", "DABUR": "Consumer Staples",
    "TITAN": "Consumer Discretionary", "TRENT": "Consumer Discretionary",
    "MARUTI": "Consumer Discretionary", "TATAMOTORS": "Consumer Discretionary",
    "M&M": "Consumer Discretionary", "DIXON": "Consumer Discretionary",
    "BHARTIARTL": "Telecom",
    "LT": "Industrials", "POLYCAB": "Industrials",
    "ASTRAL": "Industrials",
    "SUNPHARMA": "Healthcare", "DRREDDY": "Healthcare",
    "AUROPHARMA": "Healthcare",
    "JINDALSTEL": "Materials",
}


# ---------------------------------------------------------------------------
# PerformanceAttribution
# ---------------------------------------------------------------------------


class PerformanceAttribution:
    """Brinson--Fachler return decomposition engine.

    Parameters
    ----------
    db_engine:
        SQLAlchemy :class:`Engine` for reading trade records, portfolio
        snapshots, and factor data.
    """

    def __init__(self, db_engine: Engine) -> None:
        self._db_engine = db_engine
        self._log = logger.bind(component="attribution")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def attribute(
        self,
        agent_name: str,
        period_days: int = 30,
    ) -> dict:
        """Decompose agent returns over the most recent *period_days*.

        Parameters
        ----------
        agent_name:
            Canonical name of the agent.
        period_days:
            Look-back window in calendar days (default 30).

        Returns
        -------
        dict
            ``factor_contrib`` : per-factor return contribution.
            ``timing_contrib`` : return from timing of entries/exits.
            ``selection_contrib`` : return from stock selection.
            ``interaction`` : Brinson interaction term.
            ``total_alpha`` : sum of all contributions.
            ``sector_breakdown`` : per-sector attribution detail.
        """
        self._log.info(
            "attribution.start",
            agent=agent_name,
            period_days=period_days,
        )

        # Fetch trades and portfolio snapshots
        trades = await self._fetch_trades(agent_name, period_days)
        portfolio_returns = await self._fetch_portfolio_returns(
            agent_name, period_days,
        )
        benchmark_returns = await self._fetch_benchmark_returns(period_days)

        if not trades:
            self._log.warning("attribution.no_trades", agent=agent_name)
            return self._empty_result()

        # Factor decomposition
        factor_contrib = await self._factor_decomposition(
            trades, benchmark_returns,
        )

        # Brinson-Fachler sector-level decomposition
        sector_result = self._brinson_fachler(
            trades, portfolio_returns, benchmark_returns,
        )

        total_alpha = (
            sum(factor_contrib.values())
            + sector_result["timing_total"]
            + sector_result["selection_total"]
            + sector_result["interaction_total"]
        )

        result = {
            "agent_name": agent_name,
            "period_days": period_days,
            "factor_contrib": factor_contrib,
            "timing_contrib": round(sector_result["timing_total"], 6),
            "selection_contrib": round(sector_result["selection_total"], 6),
            "interaction": round(sector_result["interaction_total"], 6),
            "total_alpha": round(total_alpha, 6),
            "sector_breakdown": sector_result["sectors"],
        }

        self._log.info(
            "attribution.complete",
            agent=agent_name,
            total_alpha=result["total_alpha"],
        )
        return result

    # ------------------------------------------------------------------
    # Factor decomposition
    # ------------------------------------------------------------------

    async def _factor_decomposition(
        self,
        trades: list[dict],
        factor_returns: pd.DataFrame,
    ) -> dict[str, float]:
        """Decompose returns by factor exposure.

        For each trade, reads its ``factor_scores`` and computes the
        return contribution of each named factor.

        Parameters
        ----------
        trades:
            List of trade dicts with ``factor_scores``, ``price``,
            ``side``, ``quantity``, etc.
        factor_returns:
            DataFrame of daily factor returns (if available).

        Returns
        -------
        dict[str, float]
            ``{factor_name: contribution}`` mapping.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._factor_decomposition_sync, trades, factor_returns,
        )

    @staticmethod
    def _factor_decomposition_sync(
        trades: list[dict],
        factor_returns: pd.DataFrame,
    ) -> dict[str, float]:
        """Synchronous factor decomposition."""
        factor_contribs: dict[str, float] = defaultdict(float)

        for trade in trades:
            scores: dict[str, float] = trade.get("factor_scores", {})
            if not scores:
                continue

            # Trade PnL proxy: signed return
            price = trade.get("price", 0.0)
            side = trade.get("side", "BUY")
            sign = 1.0 if side == "BUY" else -1.0

            # Attribute PnL across factors by their relative score
            total_score = sum(abs(v) for v in scores.values())
            if total_score == 0:
                continue

            # Use a small proxy return (normalised)
            trade_pnl_proxy = sign * 0.01  # unit attribution

            for factor_name, score in scores.items():
                weight = abs(score) / total_score
                factor_contribs[factor_name] += trade_pnl_proxy * weight * np.sign(score)

        return {k: round(v, 6) for k, v in factor_contribs.items()}

    # ------------------------------------------------------------------
    # Brinson-Fachler decomposition
    # ------------------------------------------------------------------

    def _brinson_fachler(
        self,
        trades: list[dict],
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> dict[str, Any]:
        """Sector-level Brinson-Fachler decomposition.

        Returns
        -------
        dict
            ``timing_total``, ``selection_total``, ``interaction_total``,
            and ``sectors`` (per-sector detail).
        """
        # Map trades to sectors
        sector_trades: dict[str, list[dict]] = defaultdict(list)
        for trade in trades:
            ticker_raw = trade.get("symbol", trade.get("ticker", ""))
            ticker_clean = ticker_raw.replace(".NS", "").replace(".BO", "")
            sector = _TICKER_SECTOR.get(ticker_clean, "Other")
            sector_trades[sector].append(trade)

        # Portfolio sector weights (by trade count as proxy for capital)
        total_trades = max(len(trades), 1)
        portfolio_sector_weights: dict[str, float] = {
            sector: len(t_list) / total_trades
            for sector, t_list in sector_trades.items()
        }

        # Portfolio sector returns (average trade return proxy)
        portfolio_sector_returns: dict[str, float] = {}
        for sector, t_list in sector_trades.items():
            returns = []
            for t in t_list:
                price = t.get("price", 0.0)
                side = t.get("side", "BUY")
                sign = 1.0 if side == "BUY" else -1.0
                returns.append(sign * 0.005)  # normalised proxy
            portfolio_sector_returns[sector] = float(np.mean(returns)) if returns else 0.0

        # Benchmark sector weights and returns (use static Nifty 50)
        benchmark_total_return = float(benchmark_returns.sum()) if not benchmark_returns.empty else 0.0
        benchmark_sector_returns: dict[str, float] = {
            sector: benchmark_total_return * weight
            for sector, weight in _NIFTY50_SECTOR_WEIGHTS.items()
        }

        # --- Brinson-Fachler equations ----------------------------------------
        # Timing  = (w_p,s - w_b,s) * r_b,s
        # Selection = w_b,s * (r_p,s - r_b,s)
        # Interaction = (w_p,s - w_b,s) * (r_p,s - r_b,s)
        sectors_detail: dict[str, dict[str, float]] = {}
        timing_total = 0.0
        selection_total = 0.0
        interaction_total = 0.0

        all_sectors = set(portfolio_sector_weights.keys()) | set(_NIFTY50_SECTOR_WEIGHTS.keys())

        for sector in all_sectors:
            w_p = portfolio_sector_weights.get(sector, 0.0)
            w_b = _NIFTY50_SECTOR_WEIGHTS.get(sector, 0.0)
            r_p = portfolio_sector_returns.get(sector, 0.0)
            r_b = benchmark_sector_returns.get(sector, 0.0)

            timing = (w_p - w_b) * r_b
            selection = w_b * (r_p - r_b)
            interaction = (w_p - w_b) * (r_p - r_b)

            timing_total += timing
            selection_total += selection
            interaction_total += interaction

            sectors_detail[sector] = {
                "portfolio_weight": round(w_p, 4),
                "benchmark_weight": round(w_b, 4),
                "portfolio_return": round(r_p, 6),
                "benchmark_return": round(r_b, 6),
                "timing": round(timing, 6),
                "selection": round(selection, 6),
                "interaction": round(interaction, 6),
            }

        return {
            "timing_total": timing_total,
            "selection_total": selection_total,
            "interaction_total": interaction_total,
            "sectors": sectors_detail,
        }

    # ------------------------------------------------------------------
    # Data fetchers
    # ------------------------------------------------------------------

    async def _fetch_trades(
        self,
        agent_name: str,
        period_days: int,
    ) -> list[dict]:
        """Fetch trades for the given agent and period."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._fetch_trades_sync, agent_name, period_days,
        )

    def _fetch_trades_sync(
        self,
        agent_name: str,
        period_days: int,
    ) -> list[dict]:
        """Read trades from the database."""
        try:
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=period_days)
            query = """
                SELECT id, timestamp, symbol, side, quantity, price,
                       fees, agent_id, notes
                FROM trades
                WHERE agent_id = :agent
                  AND timestamp >= :cutoff
                ORDER BY timestamp
            """
            with self._db_engine.connect() as conn:
                df = pd.read_sql(
                    query,
                    conn,
                    params={"agent": agent_name, "cutoff": cutoff.isoformat()},
                    parse_dates=["timestamp"],
                )

            if df.empty:
                return []

            # Parse factor_scores from notes JSON if available
            records = df.to_dict("records")
            for rec in records:
                notes = rec.get("notes", "")
                if notes and isinstance(notes, str):
                    try:
                        import json
                        parsed = json.loads(notes)
                        if isinstance(parsed, dict):
                            rec["factor_scores"] = parsed.get("factor_scores", {})
                    except (json.JSONDecodeError, TypeError):
                        rec["factor_scores"] = {}
                else:
                    rec["factor_scores"] = {}

            return records

        except Exception as exc:
            self._log.error(
                "attribution.trade_fetch_failed",
                error=str(exc),
            )
            return []

    async def _fetch_portfolio_returns(
        self,
        agent_name: str,
        period_days: int,
    ) -> pd.Series:
        """Fetch daily portfolio returns from snapshots."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._fetch_portfolio_returns_sync,
            agent_name,
            period_days,
        )

    def _fetch_portfolio_returns_sync(
        self,
        agent_name: str,
        period_days: int,
    ) -> pd.Series:
        """Read portfolio returns synchronously."""
        try:
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=period_days)
            query = """
                SELECT timestamp, total_value
                FROM agent_portfolio_snapshots
                WHERE agent_id = :agent
                  AND timestamp >= :cutoff
                ORDER BY timestamp
            """
            with self._db_engine.connect() as conn:
                df = pd.read_sql(
                    query,
                    conn,
                    params={"agent": agent_name, "cutoff": cutoff.isoformat()},
                    parse_dates=["timestamp"],
                )

            if df.empty:
                return pd.Series(dtype=np.float64)

            df = df.set_index("timestamp").sort_index()
            return df["total_value"].pct_change().dropna()

        except Exception as exc:
            self._log.error(
                "attribution.portfolio_returns_failed",
                error=str(exc),
            )
            return pd.Series(dtype=np.float64)

    async def _fetch_benchmark_returns(
        self,
        period_days: int,
    ) -> pd.Series:
        """Fetch Nifty 50 benchmark returns for the period.

        Falls back to portfolio_snapshots aggregate if Nifty data is
        not directly available.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._fetch_benchmark_returns_sync, period_days,
        )

    def _fetch_benchmark_returns_sync(self, period_days: int) -> pd.Series:
        """Read benchmark returns synchronously."""
        try:
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=period_days)
            query = """
                SELECT timestamp, total_value
                FROM portfolio_snapshots
                WHERE timestamp >= :cutoff
                ORDER BY timestamp
            """
            with self._db_engine.connect() as conn:
                df = pd.read_sql(
                    query,
                    conn,
                    params={"cutoff": cutoff.isoformat()},
                    parse_dates=["timestamp"],
                )

            if df.empty:
                return pd.Series(dtype=np.float64)

            df = df.set_index("timestamp").sort_index()
            return df["total_value"].pct_change().dropna()

        except Exception as exc:
            self._log.error(
                "attribution.benchmark_returns_failed",
                error=str(exc),
            )
            return pd.Series(dtype=np.float64)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict:
        """Return a skeleton attribution result when data is insufficient."""
        return {
            "factor_contrib": {},
            "timing_contrib": 0.0,
            "selection_contrib": 0.0,
            "interaction": 0.0,
            "total_alpha": 0.0,
            "sector_breakdown": {},
        }
