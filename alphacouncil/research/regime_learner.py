"""Regime-adaptive agent weight learner.

Classifies each historical trading day into one of ten Indian market
regimes and computes per-agent Sharpe ratios within each regime.  The
resulting weight map feeds into MetaAgent's Thompson Sampling as
informative priors -- agents that historically performed well in the
current regime receive higher initial weight.

Regimes
-------
BULL_LOW_VOL, BULL_HIGH_VOL, BEAR_LOW_VOL, BEAR_HIGH_VOL, SIDEWAYS,
FII_BUYING, FII_SELLING, PRE_EXPIRY, EARNINGS_SEASON, BUDGET_POLICY

These are defined in :class:`alphacouncil.core.models.MarketRegime`.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats as sp_stats
from sqlalchemy import text
from sqlalchemy.engine import Engine

from alphacouncil.core.models import MarketRegime

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Classification thresholds (India-specific)
# ---------------------------------------------------------------------------

_NIFTY_SMA_PERIOD: int = 50        # days for trend classification
_VIX_LOW_THRESHOLD: float = 15.0   # India VIX below = low vol
_VIX_HIGH_THRESHOLD: float = 22.0  # India VIX above = high vol
_FII_FLOW_THRESHOLD: float = 500.0 # INR crore net flow threshold
_SIDEWAYS_RETURN_BAND: float = 0.02  # +/- 2% over SMA = sideways

# Annualisation
_ANNUALISE_FACTOR: float = np.sqrt(252)

# Default look-back for learning
_DEFAULT_HISTORY_DAYS: int = 504  # ~2 years

# Default agent weights (growth-biased fallback)
_DEFAULT_AGENT_NAMES: list[str] = [
    "GrowthMomentumAgent",
    "MeanReversionAgent",
    "MultiFactorRankingAgent",
    "VolatilityRegimeAgent",
    "SentimentAlphaAgent",
    "PortfolioOptimizationAgent",
]

_DEFAULT_WEIGHTS: dict[str, float] = {
    "GrowthMomentumAgent": 0.25,
    "MeanReversionAgent": 0.12,
    "MultiFactorRankingAgent": 0.18,
    "VolatilityRegimeAgent": 0.10,
    "SentimentAlphaAgent": 0.20,
    "PortfolioOptimizationAgent": 0.15,
}

# Growth-bias multiplier for Sharpe-to-weight conversion
_GROWTH_AGENT_BIAS: float = 1.5


# ---------------------------------------------------------------------------
# RegimeAdaptiveWeightLearner
# ---------------------------------------------------------------------------


class RegimeAdaptiveWeightLearner:
    """Learn optimal agent weights conditioned on Indian market regimes.

    For each of the ten regimes, the learner computes each agent's
    historical Sharpe ratio and derives a normalised weight vector.
    These weights become the informative priors for MetaAgent's
    Thompson Sampling allocation.

    Growth-biased agents (``GrowthMomentumAgent``, ``SentimentAlphaAgent``)
    receive a 1.5x multiplier on their Sharpe contribution during the
    weight normalisation step, reflecting AlphaCouncil's structural
    growth tilt.

    Parameters
    ----------
    db_engine:
        SQLAlchemy :class:`Engine` for reading macro data, agent
        portfolio snapshots, and trade records.
    """

    def __init__(self, db_engine: Engine) -> None:
        self._db_engine = db_engine
        self._log = logger.bind(component="regime_learner")
        self._regime_weights: dict[str, dict[str, float]] = {}
        self._init_default_weights()

    def _init_default_weights(self) -> None:
        """Seed all regimes with sensible default weights.

        Specific regimes get hand-tuned overrides that reflect their
        historical behaviour in Indian markets.
        """
        for regime in MarketRegime:
            self._regime_weights[regime.value] = dict(_DEFAULT_WEIGHTS)

        # Regime-specific overrides
        self._regime_weights[MarketRegime.BULL_LOW_VOL.value]["GrowthMomentumAgent"] = 0.30
        self._regime_weights[MarketRegime.BEAR_HIGH_VOL.value]["MeanReversionAgent"] = 0.25
        self._regime_weights[MarketRegime.BEAR_HIGH_VOL.value]["GrowthMomentumAgent"] = 0.10
        self._regime_weights[MarketRegime.SIDEWAYS.value]["MeanReversionAgent"] = 0.22
        self._regime_weights[MarketRegime.FII_BUYING.value]["GrowthMomentumAgent"] = 0.28
        self._regime_weights[MarketRegime.FII_SELLING.value]["VolatilityRegimeAgent"] = 0.20

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def learn(
        self,
        history_days: int = _DEFAULT_HISTORY_DAYS,
    ) -> dict[str, dict[str, float]]:
        """Learn regime-conditioned agent weight map.

        Steps
        -----
        1. Classify each historical day into a regime using macro data.
        2. For each regime, compute each agent's annualised Sharpe ratio
           on the days that fell under that regime.
        3. Set prior weights proportional to ``max(sharpe, 0)`` with a
           growth-agent bias multiplier, normalised to sum to 1.0.

        Parameters
        ----------
        history_days:
            Number of calendar days of history to use (default 504,
            approximately 2 years of trading days).

        Returns
        -------
        dict[str, dict[str, float]]
            ``{regime_name: {agent_name: weight}}`` where weights are
            normalised to sum to 1.0 within each regime.
        """
        self._log.info("regime_learner.start", history_days=history_days)

        # Step 1: Fetch macro data & agent returns
        macro_data = await self._fetch_macro_data(history_days)
        agent_returns = await self._fetch_agent_returns(history_days)

        if macro_data.empty:
            self._log.warning(
                "regime_learner.no_macro_data",
                fallback="default_weights",
            )
            return dict(self._regime_weights)

        if not agent_returns:
            self._log.info(
                "regime_learner.no_agent_history",
                fallback="default_weights",
            )
            return dict(self._regime_weights)

        # Step 2: Classify each day into a regime
        regime_series = self._classify_all_days(macro_data)

        # Step 3: Compute per-agent Sharpe in each regime
        learned_weights = self._compute_regime_weights(
            regime_series, agent_returns,
        )

        # Merge learned weights into defaults (learned overrides defaults)
        for regime_name, weights in learned_weights.items():
            self._regime_weights[regime_name] = weights

        self._log.info(
            "regime_learner.complete",
            n_regimes_learned=len(learned_weights),
            regimes=list(learned_weights.keys()),
        )
        return dict(self._regime_weights)

    def get_weights(self, regime: MarketRegime) -> dict[str, float]:
        """Return the learned weight map for the given *regime*.

        Parameters
        ----------
        regime:
            The current :class:`MarketRegime`.

        Returns
        -------
        dict[str, float]
            ``{agent_name: weight}`` normalised to sum to 1.0.
            Falls back to the global default weights if no learned
            weights exist for this regime.
        """
        return dict(
            self._regime_weights.get(regime.value, _DEFAULT_WEIGHTS)
        )

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_regime(
        date: Any,
        macro_data: dict[str, Any],
    ) -> MarketRegime:
        """Classify a single day into a :class:`MarketRegime`.

        Classification rules (evaluated in priority order):

        1. **BUDGET_POLICY** -- within 5 trading days of Union Budget
           (typically first week of February) or an RBI MPC date.
        2. **PRE_EXPIRY** -- last 3 trading days of an F&O monthly
           expiry week (last Thursday).
        3. **EARNINGS_SEASON** -- during corporate earnings season
           (mid-Jan to mid-Feb, mid-Apr to mid-May, mid-Jul to mid-Aug,
           mid-Oct to mid-Nov).
        4. **FII_BUYING / FII_SELLING** -- FII net flow exceeds the
           threshold (500 crore).
        5. **BULL / BEAR x LOW_VOL / HIGH_VOL** -- Nifty vs 50-day SMA
           combined with India VIX level.
        6. **SIDEWAYS** -- default when Nifty is within +/-2% of SMA
           and VIX is in the moderate range.

        Parameters
        ----------
        date:
            The date being classified (datetime-like).
        macro_data:
            Dict with keys ``nifty_level``, ``nifty_sma``,
            ``india_vix``, ``fii_net_flow``, ``is_budget_window``,
            ``is_expiry_window``, ``is_earnings_season``.
        """
        nifty = float(macro_data.get("nifty_level", 0))
        sma = float(macro_data.get("nifty_sma", nifty or 1))
        vix = float(macro_data.get("india_vix", 15))
        fii_flow = float(macro_data.get("fii_net_flow", 0))
        is_budget = bool(macro_data.get("is_budget_window", False))
        is_expiry = bool(macro_data.get("is_expiry_window", False))
        is_earnings = bool(macro_data.get("is_earnings_season", False))

        # Priority 1: Policy events
        if is_budget:
            return MarketRegime.BUDGET_POLICY

        # Priority 2: Pre-expiry
        if is_expiry:
            return MarketRegime.PRE_EXPIRY

        # Priority 3: Earnings season
        if is_earnings:
            return MarketRegime.EARNINGS_SEASON

        # Priority 4: FII flow extremes
        if fii_flow > _FII_FLOW_THRESHOLD:
            return MarketRegime.FII_BUYING
        if fii_flow < -_FII_FLOW_THRESHOLD:
            return MarketRegime.FII_SELLING

        # Priority 5+6: Trend x Volatility
        if sma == 0:
            return MarketRegime.SIDEWAYS

        nifty_vs_sma = (nifty - sma) / sma

        is_bull = nifty_vs_sma > _SIDEWAYS_RETURN_BAND
        is_bear = nifty_vs_sma < -_SIDEWAYS_RETURN_BAND
        is_low_vol = vix < _VIX_LOW_THRESHOLD
        is_high_vol = vix > _VIX_HIGH_THRESHOLD

        if is_bull and is_low_vol:
            return MarketRegime.BULL_LOW_VOL
        if is_bull and is_high_vol:
            return MarketRegime.BULL_HIGH_VOL
        if is_bull:
            return MarketRegime.BULL_LOW_VOL  # moderate vol in bull

        if is_bear and is_low_vol:
            return MarketRegime.BEAR_LOW_VOL
        if is_bear and is_high_vol:
            return MarketRegime.BEAR_HIGH_VOL
        if is_bear:
            return MarketRegime.BEAR_HIGH_VOL  # moderate vol in bear

        return MarketRegime.SIDEWAYS

    def _classify_all_days(
        self,
        macro_data: pd.DataFrame,
    ) -> pd.Series:
        """Classify every day in the macro DataFrame.

        Returns a :class:`pd.Series` indexed by date with
        :class:`MarketRegime` string values.
        """
        df = macro_data.copy()

        # Compute Nifty SMA if not already present
        if "nifty_level" in df.columns and "nifty_sma" not in df.columns:
            df["nifty_sma"] = (
                df["nifty_level"]
                .rolling(_NIFTY_SMA_PERIOD, min_periods=1)
                .mean()
            )

        regimes: list[str] = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()

            dt = pd.Timestamp(idx) if not isinstance(idx, pd.Timestamp) else idx
            row_dict["is_budget_window"] = self._is_budget_window(dt)
            row_dict["is_expiry_window"] = self._is_expiry_window(dt)
            row_dict["is_earnings_season"] = self._is_earnings_season(dt)

            regime = self._classify_regime(dt, row_dict)
            regimes.append(regime.value)

        return pd.Series(regimes, index=df.index, name="regime")

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _compute_regime_weights(
        self,
        regime_series: pd.Series,
        agent_returns: dict[str, pd.Series],
    ) -> dict[str, dict[str, float]]:
        """Compute normalised weights per agent per regime.

        For each regime:

        1. Select the days classified under that regime.
        2. Compute each agent's annualised Sharpe ratio on those days.
        3. Apply growth-agent bias (1.5x) to
           ``GrowthMomentumAgent`` and ``SentimentAlphaAgent``.
        4. Floor negative Sharpes at a small epsilon (0.01).
        5. Normalise so weights sum to 1.0.
        6. If no agent has positive Sharpe, fall back to equal weights.
        """
        regime_weights: dict[str, dict[str, float]] = {}

        for regime in MarketRegime:
            regime_mask = regime_series == regime.value
            n_days = int(regime_mask.sum())

            if n_days < 10:
                self._log.debug(
                    "regime_learner.sparse_regime",
                    regime=regime.value,
                    n_days=n_days,
                )
                continue

            regime_dates = regime_mask[regime_mask].index
            regime_sharpes: dict[str, float] = {}

            for agent_name, returns in agent_returns.items():
                common = returns.index.intersection(regime_dates)
                agent_rets = returns.loc[common]

                if len(agent_rets) < 5:
                    regime_sharpes[agent_name] = 0.0
                    continue

                mean_ret = float(agent_rets.mean())
                std_ret = float(agent_rets.std())

                if std_ret > 0:
                    sharpe = mean_ret / std_ret * _ANNUALISE_FACTOR
                else:
                    sharpe = 0.0

                regime_sharpes[agent_name] = sharpe

            # Apply growth bias and floor
            adjusted: dict[str, float] = {}
            growth_agents = {"GrowthMomentumAgent", "SentimentAlphaAgent"}

            for agent, s in regime_sharpes.items():
                base = max(s, 0.01)
                if agent in growth_agents:
                    base *= _GROWTH_AGENT_BIAS
                adjusted[agent] = base

            # Normalise
            total = sum(adjusted.values())
            if total > 0:
                regime_weights[regime.value] = {
                    agent: round(w / total, 6)
                    for agent, w in adjusted.items()
                }
            else:
                n_agents = len(adjusted)
                if n_agents > 0:
                    equal_w = round(1.0 / n_agents, 6)
                    regime_weights[regime.value] = {
                        agent: equal_w for agent in adjusted
                    }

        return regime_weights

    # ------------------------------------------------------------------
    # Event-window classifiers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_budget_window(dt: pd.Timestamp) -> bool:
        """Check if the date falls within 5 days of Union Budget (Feb 1)
        or near a typical RBI MPC date (first week of Feb, Apr, Jun,
        Aug, Oct, Dec).
        """
        month = dt.month
        day = dt.day

        # Union Budget: typically February 1
        if month == 2 and 1 <= day <= 7:
            return True

        # RBI MPC: typically first week of even months
        if month in (4, 6, 8, 10, 12) and 1 <= day <= 7:
            return True

        return False

    @staticmethod
    def _is_expiry_window(dt: pd.Timestamp) -> bool:
        """Check if the date falls in the last 3 trading days before
        monthly F&O expiry (last Thursday of the month).
        """
        # Find last Thursday of the month
        month_end = dt + pd.offsets.MonthEnd(0)
        last_thursday = month_end
        while last_thursday.weekday() != 3:  # Thursday = 3
            last_thursday -= timedelta(days=1)

        days_before = (last_thursday - dt).days
        return 0 <= days_before <= 4  # ~3 trading days accounting for weekends

    @staticmethod
    def _is_earnings_season(dt: pd.Timestamp) -> bool:
        """Check if the date falls in a quarterly earnings season.

        Approximate windows:

        * Q3 results: 15 Jan -- 15 Feb
        * Q4 results: 15 Apr -- 15 May
        * Q1 results: 15 Jul -- 15 Aug
        * Q2 results: 15 Oct -- 15 Nov
        """
        month, day = dt.month, dt.day

        if month == 1 and day >= 15:
            return True
        if month == 2 and day <= 15:
            return True
        if month == 4 and day >= 15:
            return True
        if month == 5 and day <= 15:
            return True
        if month == 7 and day >= 15:
            return True
        if month == 8 and day <= 15:
            return True
        if month == 10 and day >= 15:
            return True
        if month == 11 and day <= 15:
            return True

        return False

    # ------------------------------------------------------------------
    # Data fetchers
    # ------------------------------------------------------------------

    async def _fetch_macro_data(
        self,
        history_days: int,
    ) -> pd.DataFrame:
        """Fetch historical macro data from portfolio snapshots.

        Uses ``portfolio_snapshots.total_value`` as a proxy for the Nifty
        level and derives VIX from realised volatility when the VIX
        column is unavailable.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._fetch_macro_data_sync, history_days,
        )

    def _fetch_macro_data_sync(self, history_days: int) -> pd.DataFrame:
        """Read macro data synchronously from the database."""
        try:
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=history_days)
            query = """
                SELECT timestamp, total_value, drawdown, sharpe_ratio
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
                return pd.DataFrame()

            df = df.set_index("timestamp").sort_index()

            # Derive proxy macro columns
            df["nifty_level"] = df["total_value"]

            # VIX proxy: rolling 20-day realised vol * 100 (scaled to VIX range)
            daily_rets = df["total_value"].pct_change()
            df["india_vix"] = daily_rets.rolling(20).std() * np.sqrt(252) * 100
            df["india_vix"] = df["india_vix"].fillna(15.0)

            # FII flow proxy: unavailable from snapshots; default to 0
            df["fii_net_flow"] = 0.0

            return df

        except Exception as exc:
            self._log.error(
                "regime_learner.macro_fetch_failed",
                error=str(exc),
            )
            return pd.DataFrame()

    async def _fetch_agent_returns(
        self,
        history_days: int,
    ) -> dict[str, pd.Series]:
        """Fetch daily returns per agent from portfolio snapshots."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._fetch_agent_returns_sync, history_days,
        )

    def _fetch_agent_returns_sync(
        self,
        history_days: int,
    ) -> dict[str, pd.Series]:
        """Read agent returns synchronously."""
        try:
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=history_days)
            query = """
                SELECT timestamp, agent_id, total_value
                FROM agent_portfolio_snapshots
                WHERE timestamp >= :cutoff
                ORDER BY agent_id, timestamp
            """
            with self._db_engine.connect() as conn:
                df = pd.read_sql(
                    query,
                    conn,
                    params={"cutoff": cutoff.isoformat()},
                    parse_dates=["timestamp"],
                )

            if df.empty:
                return {}

            result: dict[str, pd.Series] = {}
            for agent_id, group in df.groupby("agent_id"):
                ts = group.set_index("timestamp").sort_index()
                rets = ts["total_value"].pct_change().dropna()
                if not rets.empty:
                    result[str(agent_id)] = rets

            return result

        except Exception as exc:
            self._log.error(
                "regime_learner.agent_returns_failed",
                error=str(exc),
            )
            return {}
