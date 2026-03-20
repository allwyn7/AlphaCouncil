"""Multi-factor cross-sectional ranking agent with a growth tilt.

:class:`MultiFactorRankingAgent` ranks every stock in the universe by a
weighted composite z-score of nine fundamental, momentum, and India-specific
factors.  The factor weights are deliberately tilted toward growth (35 %
pure growth) with only a light value allocation (8 %).

Signal logic
------------
* **BUY** the top ``buy_percentile`` of the ranked universe.
* **SELL** the bottom ``sell_percentile`` *if currently held*.
* **HOLD** everything in between.
* Rebalance weekly on Monday open; equal-weight positions.

Portfolio constraints
---------------------
* 15--20 positions, max 8 % per position.
* Holding period: 5--20 calendar days.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

from alphacouncil.agents.base import BaseAgent
from alphacouncil.core.models import Action, AgentSignal, AgentStatus, VolatilityRegime

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Default factor weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "revenue_growth": 0.20,       # Growth
    "eps_growth": 0.15,           # Growth
    "roe": 0.12,                  # Quality
    "fcf_yield": 0.08,            # Value (light weight -- growth-biased)
    "momentum_6m": 0.15,          # Momentum
    "low_volatility": 0.05,       # Defensive (inverse of 60-day realised vol)
    "delivery_pct": 0.10,         # India-specific: institutional buying proxy
    "fii_dii_net_flow": 0.10,     # India-specific: institutional interest
    "gross_margin_expansion": 0.05,  # Improving profitability
}

# Sanity check at import time -- weights must sum to 1.
assert abs(sum(_DEFAULT_FACTOR_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"Factor weights sum to {sum(_DEFAULT_FACTOR_WEIGHTS.values())}, expected 1.0"
)


class MultiFactorRankingAgent(BaseAgent):
    """Cross-sectional multi-factor ranking agent, growth-tilted.

    Parameters (tunable via ``set_parameters``)
    --------------------------------------------
    buy_percentile : float
        Fraction of the universe to BUY (top-ranked).  Default ``0.20``.
    sell_percentile : float
        Fraction of the universe to SELL (bottom-ranked, if held).
        Default ``0.20``.
    factor_weights : dict[str, float]
        Mapping of factor name -> weight.  Must sum to 1.0.
    rebalance_day : int
        ISO weekday for rebalancing (0 = Monday).  Default ``0``.
    max_positions : int
        Maximum number of concurrent positions.  Default ``20``.
    max_weight_per_stock : float
        Maximum portfolio weight for a single stock.  Default ``0.08``.
    holding_period_min : int
        Minimum holding period in calendar days.  Default ``5``.
    holding_period_max : int
        Maximum holding period in calendar days.  Default ``20``.
    """

    AGENT_NAME: str = "multifactor_ranking"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Any,
        cache: Any,
        bus: Any,
        db_engine: Any,
        *,
        name: str = "multifactor_ranking",
    ) -> None:
        super().__init__(
            name=name,
            config=config,
            cache=cache,
            bus=bus,
            db_engine=db_engine,
        )
        # Initialise tunable parameters with defaults.
        self._parameters: dict[str, Any] = {
            "buy_percentile": 0.20,
            "sell_percentile": 0.20,
            "factor_weights": dict(_DEFAULT_FACTOR_WEIGHTS),
            "rebalance_day": 0,  # Monday
            "max_positions": 20,
            "max_weight_per_stock": 0.08,
            "holding_period_min": 5,
            "holding_period_max": 20,
        }
        self._log = logger.bind(agent=self._name)

    # ------------------------------------------------------------------
    # Parameter access (abstract interface)
    # ------------------------------------------------------------------

    def get_parameters(self) -> dict[str, Any]:
        """Return current tunable parameters."""
        return dict(self._parameters)

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Hot-reload tunable parameters with validation."""
        if "factor_weights" in params:
            weights = params["factor_weights"]
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-6:
                self._log.warning(
                    "set_parameters.invalid_weights",
                    total=total,
                    msg="Factor weights must sum to 1.0 -- ignored",
                )
                params = {k: v for k, v in params.items() if k != "factor_weights"}

        for key, value in params.items():
            if key in self._parameters:
                old = self._parameters[key]
                self._parameters[key] = value
                self._log.info("parameter_changed", key=key, old=old, new=value)
            else:
                self._log.warning("set_parameters.unknown_key", key=key)

    # ------------------------------------------------------------------
    # Core signal generation
    # ------------------------------------------------------------------

    async def generate_signals(
        self,
        universe: list[str],
        market_data: dict[str, Any],
    ) -> list[AgentSignal]:
        """Rank *universe* by composite z-score and emit BUY / SELL / HOLD signals.

        Expected *market_data* keys
        ----------------------------
        ``"fundamentals"``
            :class:`pandas.DataFrame` indexed by ticker with columns including:
            ``revenue_growth``, ``eps_growth``, ``roe``, ``fcf_yield``,
            ``gross_margin``, ``prev_gross_margin``, ``delivery_pct``,
            ``fii_dii_net_flow``.
        ``"prices"``
            :class:`pandas.DataFrame` with columns ``ticker``, ``close``,
            and a DatetimeIndex.  Used for momentum and volatility.
        ``"current_positions"``
            ``dict[str, float]`` mapping ticker -> current weight.
        ``"portfolio_value"``
            ``float`` total NAV in INR.
        """
        now = datetime.now()

        # Gate: rebalance only on the configured weekday.
        rebalance_day: int = self._parameters["rebalance_day"]
        if now.weekday() != rebalance_day:
            self._log.debug(
                "generate_signals.skip_non_rebalance_day",
                today=now.strftime("%A"),
                rebalance_day=rebalance_day,
            )
            return []

        # -----------------------------------------------------------------
        # Extract data (graceful fallbacks)
        # -----------------------------------------------------------------
        fundamentals: pd.DataFrame | None = market_data.get("fundamentals")
        prices: pd.DataFrame | None = market_data.get("prices")
        current_positions: dict[str, float] = market_data.get("current_positions", {})
        portfolio_value: float = market_data.get("portfolio_value", 0.0)

        if fundamentals is None or fundamentals.empty:
            self._log.warning("generate_signals.no_fundamentals")
            return []
        if prices is None or prices.empty:
            self._log.warning("generate_signals.no_prices")
            return []

        # Filter to our universe.
        available = [t for t in universe if t in fundamentals.index]
        if len(available) < 5:
            self._log.warning(
                "generate_signals.universe_too_small",
                available=len(available),
            )
            return []

        factor_df = self._build_factor_matrix(available, fundamentals, prices)
        if factor_df.empty:
            self._log.warning("generate_signals.empty_factor_matrix")
            return []

        # -----------------------------------------------------------------
        # Composite scoring
        # -----------------------------------------------------------------
        composite = self._compute_composite(factor_df)

        # -----------------------------------------------------------------
        # Percentile-based signal assignment
        # -----------------------------------------------------------------
        buy_pct: float = self._parameters["buy_percentile"]
        sell_pct: float = self._parameters["sell_percentile"]

        ranked = composite.rank(pct=True)
        buy_threshold = 1.0 - buy_pct
        sell_threshold = sell_pct

        signals: list[AgentSignal] = []
        max_weight = self._parameters["max_weight_per_stock"]
        max_positions = self._parameters["max_positions"]
        holding_min = self._parameters["holding_period_min"]
        holding_max = self._parameters["holding_period_max"]

        # Count how many BUY signals we plan to emit to cap at max_positions.
        buy_candidates = ranked[ranked >= buy_threshold].sort_values(ascending=False)
        buy_candidates = buy_candidates.head(max_positions)

        for ticker in composite.index:
            pct_rank = ranked.loc[ticker]
            score = composite.loc[ticker]
            conviction = self._compute_conviction(
                score, min_score=composite.min(), max_score=composite.max(),
            )

            # Determine action.
            if ticker in buy_candidates.index:
                action = Action.BUY
            elif pct_rank <= sell_threshold and ticker in current_positions:
                action = Action.SELL
            else:
                action = Action.HOLD

            # Skip HOLD signals to reduce noise -- only emit actionable signals.
            if action == Action.HOLD:
                continue

            # Target weight: equal-weight across buy candidates, capped.
            n_buys = max(len(buy_candidates), 1)
            target_weight = min(1.0 / n_buys, max_weight) if action == Action.BUY else 0.0

            # Price-based stop / target from latest price.
            latest_price = self._latest_price(ticker, prices)
            if latest_price is None or latest_price <= 0:
                self._log.debug("generate_signals.no_price", ticker=ticker)
                continue

            # Holding period: scale by conviction -- higher conviction -> longer hold.
            holding_days = int(
                holding_min + (holding_max - holding_min) * (conviction / 100.0)
            )
            holding_days = max(holding_min, min(holding_max, holding_days))

            # Stop loss / take profit: tighter stops for lower conviction.
            stop_loss_pct = 0.05 + 0.03 * (1.0 - conviction / 100.0)
            take_profit_pct = 0.08 + 0.07 * (conviction / 100.0)

            stop_loss = round(latest_price * (1.0 - stop_loss_pct), 2)
            take_profit = round(latest_price * (1.0 + take_profit_pct), 2)

            # Build per-factor scores dict for transparency.
            factor_scores = self._factor_scores_for_ticker(ticker, factor_df, composite)

            signal = AgentSignal(
                ticker=ticker,
                action=action,
                conviction=conviction,
                target_weight=round(target_weight, 4),
                stop_loss=max(stop_loss, 0.01),
                take_profit=max(take_profit, 0.01),
                factor_scores=factor_scores,
                reasoning=self._build_reasoning(ticker, action, factor_scores, conviction),
                holding_period_days=holding_days,
                agent_name=self._name,
                timestamp=now,
            )
            signals.append(signal)

        self._log.info(
            "generate_signals.complete",
            total_ranked=len(composite),
            buy_signals=sum(1 for s in signals if s.action == Action.BUY),
            sell_signals=sum(1 for s in signals if s.action == Action.SELL),
        )
        return signals

    # ------------------------------------------------------------------
    # Factor matrix construction
    # ------------------------------------------------------------------

    def _build_factor_matrix(
        self,
        tickers: list[str],
        fundamentals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build a DataFrame of raw factor values for *tickers*.

        Each column is a factor name, each row is a ticker.  Missing values
        are filled with the cross-sectional median so the z-score is neutral.
        """
        records: dict[str, dict[str, float]] = {}

        for ticker in tickers:
            row: dict[str, float] = {}
            fund = fundamentals.loc[ticker] if ticker in fundamentals.index else None

            # --- Growth factors ---
            row["revenue_growth"] = self._safe_float(fund, "revenue_growth")
            row["eps_growth"] = self._safe_float(fund, "eps_growth")

            # --- Quality ---
            row["roe"] = self._safe_float(fund, "roe")

            # --- Value (light) ---
            row["fcf_yield"] = self._safe_float(fund, "fcf_yield")

            # --- Momentum: 6-month price return ---
            row["momentum_6m"] = self._compute_momentum(ticker, prices, lookback_days=126)

            # --- Defensive: inverse of 60-day realised vol ---
            row["low_volatility"] = self._compute_low_vol(ticker, prices, window=60)

            # --- India-specific ---
            row["delivery_pct"] = self._safe_float(fund, "delivery_pct")
            row["fii_dii_net_flow"] = self._safe_float(fund, "fii_dii_net_flow")

            # --- Improving profitability ---
            gm = self._safe_float(fund, "gross_margin")
            prev_gm = self._safe_float(fund, "prev_gross_margin")
            row["gross_margin_expansion"] = (
                gm - prev_gm if not (np.isnan(gm) or np.isnan(prev_gm)) else np.nan
            )

            records[ticker] = row

        df = pd.DataFrame.from_dict(records, orient="index")

        # Fill NaN with cross-sectional median (neutral z-score).
        for col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median if not np.isnan(median) else 0.0)

        return df

    def _compute_composite(self, factor_df: pd.DataFrame) -> pd.Series:
        """Compute weighted composite z-score from the raw factor matrix.

        Returns a :class:`pd.Series` indexed by ticker.
        """
        weights: dict[str, float] = self._parameters["factor_weights"]
        composite = pd.Series(0.0, index=factor_df.index, dtype=np.float64)

        for factor_name, weight in weights.items():
            if factor_name not in factor_df.columns:
                self._log.debug(
                    "composite.missing_factor",
                    factor=factor_name,
                    msg="Skipped -- not in factor matrix",
                )
                continue
            z = self._zscore(factor_df[factor_name])
            composite += weight * z

        return composite

    # ------------------------------------------------------------------
    # Price / momentum helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _latest_price(ticker: str, prices: pd.DataFrame) -> float | None:
        """Return the latest closing price for *ticker*, or ``None``."""
        if "ticker" in prices.columns:
            subset = prices[prices["ticker"] == ticker]
            if subset.empty:
                return None
            return float(subset["close"].iloc[-1])
        elif ticker in prices.columns:
            series = prices[ticker].dropna()
            return float(series.iloc[-1]) if not series.empty else None
        return None

    @staticmethod
    def _compute_momentum(
        ticker: str,
        prices: pd.DataFrame,
        lookback_days: int = 126,
    ) -> float:
        """6-month price return for *ticker*.  Returns ``NaN`` on failure."""
        try:
            if "ticker" in prices.columns:
                subset = prices[prices["ticker"] == ticker].sort_index()
                if len(subset) < lookback_days:
                    return np.nan
                p_now = float(subset["close"].iloc[-1])
                p_prev = float(subset["close"].iloc[-lookback_days])
            elif ticker in prices.columns:
                series = prices[ticker].dropna()
                if len(series) < lookback_days:
                    return np.nan
                p_now = float(series.iloc[-1])
                p_prev = float(series.iloc[-lookback_days])
            else:
                return np.nan
            if p_prev == 0:
                return np.nan
            return (p_now - p_prev) / p_prev
        except (IndexError, KeyError, TypeError):
            return np.nan

    @staticmethod
    def _compute_low_vol(
        ticker: str,
        prices: pd.DataFrame,
        window: int = 60,
    ) -> float:
        """Inverse of 60-day realised volatility (higher = less volatile).

        Returns ``NaN`` if insufficient data.
        """
        try:
            if "ticker" in prices.columns:
                subset = prices[prices["ticker"] == ticker].sort_index()
                returns = subset["close"].pct_change().dropna()
            elif ticker in prices.columns:
                returns = prices[ticker].pct_change().dropna()
            else:
                return np.nan
            if len(returns) < window:
                return np.nan
            vol = returns.iloc[-window:].std() * np.sqrt(252)
            if vol == 0 or np.isnan(vol):
                return np.nan
            return 1.0 / vol
        except (IndexError, KeyError, TypeError):
            return np.nan

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(
        row: Any,
        column: str,
        default: float = np.nan,
    ) -> float:
        """Extract a float from a pandas row or dict, returning *default* on failure."""
        if row is None:
            return default
        try:
            val = row[column] if isinstance(row, dict) else getattr(row, column, default)
            return float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else default
        except (KeyError, AttributeError, TypeError, ValueError):
            return default

    def _factor_scores_for_ticker(
        self,
        ticker: str,
        factor_df: pd.DataFrame,
        composite: pd.Series,
    ) -> dict[str, float]:
        """Build a ``factor_scores`` dict for a single *ticker*."""
        scores: dict[str, float] = {}
        for col in factor_df.columns:
            val = factor_df.at[ticker, col]
            scores[col] = round(float(val), 4) if not np.isnan(val) else 0.0
        scores["composite_zscore"] = round(float(composite.loc[ticker]), 4)
        scores["rank_percentile"] = round(
            float(composite.rank(pct=True).loc[ticker]), 4,
        )
        return scores

    @staticmethod
    def _build_reasoning(
        ticker: str,
        action: Action,
        factor_scores: dict[str, float],
        conviction: int,
    ) -> str:
        """Construct a human-readable reasoning string for the signal."""
        rank_pct = factor_scores.get("rank_percentile", 0.0)
        composite = factor_scores.get("composite_zscore", 0.0)

        # Identify top contributing factors.
        contrib = {
            k: v
            for k, v in factor_scores.items()
            if k not in ("composite_zscore", "rank_percentile")
        }
        top_factors = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        top_str = ", ".join(f"{k}={v:+.2f}" for k, v in top_factors)

        return (
            f"{action.value} {ticker} | "
            f"rank={rank_pct:.0%} | composite_z={composite:+.2f} | "
            f"conviction={conviction} | top_factors: [{top_str}]"
        )
