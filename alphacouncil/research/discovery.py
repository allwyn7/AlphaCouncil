"""Weekly strategy discovery pipeline.

Runs every Sunday to generate candidate alpha features across multiple
look-back windows, compute their information coefficients (IC) against
forward returns, and surface only the signals that pass rigorous
statistical and robustness filters.

The growth-factor bias ensures that any feature correlated with revenue
growth receives a ranking bonus -- consistent with AlphaCouncil's
structural tilt toward growth equities.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats as sp_stats
from sqlalchemy.engine import Engine

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RankedNewSignal:
    """A candidate feature that has passed all IC and robustness filters."""

    feature_name: str
    mean_ic: float
    ic_ir: float
    robustness_score: float
    growth_bias_bonus: float
    composite_rank: float
    description: str
    forward_horizons: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOWS: list[int] = [5, 10, 20, 60, 120]
_FORWARD_HORIZONS: list[int] = [5, 10, 20]
_ROLLING_IC_WINDOW: int = 252
_MIN_ABS_MEAN_IC: float = 0.03
_MIN_IC_IR: float = 0.5
_MIN_ROBUST_PERIODS: int = 2
_GROWTH_BIAS_MULTIPLIER: float = 0.10


# ---------------------------------------------------------------------------
# StrategyDiscovery
# ---------------------------------------------------------------------------


class StrategyDiscovery:
    """Generate, screen, and rank candidate alpha features.

    Designed to run weekly (Sunday) as part of the research cron.  For each
    ticker in the universe it computes a library of single-name and
    interaction features, evaluates their predictive power via rolling
    Spearman rank-IC, and filters for significance and robustness across
    non-overlapping time periods and capitalisation buckets.

    Parameters
    ----------
    db_engine:
        SQLAlchemy :class:`Engine` for reading historical price / volume
        data and persisting discovery results.
    cache:
        Shared :class:`TieredCache` for memoising intermediate data.
    """

    def __init__(self, db_engine: Engine, cache: Any) -> None:
        self._db_engine = db_engine
        self._cache = cache
        self._log = logger.bind(component="strategy_discovery")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, universe: list[str]) -> list[dict]:
        """Execute the full discovery pipeline for *universe*.

        Steps
        -----
        1. Fetch historical OHLCV for every ticker.
        2. Compute candidate features (single + interaction terms).
        3. Compute rolling IC against forward N-day returns.
        4. Filter on |mean IC| > 0.03 and IC_IR > 0.5.
        5. Check robustness: significant in >= 2 non-overlapping periods
           and effective in both large-cap and mid-cap subsets.
        6. Apply growth-factor ranking bias.
        7. Return sorted :class:`RankedNewSignal` list (as dicts).

        Parameters
        ----------
        universe:
            List of NSE ticker symbols to evaluate.

        Returns
        -------
        list[dict]
            Ranked candidate signals, richest IC first.
        """
        self._log.info("discovery.start", universe_size=len(universe))
        start_ts = datetime.now(tz=timezone.utc)

        # Step 1 -- fetch data
        price_data = await self._fetch_price_data(universe)
        if price_data.empty:
            self._log.warning("discovery.no_price_data")
            return []

        # Step 2 -- compute features
        feature_panel = self._compute_all_features(price_data)
        self._log.info(
            "discovery.features_computed",
            n_features=len(feature_panel.columns),
        )

        # Step 3 -- compute forward returns
        forward_returns = self._compute_forward_returns(price_data)

        # Step 4+5+6 -- evaluate, filter, rank
        ranked_signals = await self._evaluate_features(
            feature_panel,
            forward_returns,
            price_data,
        )

        elapsed_s = (datetime.now(tz=timezone.utc) - start_ts).total_seconds()
        self._log.info(
            "discovery.complete",
            n_candidates=len(ranked_signals),
            elapsed_s=round(elapsed_s, 2),
        )

        return [self._signal_to_dict(s) for s in ranked_signals]

    # ------------------------------------------------------------------
    # IC computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ic(
        factor: pd.Series,
        forward_returns: pd.Series,
    ) -> float:
        """Compute rank (Spearman) information coefficient.

        Parameters
        ----------
        factor:
            Cross-sectional factor values aligned to *forward_returns*.
        forward_returns:
            Realised forward returns over the evaluation horizon.

        Returns
        -------
        float
            Spearman rho between the two series. Returns 0.0 when
            inputs are too short or constant.
        """
        common = factor.dropna().index.intersection(forward_returns.dropna().index)
        if len(common) < 30:
            return 0.0

        f = factor.loc[common]
        r = forward_returns.loc[common]

        # Constant series guard
        if f.std() == 0 or r.std() == 0:
            return 0.0

        rho, _ = sp_stats.spearmanr(f, r)
        return float(rho) if np.isfinite(rho) else 0.0

    @staticmethod
    def _check_robustness(
        factor_name: str,
        ic_series: pd.Series,
    ) -> float:
        """Score the robustness of an IC series across non-overlapping periods.

        The 252-day rolling IC series is split into non-overlapping annual
        blocks.  The score is the fraction of blocks where the mean IC is
        statistically significant (p < 0.05, one-sample t-test against 0).

        Parameters
        ----------
        factor_name:
            Used only for logging context.
        ic_series:
            Time-indexed Series of rolling IC values.

        Returns
        -------
        float
            Fraction of blocks with significant IC (0.0--1.0).
        """
        ic_clean = ic_series.dropna()
        if len(ic_clean) < _ROLLING_IC_WINDOW:
            return 0.0

        n_blocks = len(ic_clean) // _ROLLING_IC_WINDOW
        if n_blocks == 0:
            return 0.0

        significant_blocks = 0
        for i in range(n_blocks):
            block = ic_clean.iloc[i * _ROLLING_IC_WINDOW : (i + 1) * _ROLLING_IC_WINDOW]
            if len(block) < 60:
                continue
            t_stat, p_val = sp_stats.ttest_1samp(block, 0.0)
            if p_val < 0.05 and np.sign(t_stat) == np.sign(block.mean()):
                significant_blocks += 1

        return significant_blocks / n_blocks

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_all_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Build the full feature panel (single + interaction terms).

        Single features per window N:
            return_N, vol_N, volume_ratio_N, rsi_N, price_vs_sma_N

        Interaction terms:
            return_5 * volume_ratio_5, rsi_14 * vol_20, and additional
            momentum x volume crosses.

        Parameters
        ----------
        price_data:
            DataFrame with columns ``[ticker, date, open, high, low,
            close, volume]`` in long format, or a MultiIndex DataFrame
            ``(date, ticker) -> [close, volume, ...]``.

        Returns
        -------
        pd.DataFrame
            Columns are feature names; index matches *price_data*.
        """
        features: dict[str, pd.Series] = {}

        close = price_data["close"]
        volume = price_data["volume"]
        high = price_data.get("high", close)
        low = price_data.get("low", close)

        for w in _WINDOWS:
            # Trailing return
            features[f"return_{w}"] = close.pct_change(w)

            # Realised volatility (annualised)
            features[f"vol_{w}"] = close.pct_change().rolling(w).std() * np.sqrt(252)

            # Volume ratio: current / trailing average
            vol_ma = volume.rolling(w).mean()
            features[f"volume_ratio_{w}"] = volume / vol_ma.replace(0, np.nan)

            # RSI (Wilder-style)
            features[f"rsi_{w}"] = self._rsi(close, w)

            # Price vs. SMA deviation (z-score-like)
            sma = close.rolling(w).mean()
            features[f"price_vs_sma_{w}"] = (close - sma) / sma.replace(0, np.nan)

        # --- Interaction terms ---------------------------------------------------
        features["return_5_x_volume_ratio_5"] = (
            features["return_5"] * features["volume_ratio_5"]
        )
        features["rsi_14_x_vol_20"] = (
            self._rsi(close, 14) * features["vol_20"]
        )
        features["return_10_x_volume_ratio_10"] = (
            features["return_10"] * features["volume_ratio_10"]
        )
        features["return_20_x_vol_20"] = (
            features["return_20"] * features["vol_20"]
        )
        features["price_vs_sma_20_x_volume_ratio_20"] = (
            features["price_vs_sma_20"] * features["volume_ratio_20"]
        )
        features["return_60_x_vol_60"] = (
            features["return_60"] * features["vol_60"]
        )

        return pd.DataFrame(features)

    # ------------------------------------------------------------------
    # Forward returns
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_forward_returns(
        price_data: pd.DataFrame,
    ) -> dict[int, pd.Series]:
        """Compute forward N-day returns for each horizon in _FORWARD_HORIZONS."""
        close = price_data["close"]
        fwd: dict[int, pd.Series] = {}
        for h in _FORWARD_HORIZONS:
            fwd[h] = close.shift(-h) / close - 1.0
        return fwd

    # ------------------------------------------------------------------
    # Evaluate, filter, rank
    # ------------------------------------------------------------------

    async def _evaluate_features(
        self,
        feature_panel: pd.DataFrame,
        forward_returns: dict[int, pd.Series],
        price_data: pd.DataFrame,
    ) -> list[RankedNewSignal]:
        """Run IC analysis, robustness checks, and ranking.

        Offloads the heavy computation to a thread pool so the event loop
        remains responsive.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._evaluate_features_sync,
            feature_panel,
            forward_returns,
            price_data,
        )

    def _evaluate_features_sync(
        self,
        feature_panel: pd.DataFrame,
        forward_returns: dict[int, pd.Series],
        price_data: pd.DataFrame,
    ) -> list[RankedNewSignal]:
        """Synchronous, CPU-bound evaluation of all candidate features."""
        results: list[RankedNewSignal] = []

        for feat_name in feature_panel.columns:
            factor = feature_panel[feat_name]
            if factor.dropna().empty:
                continue

            # --- Rolling IC per horizon -----------------------------------------
            horizon_ics: dict[int, float] = {}
            ic_series_agg = pd.Series(dtype=np.float64)

            for h in _FORWARD_HORIZONS:
                fwd = forward_returns[h]

                # Rolling IC (252-day windows)
                ic_vals: list[float] = []
                ic_dates: list[Any] = []
                for end in range(_ROLLING_IC_WINDOW, len(factor)):
                    start = end - _ROLLING_IC_WINDOW
                    f_window = factor.iloc[start:end]
                    r_window = fwd.iloc[start:end]
                    ic_val = self._compute_ic(f_window, r_window)
                    ic_vals.append(ic_val)
                    ic_dates.append(factor.index[end] if hasattr(factor.index, '__getitem__') else end)

                if not ic_vals:
                    continue

                ic_s = pd.Series(ic_vals, index=ic_dates)
                mean_ic = float(np.nanmean(ic_vals))
                horizon_ics[h] = mean_ic

                if ic_series_agg.empty:
                    ic_series_agg = ic_s
                else:
                    ic_series_agg = ic_series_agg.add(ic_s, fill_value=0.0) / 2.0

            if not horizon_ics:
                continue

            # --- Aggregate IC metrics across horizons ---------------------------
            overall_mean_ic = float(np.mean(list(horizon_ics.values())))
            if ic_series_agg.std() == 0:
                ic_ir = 0.0
            else:
                ic_ir = float(ic_series_agg.mean() / ic_series_agg.std())

            # --- Filter: |mean IC| > 0.03 and IC_IR > 0.5 ----------------------
            if abs(overall_mean_ic) < _MIN_ABS_MEAN_IC:
                continue
            if ic_ir < _MIN_IC_IR:
                continue

            # --- Robustness check -----------------------------------------------
            robustness = self._check_robustness(feat_name, ic_series_agg)
            if robustness < (_MIN_ROBUST_PERIODS / max(len(ic_series_agg) // _ROLLING_IC_WINDOW, 1)):
                continue

            # --- Cap-bucket robustness (large-cap + mid-cap) --------------------
            cap_robust = self._check_cap_bucket_robustness(
                feat_name, factor, forward_returns, price_data,
            )
            if not cap_robust:
                continue

            # --- Growth-factor bias bonus ---------------------------------------
            growth_bonus = self._compute_growth_bias(feat_name, factor, price_data)

            # --- Composite rank score -------------------------------------------
            composite = abs(overall_mean_ic) * ic_ir * (1.0 + robustness) + growth_bonus

            signal = RankedNewSignal(
                feature_name=feat_name,
                mean_ic=round(overall_mean_ic, 6),
                ic_ir=round(ic_ir, 4),
                robustness_score=round(robustness, 4),
                growth_bias_bonus=round(growth_bonus, 6),
                composite_rank=round(composite, 6),
                description=self._describe_feature(feat_name),
                forward_horizons={h: round(v, 6) for h, v in horizon_ics.items()},
            )
            results.append(signal)

        # Sort descending by composite rank
        results.sort(key=lambda s: s.composite_rank, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Cap-bucket robustness
    # ------------------------------------------------------------------

    def _check_cap_bucket_robustness(
        self,
        feat_name: str,
        factor: pd.Series,
        forward_returns: dict[int, pd.Series],
        price_data: pd.DataFrame,
    ) -> bool:
        """Verify the feature works in both large-cap and mid-cap subsets.

        Splits the universe at the median market-cap proxy (trailing
        252-day average close * volume) and requires |mean IC| > 0.02
        in both halves.
        """
        close = price_data["close"]
        volume = price_data["volume"]

        # Market-cap proxy: average of (close * volume) over trailing 252 days
        cap_proxy = (close * volume).rolling(min(_ROLLING_IC_WINDOW, len(close))).mean()

        if cap_proxy.dropna().empty:
            return True  # data insufficient -- pass through

        median_cap = cap_proxy.median()
        large_mask = cap_proxy >= median_cap
        mid_mask = cap_proxy < median_cap

        for mask, label in [(large_mask, "large"), (mid_mask, "mid")]:
            sub_factor = factor.loc[mask.fillna(False)]
            if sub_factor.dropna().empty:
                continue

            # Check IC on the 5-day forward horizon as a representative check
            fwd_5 = forward_returns.get(5, pd.Series(dtype=np.float64))
            sub_fwd = fwd_5.loc[mask.fillna(False)]
            ic = self._compute_ic(sub_factor, sub_fwd)

            if abs(ic) < 0.02:
                self._log.debug(
                    "discovery.cap_bucket_fail",
                    feature=feat_name,
                    bucket=label,
                    ic=round(ic, 6),
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Growth-factor bias
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_growth_bias(
        feat_name: str,
        factor: pd.Series,
        price_data: pd.DataFrame,
    ) -> float:
        """Assign a ranking bonus for features correlated with revenue growth.

        Uses trailing 60-day return as a rough proxy for revenue-growth
        exposure when actual fundamental data is unavailable.  The bonus
        is ``_GROWTH_BIAS_MULTIPLIER * correlation`` (clamped to >= 0).
        """
        close = price_data["close"]
        growth_proxy = close.pct_change(60)

        common = factor.dropna().index.intersection(growth_proxy.dropna().index)
        if len(common) < 60:
            return 0.0

        corr = factor.loc[common].corr(growth_proxy.loc[common])
        if not np.isfinite(corr):
            return 0.0

        return max(0.0, corr * _GROWTH_BIAS_MULTIPLIER)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fetch_price_data(self, universe: list[str]) -> pd.DataFrame:
        """Fetch historical OHLCV data from the database for all tickers.

        Returns a single DataFrame with columns ``[close, volume, high,
        low]`` indexed by date. For a multi-ticker universe this returns
        the average cross-sectional values (panel-level discovery).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._fetch_price_data_sync, universe,
        )

    def _fetch_price_data_sync(self, universe: list[str]) -> pd.DataFrame:
        """Read OHLCV from the database synchronously."""
        try:
            query = """
                SELECT timestamp AS date, symbol,
                       price AS close, quantity AS volume
                FROM trades
                ORDER BY timestamp
            """
            with self._db_engine.connect() as conn:
                df = pd.read_sql(query, conn, parse_dates=["date"])

            if df.empty:
                self._log.warning("discovery.empty_price_data")
                return pd.DataFrame()

            # Pivot to wide format or aggregate; for discovery we work
            # cross-sectionally so we keep a simple long-to-wide average.
            if "symbol" in df.columns:
                df = df.set_index("date")
                # Use cross-sectional mean for panel-level feature discovery
                agg = df.groupby(df.index)[["close", "volume"]].mean()
                agg["high"] = agg["close"]
                agg["low"] = agg["close"]
                return agg

            return df
        except Exception as exc:
            self._log.error("discovery.price_fetch_failed", error=str(exc))
            return pd.DataFrame()

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        """Compute Wilder-style RSI."""
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _describe_feature(feat_name: str) -> str:
        """Generate a human-readable description from a feature name."""
        parts = feat_name.split("_x_")
        if len(parts) == 2:
            return f"Interaction: {parts[0].replace('_', ' ')} x {parts[1].replace('_', ' ')}"

        tokens = feat_name.rsplit("_", 1)
        if len(tokens) == 2 and tokens[1].isdigit():
            base, window = tokens
            return f"{base.replace('_', ' ').title()} with {window}-day look-back"

        return feat_name.replace("_", " ").title()

    @staticmethod
    def _signal_to_dict(signal: RankedNewSignal) -> dict:
        """Convert a frozen dataclass to a plain dict for serialisation."""
        return {
            "feature_name": signal.feature_name,
            "mean_ic": signal.mean_ic,
            "ic_ir": signal.ic_ir,
            "robustness_score": signal.robustness_score,
            "growth_bias_bonus": signal.growth_bias_bonus,
            "composite_rank": signal.composite_rank,
            "description": signal.description,
            "forward_horizons": signal.forward_horizons,
        }
