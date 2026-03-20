"""Growth-Momentum agent -- the PRIMARY alpha source in AlphaCouncil.

Strategy
--------
Hunts high-growth stocks with strong price momentum.  The factor model
combines fundamental growth metrics (revenue growth, EPS growth, revenue
acceleration) with technical momentum signals (6-month return, relative
strength vs Nifty 50, sentiment momentum, volume trend).  All factors are
z-scored cross-sectionally, then weighted to produce a composite score.

The "secret sauce" is **revenue acceleration** -- if the rate of revenue
growth is *itself* accelerating, conviction is amplified, capturing stocks
at the inflection point before the broader market reprices them.

Portfolio rules
~~~~~~~~~~~~~~~
- Max 12 concurrent positions, max 10 % per position.
- Stop-loss: 2.5x ATR from entry price.
- Take-profit: none (let winners run).
- Holding period: 30--90 days.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401 (used dynamically)
import structlog

from alphacouncil.agents.base import BaseAgent, MessageBus
from alphacouncil.core.models import Action, AgentSignal, AgentStatus

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Default factor weights
# ---------------------------------------------------------------------------

_DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "revenue_growth_yoy": 0.25,
    "revenue_acceleration": 0.15,
    "price_momentum_6m": 0.15,      # was 0.20, reduced by 0.05
    "eps_growth_yoy": 0.15,
    "relative_strength_nifty": 0.10,
    "sentiment_momentum": 0.15,     # was 0.10, increased by 0.05
    "volume_trend": 0.05,
}

# ---------------------------------------------------------------------------
# Default tunable parameters
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict[str, Any] = {
    "buy_threshold": 0.70,
    "sell_threshold": -0.50,
    "adx_threshold": 20.0,
    "volume_confirm": 1.0,
    "atr_multiplier": 2.5,
    "factor_weights": dict(_DEFAULT_FACTOR_WEIGHTS),
    "max_positions": 12,
    "max_weight_per_position": 0.10,
    "holding_period_min": 30,
    "holding_period_max": 90,
}


class GrowthMomentumAgent(BaseAgent):
    """Primary growth-momentum agent.

    See module docstring for full strategy description and factor catalogue.
    """

    def __init__(
        self,
        config: Any,
        cache: Any,
        bus: MessageBus,
        db_engine: Any,
        *,
        name: str = "growth_momentum",
    ) -> None:
        super().__init__(
            name=name,
            config=config,
            cache=cache,
            bus=bus,
            db_engine=db_engine,
        )
        # Initialise tunable parameters with defaults.
        self._parameters: dict[str, Any] = dict(_DEFAULT_PARAMS)
        self._log.info(
            "agent.params_loaded",
            params={k: v for k, v in self._parameters.items() if k != "factor_weights"},
        )

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def get_parameters(self) -> dict[str, Any]:
        """Return a copy of the current tunable parameters."""
        return dict(self._parameters)

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Hot-reload tunable parameters.

        Only known keys are accepted; unknown keys are logged and ignored.
        Factor weights are validated to ensure they sum to ~1.0.
        """
        for key, value in params.items():
            if key not in self._parameters:
                self._log.warning("agent.unknown_param", key=key)
                continue

            if key == "factor_weights":
                weight_sum = sum(value.values())
                if not np.isclose(weight_sum, 1.0, atol=0.01):
                    self._log.error(
                        "agent.invalid_weights",
                        weight_sum=weight_sum,
                    )
                    continue

            old = self._parameters[key]
            self._parameters[key] = value
            self._log.info("agent.param_updated", key=key, old=old, new=value)

    # ------------------------------------------------------------------
    # Factor computation helpers
    # ------------------------------------------------------------------

    def _compute_revenue_growth(
        self,
        fundamentals: dict[str, Any],
        universe: list[str],
    ) -> pd.Series:
        """Extract YoY revenue growth for each ticker (from yfinance ``revenueGrowth``)."""
        data: dict[str, float] = {}
        for ticker in universe:
            info = fundamentals.get(ticker, {})
            data[ticker] = float(info.get("revenueGrowth", 0.0) or 0.0)
        return pd.Series(data, dtype=np.float64)

    def _compute_revenue_acceleration(
        self,
        fundamentals: dict[str, Any],
        universe: list[str],
    ) -> pd.Series:
        """QoQ change in YoY revenue growth rate.

        If the data source provides quarterly revenue growth history we use
        the delta; otherwise we fall back to zero.
        """
        data: dict[str, float] = {}
        for ticker in universe:
            info = fundamentals.get(ticker, {})
            quarterly_growth: list[float] | None = info.get("quarterly_revenue_growth")
            if quarterly_growth and len(quarterly_growth) >= 2:
                # Most recent minus prior quarter.
                data[ticker] = float(quarterly_growth[-1] - quarterly_growth[-2])
            else:
                data[ticker] = 0.0
        return pd.Series(data, dtype=np.float64)

    def _compute_price_momentum_6m(
        self,
        prices: dict[str, pd.DataFrame],
        universe: list[str],
    ) -> pd.Series:
        """6-month price return excluding the most recent month.

        The skip-month convention avoids the well-documented short-term
        reversal effect.
        """
        data: dict[str, float] = {}
        for ticker in universe:
            df = prices.get(ticker)
            if df is None or df.empty or len(df) < 126:
                data[ticker] = 0.0
                continue
            close = df["Close"]
            # 6 months ~ 126 trading days; skip last ~21 trading days.
            price_6m_ago = close.iloc[-126]
            price_1m_ago = close.iloc[-21]
            if price_6m_ago > 0:
                data[ticker] = float((price_1m_ago - price_6m_ago) / price_6m_ago)
            else:
                data[ticker] = 0.0
        return pd.Series(data, dtype=np.float64)

    def _compute_eps_growth(
        self,
        fundamentals: dict[str, Any],
        universe: list[str],
    ) -> pd.Series:
        """YoY EPS growth from fundamentals."""
        data: dict[str, float] = {}
        for ticker in universe:
            info = fundamentals.get(ticker, {})
            data[ticker] = float(info.get("earningsGrowth", 0.0) or 0.0)
        return pd.Series(data, dtype=np.float64)

    def _compute_relative_strength(
        self,
        prices: dict[str, pd.DataFrame],
        universe: list[str],
        nifty_prices: pd.DataFrame | None,
    ) -> pd.Series:
        """Stock return minus Nifty 50 return over the last 3 months (~63 trading days)."""
        data: dict[str, float] = {}
        nifty_ret = 0.0
        if nifty_prices is not None and len(nifty_prices) >= 63:
            nifty_close = nifty_prices["Close"]
            nifty_ret = float(
                (nifty_close.iloc[-1] - nifty_close.iloc[-63]) / nifty_close.iloc[-63]
            )

        for ticker in universe:
            df = prices.get(ticker)
            if df is None or df.empty or len(df) < 63:
                data[ticker] = 0.0
                continue
            close = df["Close"]
            stock_ret = float((close.iloc[-1] - close.iloc[-63]) / close.iloc[-63])
            data[ticker] = stock_ret - nifty_ret
        return pd.Series(data, dtype=np.float64)

    def _compute_sentiment_momentum(
        self,
        sentiment: dict[str, Any],
        universe: list[str],
    ) -> pd.Series:
        """7-day average FinBERT sentiment minus 30-day average."""
        data: dict[str, float] = {}
        for ticker in universe:
            sent = sentiment.get(ticker, {})
            avg_7d = float(sent.get("sentiment_7d", 0.0) or 0.0)
            avg_30d = float(sent.get("sentiment_30d", 0.0) or 0.0)
            data[ticker] = avg_7d - avg_30d
        return pd.Series(data, dtype=np.float64)

    def _compute_volume_trend(
        self,
        prices: dict[str, pd.DataFrame],
        universe: list[str],
    ) -> pd.Series:
        """5-day average volume / 20-day average volume."""
        data: dict[str, float] = {}
        for ticker in universe:
            df = prices.get(ticker)
            if df is None or df.empty or len(df) < 20:
                data[ticker] = 1.0
                continue
            vol = df["Volume"].astype(np.float64)
            avg_5 = vol.iloc[-5:].mean()
            avg_20 = vol.iloc[-20:].mean()
            data[ticker] = float(avg_5 / avg_20) if avg_20 > 0 else 1.0
        return pd.Series(data, dtype=np.float64)

    # ------------------------------------------------------------------
    # Technical indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_adx(df: pd.DataFrame, length: int = 14) -> float:
        """Compute the latest ADX value for a single ticker's OHLCV dataframe."""
        if df is None or df.empty or len(df) < length + 1:
            return 0.0
        adx_df = df.ta.adx(length=length)
        if adx_df is None or adx_df.empty:
            return 0.0
        col = f"ADX_{length}"
        if col not in adx_df.columns:
            return 0.0
        val = adx_df[col].iloc[-1]
        return float(val) if not np.isnan(val) else 0.0

    @staticmethod
    def _get_atr(df: pd.DataFrame, length: int = 14) -> float:
        """Compute the latest ATR value for a single ticker's OHLCV dataframe."""
        if df is None or df.empty or len(df) < length + 1:
            return 0.0
        atr_s = df.ta.atr(length=length)
        if atr_s is None or atr_s.empty:
            return 0.0
        val = atr_s.iloc[-1]
        return float(val) if not np.isnan(val) else 0.0

    @staticmethod
    def _ema_crossover_bearish(df: pd.DataFrame, short: int = 50, long: int = 200) -> bool:
        """Return True if EMA *short* has just crossed below EMA *long*."""
        if df is None or df.empty or len(df) < long + 2:
            return False
        ema_short = df.ta.ema(length=short)
        ema_long = df.ta.ema(length=long)
        if ema_short is None or ema_long is None:
            return False
        if len(ema_short) < 2 or len(ema_long) < 2:
            return False
        prev_above = ema_short.iloc[-2] >= ema_long.iloc[-2]
        now_below = ema_short.iloc[-1] < ema_long.iloc[-1]
        return bool(prev_above and now_below)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    async def generate_signals(
        self,
        universe: list[str],
        market_data: dict[str, Any],
    ) -> list[AgentSignal]:
        """Run the full growth-momentum factor model and emit signals.

        Expected ``market_data`` keys:
            - ``"prices"``: ``dict[str, pd.DataFrame]`` -- OHLCV per ticker.
            - ``"fundamentals"``: ``dict[str, dict]`` -- yfinance ``info`` dicts.
            - ``"sentiment"``: ``dict[str, dict]`` -- per-ticker sentiment
              with ``sentiment_7d`` and ``sentiment_30d``.
            - ``"nifty_prices"``: ``pd.DataFrame`` -- Nifty 50 OHLCV.
        """
        prices: dict[str, pd.DataFrame] = market_data.get("prices", {})
        fundamentals: dict[str, Any] = market_data.get("fundamentals", {})
        sentiment: dict[str, Any] = market_data.get("sentiment", {})
        nifty_prices: pd.DataFrame | None = market_data.get("nifty_prices")

        weights = self._parameters["factor_weights"]

        # -- 1. Compute raw factor values for every ticker ----------------
        rev_growth = self._compute_revenue_growth(fundamentals, universe)
        rev_accel = self._compute_revenue_acceleration(fundamentals, universe)
        mom_6m = self._compute_price_momentum_6m(prices, universe)
        eps_growth = self._compute_eps_growth(fundamentals, universe)
        rel_strength = self._compute_relative_strength(prices, universe, nifty_prices)
        sent_mom = self._compute_sentiment_momentum(sentiment, universe)
        vol_trend = self._compute_volume_trend(prices, universe)

        # -- 2. Z-score each factor cross-sectionally ---------------------
        z_rev_growth = self._zscore(rev_growth)
        z_rev_accel = self._zscore(rev_accel)
        z_mom_6m = self._zscore(mom_6m)
        z_eps_growth = self._zscore(eps_growth)
        z_rel_strength = self._zscore(rel_strength)
        z_sent_mom = self._zscore(sent_mom)
        z_vol_trend = self._zscore(vol_trend)

        # -- 3. Weighted composite score ----------------------------------
        composite = (
            z_rev_growth * weights["revenue_growth_yoy"]
            + z_rev_accel * weights["revenue_acceleration"]
            + z_mom_6m * weights["price_momentum_6m"]
            + z_eps_growth * weights["eps_growth_yoy"]
            + z_rel_strength * weights["relative_strength_nifty"]
            + z_sent_mom * weights["sentiment_momentum"]
            + z_vol_trend * weights["volume_trend"]
        )

        # -- 4. Per-ticker signal logic -----------------------------------
        buy_thresh = self._parameters["buy_threshold"]
        sell_thresh = self._parameters["sell_threshold"]
        adx_thresh = self._parameters["adx_threshold"]
        vol_confirm = self._parameters["volume_confirm"]
        atr_mult = self._parameters["atr_multiplier"]
        now = datetime.now(tz=timezone.utc)

        signals: list[AgentSignal] = []

        for ticker in universe:
            score = float(composite.get(ticker, 0.0))
            df = prices.get(ticker)
            if df is None or df.empty:
                continue

            current_price = float(df["Close"].iloc[-1])
            adx_val = self._get_adx(df)
            atr_val = self._get_atr(df)
            volume_ratio = float(vol_trend.get(ticker, 1.0))
            raw_rev_accel = float(rev_accel.get(ticker, 0.0))
            ema_bearish = self._ema_crossover_bearish(df)

            # Determine action.
            action: Action
            if (
                score > buy_thresh
                and adx_val > adx_thresh
                and volume_ratio > vol_confirm
            ):
                action = Action.BUY
            elif ema_bearish or score < sell_thresh:
                action = Action.SELL
            else:
                action = Action.HOLD

            # --- Sentiment hard override ---
            ticker_sent = sentiment.get(ticker, {})
            sent_7d = float(ticker_sent.get("sentiment_7d", 0.0))
            sent_30d = float(ticker_sent.get("sentiment_30d", 0.0))
            sent_momentum_raw = sent_7d - sent_30d
            if sent_7d < -0.5 and sent_momentum_raw < -0.3:
                action = Action.SELL
                logger.warning(
                    "sentiment_hard_override",
                    ticker=ticker,
                    sentiment_7d=sent_7d,
                    sentiment_momentum=sent_momentum_raw,
                )

            # Only emit BUY / SELL signals (HOLD is implicit).
            if action == Action.HOLD:
                continue

            # Conviction: base mapping then amplify by revenue acceleration.
            conviction = self._compute_conviction(abs(score), min_score=0.0, max_score=2.0)
            if action == Action.BUY and raw_rev_accel > 0:
                amplifier = 1.0 + min(raw_rev_accel, 0.5)
                conviction = int(min(conviction * amplifier, 100))

            # Stop-loss: 2.5x ATR below current price.
            stop_loss = max(current_price - atr_mult * atr_val, 0.01)

            # Take-profit: let winners run -- set a distant sentinel.
            take_profit = current_price * 5.0

            # Factor scores dict for auditing.
            factor_scores: dict[str, float] = {
                "revenue_growth_yoy": float(z_rev_growth.get(ticker, 0.0)),
                "revenue_acceleration": float(z_rev_accel.get(ticker, 0.0)),
                "price_momentum_6m": float(z_mom_6m.get(ticker, 0.0)),
                "eps_growth_yoy": float(z_eps_growth.get(ticker, 0.0)),
                "relative_strength_nifty": float(z_rel_strength.get(ticker, 0.0)),
                "sentiment_momentum": float(z_sent_mom.get(ticker, 0.0)),
                "volume_trend": float(z_vol_trend.get(ticker, 0.0)),
                "composite": score,
                "adx": adx_val,
                "atr": atr_val,
                "volume_ratio": volume_ratio,
                "revenue_acceleration_raw": raw_rev_accel,
            }

            reasoning = self._build_reasoning(ticker, action, score, factor_scores)

            signals.append(
                AgentSignal(
                    ticker=ticker,
                    action=action,
                    conviction=conviction,
                    target_weight=self._parameters["max_weight_per_position"],
                    stop_loss=round(stop_loss, 2),
                    take_profit=round(take_profit, 2),
                    factor_scores=factor_scores,
                    reasoning=reasoning,
                    holding_period_days=self._parameters["holding_period_min"],
                    agent_name=self._name,
                    timestamp=now,
                )
            )

        self._log.info(
            "signals.generated",
            total=len(signals),
            buys=sum(1 for s in signals if s.action == Action.BUY),
            sells=sum(1 for s in signals if s.action == Action.SELL),
        )
        return signals

    # ------------------------------------------------------------------
    # Reasoning builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reasoning(
        ticker: str,
        action: Action,
        composite: float,
        factors: dict[str, float],
    ) -> str:
        """Construct a human-readable rationale string."""
        parts: list[str] = [
            f"{action.value} {ticker}: composite={composite:+.3f}.",
        ]
        # Highlight the top contributing factor.
        factor_subset = {
            k: v
            for k, v in factors.items()
            if k not in {"composite", "adx", "atr", "volume_ratio", "revenue_acceleration_raw"}
        }
        if factor_subset:
            top_factor = max(factor_subset, key=lambda k: abs(factor_subset[k]))
            parts.append(f"Top factor: {top_factor} (z={factors[top_factor]:+.2f}).")

        if action == Action.BUY:
            parts.append(f"ADX={factors.get('adx', 0):.1f} confirms trend.")
            raw_accel = factors.get("revenue_acceleration_raw", 0.0)
            if raw_accel > 0:
                parts.append(f"Revenue accelerating (+{raw_accel:.2f}): conviction amplified.")
        elif action == Action.SELL:
            if composite < -0.5:
                parts.append("Composite below sell threshold.")
            else:
                parts.append("EMA50 crossed below EMA200: bearish momentum break.")

        return " ".join(parts)
