"""Sentiment-driven alpha agent for the AlphaCouncil trading system.

Trades on news narratives, social buzz, and institutional flow confirmation.
Carries an explicit **growth bias** -- growth-positive keywords amplify
conviction by a configurable multiplier (default 1.3x).

Data pipeline
-------------
``SentimentEngine`` (from ``analysis/sentiment.py``) feeds RSS + Reddit
headlines through FinBERT to produce a :class:`~alphacouncil.core.models.SentimentSignal`
per ticker.  This agent consumes those signals and layers its own multi-factor
scoring and trade-classification logic on top.

Factor model (six factors, weights sum to 1.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Sentiment Score** (20 %) -- FinBERT average over the last 24 h.
2. **Sentiment Momentum** (25 %) -- 3-day avg minus 14-day avg; improving
   sentiment is the single strongest signal.
3. **Article Volume Spike** (20 %) -- today's article count / 30-day average.
   A ratio > 2x is flagged as an event.
4. **Growth Narrative Keywords** (15 %) -- bonus for growth-positive keywords
   (``"revenue beat"``, ``"expansion"``, etc.); penalty for negative keywords
   (``"downgrade"``, ``"fraud"``, etc.).
5. **FII / DII Flow Direction** (10 %) -- institutional money confirming the
   narrative.
6. **Social Buzz** (10 %) -- Reddit mention velocity vs. historical average.

Signal classification
~~~~~~~~~~~~~~~~~~~~~
* **Event trade** -- sentiment spike (> 2 sigma) *and* volume spike -> high-
  conviction directional trade.  Stop-loss -3 %, take-profit +8 %.
* **Trend trade** -- sustained improving sentiment over 3+ days -> gradual
  position build.  Stop-loss -6 %, take-profit +15 %.
* **Contrarian fade** -- sentiment > 0.85 sustained for 5+ days -> potential
  mean-reversion SELL.
* Growth keyword bonus: multiply conviction by 1.3x when growth-positive
  keywords are present.

Portfolio profile: 5-8 positions, max 15 % each, high turnover, 1-10 day
holding period.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog

from alphacouncil.agents.base import BaseAgent
from alphacouncil.core.models import Action, AgentSignal, AgentStatus

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_NAME = "sentiment_alpha"

# Growth-positive keywords that amplify conviction.
GROWTH_POSITIVE_KEYWORDS: frozenset[str] = frozenset(
    {
        "revenue beat",
        "growth",
        "expansion",
        "market share",
        "new product",
        "order win",
        "capacity addition",
    }
)

# Negative keywords that penalise the keyword factor.
NEGATIVE_KEYWORDS: frozenset[str] = frozenset(
    {
        "downgrade",
        "miss",
        "fraud",
        "sebi probe",
        "debt concern",
    }
)

# Default factor weights (must sum to 1.0).
_DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "sentiment_score": 0.20,
    "sentiment_momentum": 0.25,
    "article_volume_spike": 0.20,
    "growth_narrative": 0.15,
    "fii_dii_flow": 0.10,
    "social_buzz": 0.10,
}

# ---------------------------------------------------------------------------
# Portfolio constraints
# ---------------------------------------------------------------------------

_MAX_POSITIONS = 8
_MIN_POSITIONS = 5
_MAX_WEIGHT_PER_POSITION = 0.15


# ---------------------------------------------------------------------------
# SentimentAlphaAgent
# ---------------------------------------------------------------------------


class SentimentAlphaAgent(BaseAgent):
    """Sentiment-driven alpha agent with an explicit growth bias.

    Parameters
    ----------
    name:
        Canonical agent name (defaults to ``"sentiment_alpha"``).
    config:
        System-wide settings.
    cache:
        Shared cache backend.
    bus:
        Message bus satisfying :class:`~alphacouncil.agents.base.MessageBus`.
    db_engine:
        SQLAlchemy engine for persistence.
    sentiment_engine:
        An instance of ``SentimentEngine`` from ``analysis/sentiment.py``.
        Used to fetch :class:`~alphacouncil.core.models.SentimentSignal`
        per ticker.
    """

    def __init__(
        self,
        name: str = AGENT_NAME,
        config: Any = None,
        cache: Any = None,
        bus: Any = None,
        db_engine: Any = None,
        *,
        sentiment_engine: Any = None,
    ) -> None:
        super().__init__(name=name, config=config, cache=cache, bus=bus, db_engine=db_engine)
        self._sentiment_engine = sentiment_engine
        self._log = logger.bind(agent=name)

        # ----- Tunable parameters -----------------------------------------
        self._event_sigma: float = 2.0
        self._volume_spike_threshold: float = 2.0
        self._contrarian_threshold: float = 0.85
        self._contrarian_days: int = 5
        self._growth_keyword_bonus: float = 1.3
        self._factor_weights: dict[str, float] = dict(_DEFAULT_FACTOR_WEIGHTS)
        self._event_stop_loss: float = 0.03
        self._trend_stop_loss: float = 0.06
        self._event_take_profit: float = 0.08
        self._trend_take_profit: float = 0.15

        self._log.info(
            "sentiment_agent.configured",
            event_sigma=self._event_sigma,
            volume_spike_threshold=self._volume_spike_threshold,
            growth_keyword_bonus=self._growth_keyword_bonus,
        )

    # ------------------------------------------------------------------
    # BaseAgent abstract interface
    # ------------------------------------------------------------------

    def get_parameters(self) -> dict[str, Any]:
        """Return the agent's current tunable parameters."""
        return {
            "event_sigma": self._event_sigma,
            "volume_spike_threshold": self._volume_spike_threshold,
            "contrarian_threshold": self._contrarian_threshold,
            "contrarian_days": self._contrarian_days,
            "growth_keyword_bonus": self._growth_keyword_bonus,
            "factor_weights": dict(self._factor_weights),
            "event_stop_loss": self._event_stop_loss,
            "trend_stop_loss": self._trend_stop_loss,
            "event_take_profit": self._event_take_profit,
            "trend_take_profit": self._trend_take_profit,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Hot-reload tunable parameters.

        Only recognised keys are applied; unknown keys are logged and skipped.
        """
        allowed = {
            "event_sigma",
            "volume_spike_threshold",
            "contrarian_threshold",
            "contrarian_days",
            "growth_keyword_bonus",
            "factor_weights",
            "event_stop_loss",
            "trend_stop_loss",
            "event_take_profit",
            "trend_take_profit",
        }
        for key, value in params.items():
            if key not in allowed:
                self._log.warning("set_parameters.unknown_key", key=key)
                continue
            old_value = getattr(self, f"_{key}")
            setattr(self, f"_{key}", value)
            self._log.info("set_parameters.updated", key=key, old=old_value, new=value)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    async def generate_signals(
        self,
        universe: list[str],
        market_data: dict[str, Any],
    ) -> list[AgentSignal]:
        """Analyse sentiment data for every ticker and emit trading signals.

        Expected ``market_data`` keys
        ------------------------------
        ``"sentiment_signals"``
            ``dict[str, list[SentimentSignal]]`` -- per-ticker list of recent
            sentiment signals from the ``SentimentEngine``.
        ``"prices"``
            ``dict[str, pd.DataFrame]`` -- OHLCV DataFrames keyed by ticker.
        ``"sentiment_history"``
            ``dict[str, pd.DataFrame]`` -- daily sentiment score history per
            ticker with columns ``["date", "score", "volume", "keywords"]``.
        ``"fii_dii_flow"``
            ``dict[str, float]`` -- most recent net institutional flow per
            ticker (positive = buying).
        ``"social_buzz"``
            ``dict[str, dict]`` -- per-ticker social metrics with keys
            ``"mention_velocity"`` (current) and ``"avg_velocity"`` (30-day).
        """
        sentiment_signals: dict[str, list[Any]] = market_data.get("sentiment_signals", {})
        prices: dict[str, pd.DataFrame] = market_data.get("prices", {})
        sentiment_history: dict[str, pd.DataFrame] = market_data.get("sentiment_history", {})
        fii_dii_flow: dict[str, float] = market_data.get("fii_dii_flow", {})
        social_buzz: dict[str, dict[str, float]] = market_data.get("social_buzz", {})

        signals: list[AgentSignal] = []

        # Process tickers concurrently (IO-bound sentiment engine calls are
        # already awaitable; factor scoring is CPU-light).
        tasks = [
            self._evaluate_ticker(
                ticker=ticker,
                ticker_sentiments=sentiment_signals.get(ticker, []),
                price_df=prices.get(ticker, pd.DataFrame()),
                history_df=sentiment_history.get(ticker, pd.DataFrame()),
                flow=fii_dii_flow.get(ticker, 0.0),
                buzz=social_buzz.get(ticker, {}),
            )
            for ticker in universe
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for ticker, result in zip(universe, results, strict=False):
            if isinstance(result, Exception):
                self._log.error("evaluate_ticker.failed", ticker=ticker, error=str(result))
                continue
            if result is not None:
                signals.append(result)

        # Enforce portfolio-level constraints: rank by conviction, take top N.
        signals.sort(key=lambda s: s.conviction, reverse=True)
        signals = signals[:_MAX_POSITIONS]

        self._log.info(
            "generate_signals.complete",
            universe_size=len(universe),
            signals_emitted=len(signals),
        )
        return signals

    # ------------------------------------------------------------------
    # Per-ticker evaluation
    # ------------------------------------------------------------------

    async def _evaluate_ticker(
        self,
        ticker: str,
        ticker_sentiments: list[Any],
        price_df: pd.DataFrame,
        history_df: pd.DataFrame,
        flow: float,
        buzz: dict[str, float],
    ) -> AgentSignal | None:
        """Score a single ticker across all six sentiment factors.

        Returns ``None`` if the ticker does not pass the minimum signal
        threshold or if data is insufficient.
        """
        # ---- Guard: need at least some sentiment data --------------------
        if not ticker_sentiments and history_df.empty:
            return None
        if price_df.empty or len(price_df) < 2:
            return None

        current_price = float(price_df["close"].iloc[-1])

        # ---- Factor 1: Sentiment Score (24 h FinBERT average) ------------
        sentiment_score = self._compute_sentiment_score(ticker_sentiments)

        # ---- Factor 2: Sentiment Momentum (3d avg - 14d avg) -------------
        sentiment_momentum = self._compute_sentiment_momentum(history_df)

        # ---- Factor 3: Article Volume Spike (today / 30d avg) ------------
        volume_spike_ratio = self._compute_volume_spike(history_df)

        # ---- Factor 4: Growth Narrative Keywords -------------------------
        keyword_score = self._compute_keyword_score(ticker_sentiments)

        # ---- Factor 5: FII / DII Flow Direction --------------------------
        flow_score = self._normalise_flow(flow)

        # ---- Factor 6: Social Buzz velocity ------------------------------
        buzz_score = self._compute_buzz_score(buzz)

        # ---- Weighted composite ------------------------------------------
        factor_values: dict[str, float] = {
            "sentiment_score": sentiment_score,
            "sentiment_momentum": sentiment_momentum,
            "article_volume_spike": volume_spike_ratio,
            "growth_narrative": keyword_score,
            "fii_dii_flow": flow_score,
            "social_buzz": buzz_score,
        }
        composite = sum(
            self._factor_weights[k] * v for k, v in factor_values.items()
        )

        # ---- Classify trade type -----------------------------------------
        trade_type = self._classify_trade(
            sentiment_score=sentiment_score,
            sentiment_momentum=sentiment_momentum,
            volume_spike_ratio=volume_spike_ratio,
            history_df=history_df,
        )

        if trade_type is None:
            return None  # No actionable signal for this ticker.

        # ---- Determine action and conviction -----------------------------
        action, raw_conviction, reasoning = self._determine_action(
            trade_type=trade_type,
            composite=composite,
            sentiment_score=sentiment_score,
            sentiment_momentum=sentiment_momentum,
            volume_spike_ratio=volume_spike_ratio,
        )

        # ---- Apply growth keyword bonus ----------------------------------
        has_growth_keywords = self._has_growth_keywords(ticker_sentiments)
        if has_growth_keywords and action == Action.BUY:
            raw_conviction *= self._growth_keyword_bonus
            reasoning += " | Growth-keyword bonus applied (1.3x)."

        conviction = self._compute_conviction(
            composite_score=raw_conviction,
            min_score=0.0,
            max_score=2.5,
        )
        if conviction < 20:
            return None  # Below minimum conviction threshold.

        # ---- Stop-loss / take-profit levels ------------------------------
        if trade_type == "event":
            stop_loss = current_price * (1.0 - self._event_stop_loss)
            take_profit = current_price * (1.0 + self._event_take_profit)
            holding_period = 3
        elif trade_type == "contrarian":
            stop_loss = current_price * (1.0 - self._event_stop_loss)
            take_profit = current_price * (1.0 - self._event_take_profit)  # Selling
            holding_period = 5
        else:  # trend
            stop_loss = current_price * (1.0 - self._trend_stop_loss)
            take_profit = current_price * (1.0 + self._trend_take_profit)
            holding_period = 7

        # ---- Target weight (conviction-proportional, capped) -------------
        target_weight = min(
            (conviction / 100.0) * _MAX_WEIGHT_PER_POSITION,
            _MAX_WEIGHT_PER_POSITION,
        )

        return AgentSignal(
            ticker=ticker,
            action=action,
            conviction=conviction,
            target_weight=round(target_weight, 4),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            factor_scores={k: round(v, 4) for k, v in factor_values.items()},
            reasoning=reasoning,
            holding_period_days=holding_period,
            agent_name=self._name,
            timestamp=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Factor computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sentiment_score(sentiments: list[Any]) -> float:
        """Average FinBERT score over the last 24 h of sentiment signals.

        Returns a value in ``[-1.0, 1.0]``.  Falls back to ``0.0`` when no
        signals are available.
        """
        if not sentiments:
            return 0.0
        scores = [float(s.score) for s in sentiments if hasattr(s, "score")]
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _compute_sentiment_momentum(history_df: pd.DataFrame) -> float:
        """Sentiment momentum: 3-day rolling average minus 14-day rolling average.

        Positive values indicate improving sentiment -- the strongest factor.
        """
        if history_df.empty or "score" not in history_df.columns:
            return 0.0

        scores = history_df["score"].astype(float)
        if len(scores) < 14:
            return 0.0

        avg_3d = float(scores.iloc[-3:].mean())
        avg_14d = float(scores.iloc[-14:].mean())
        return avg_3d - avg_14d

    @staticmethod
    def _compute_volume_spike(history_df: pd.DataFrame) -> float:
        """Article volume spike: today's count / 30-day average.

        A ratio > 2.0 flags an information event.
        """
        if history_df.empty or "volume" not in history_df.columns:
            return 1.0

        volumes = history_df["volume"].astype(float)
        if len(volumes) < 2:
            return 1.0

        today_vol = float(volumes.iloc[-1])
        avg_30d = float(volumes.iloc[-30:].mean()) if len(volumes) >= 30 else float(volumes.mean())
        if avg_30d <= 0:
            return 1.0
        return today_vol / avg_30d

    @staticmethod
    def _compute_keyword_score(sentiments: list[Any]) -> float:
        """Score based on presence of growth-positive and negative keywords.

        Returns a value in ``[-1.0, 1.0]``.
        """
        if not sentiments:
            return 0.0

        positive_hits = 0
        negative_hits = 0
        total_checked = 0

        for s in sentiments:
            keywords: list[str] = getattr(s, "keywords", [])
            if not keywords:
                continue
            lowered = [kw.lower() for kw in keywords]
            total_checked += 1
            for kw in lowered:
                if any(pos in kw for pos in GROWTH_POSITIVE_KEYWORDS):
                    positive_hits += 1
                if any(neg in kw for neg in NEGATIVE_KEYWORDS):
                    negative_hits += 1

        if total_checked == 0:
            return 0.0

        # Normalise to [-1, 1] range.
        net = positive_hits - negative_hits
        max_possible = max(total_checked, 1)
        return float(np.clip(net / max_possible, -1.0, 1.0))

    @staticmethod
    def _has_growth_keywords(sentiments: list[Any]) -> bool:
        """Return ``True`` if any sentiment signal contains growth-positive keywords."""
        for s in sentiments:
            keywords: list[str] = getattr(s, "keywords", [])
            lowered = [kw.lower() for kw in keywords]
            for kw in lowered:
                if any(pos in kw for pos in GROWTH_POSITIVE_KEYWORDS):
                    return True
        return False

    @staticmethod
    def _normalise_flow(flow: float) -> float:
        """Normalise FII/DII net flow to ``[-1, 1]``.

        Assumes flow is in crore INR.  +-500 crore maps to +-1.0.
        """
        return float(np.clip(flow / 500.0, -1.0, 1.0))

    @staticmethod
    def _compute_buzz_score(buzz: dict[str, float]) -> float:
        """Social buzz: mention velocity vs. historical average.

        Returns a normalised ``[-1, 1]`` score.
        """
        velocity = buzz.get("mention_velocity", 0.0)
        avg_velocity = buzz.get("avg_velocity", 0.0)
        if avg_velocity <= 0:
            return 0.0
        ratio = velocity / avg_velocity
        # Map ratio to [-1, 1]: ratio of 1.0 is neutral, 3.0+ saturates at 1.0.
        return float(np.clip((ratio - 1.0) / 2.0, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Trade classification
    # ------------------------------------------------------------------

    def _classify_trade(
        self,
        sentiment_score: float,
        sentiment_momentum: float,
        volume_spike_ratio: float,
        history_df: pd.DataFrame,
    ) -> str | None:
        """Classify a potential trade as ``"event"``, ``"trend"``, ``"contrarian"``, or ``None``.

        Classification hierarchy (first match wins):

        1. **Event trade** -- sentiment spike > ``event_sigma`` standard
           deviations from the historical mean *and* volume spike above
           threshold.
        2. **Contrarian fade** -- sentiment pinned above
           ``contrarian_threshold`` for ``contrarian_days`` consecutive days.
        3. **Trend trade** -- sustained positive sentiment momentum over the
           last 3+ days.
        4. ``None`` -- no actionable signal.
        """
        # --- Event detection: z-score of latest sentiment vs. history -----
        if not history_df.empty and "score" in history_df.columns:
            scores = history_df["score"].astype(float)
            if len(scores) >= 14:
                mean = float(scores.mean())
                std = float(scores.std(ddof=0))
                if std > 0:
                    z = (sentiment_score - mean) / std
                    if abs(z) > self._event_sigma and volume_spike_ratio > self._volume_spike_threshold:
                        return "event"

        # --- Contrarian detection -----------------------------------------
        if not history_df.empty and "score" in history_df.columns:
            scores = history_df["score"].astype(float)
            if len(scores) >= self._contrarian_days:
                recent = scores.iloc[-self._contrarian_days:]
                if all(s > self._contrarian_threshold for s in recent):
                    return "contrarian"

        # --- Trend detection ----------------------------------------------
        if sentiment_momentum > 0.05:
            # Verify momentum is sustained (3+ days of improvement).
            if not history_df.empty and "score" in history_df.columns and len(history_df) >= 3:
                recent_scores = history_df["score"].astype(float).iloc[-3:]
                if recent_scores.is_monotonic_increasing or sentiment_momentum > 0.10:
                    return "trend"

        return None

    # ------------------------------------------------------------------
    # Action determination
    # ------------------------------------------------------------------

    def _determine_action(
        self,
        trade_type: str,
        composite: float,
        sentiment_score: float,
        sentiment_momentum: float,
        volume_spike_ratio: float,
    ) -> tuple[Action, float, str]:
        """Determine action, raw conviction multiplier, and reasoning string.

        Returns
        -------
        tuple[Action, float, str]
            ``(action, raw_conviction, reasoning)``
        """
        if trade_type == "event":
            if sentiment_score > 0:
                action = Action.BUY
                raw_conviction = abs(composite) * (1.0 + volume_spike_ratio / 5.0)
                reasoning = (
                    f"Event trade: sentiment spike ({sentiment_score:.2f}) with "
                    f"volume spike ({volume_spike_ratio:.1f}x). "
                    f"Composite={composite:.3f}."
                )
            else:
                action = Action.SELL
                raw_conviction = abs(composite) * (1.0 + volume_spike_ratio / 5.0)
                reasoning = (
                    f"Event trade (bearish): negative sentiment spike "
                    f"({sentiment_score:.2f}) with volume spike "
                    f"({volume_spike_ratio:.1f}x). Composite={composite:.3f}."
                )

        elif trade_type == "contrarian":
            action = Action.SELL
            raw_conviction = abs(composite) * 0.8  # Slightly reduced for fade trades.
            reasoning = (
                f"Contrarian fade: sentiment pinned above "
                f"{self._contrarian_threshold} for {self._contrarian_days}+ days. "
                f"Mean-reversion SELL. Composite={composite:.3f}."
            )

        else:  # trend
            action = Action.BUY if sentiment_momentum > 0 else Action.SELL
            raw_conviction = abs(composite)
            reasoning = (
                f"Trend trade: sustained sentiment momentum "
                f"({sentiment_momentum:+.3f} over 3d). "
                f"Composite={composite:.3f}."
            )

        return action, raw_conviction, reasoning
