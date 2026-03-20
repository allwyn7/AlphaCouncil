"""NewsShockDetector — detects abnormal news events that should override normal signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class NewsShock:
    """Result of shock detection for a single ticker."""
    is_shock: bool
    severity: float  # 0.0-1.0
    direction: str   # "positive" / "negative" / "neutral"
    article_spike_ratio: float
    sentiment_score: float
    description: str


class NewsShockDetector:
    """Detects abnormal news/sentiment spikes that could invalidate normal signals.

    A shock is detected when:
    - Article volume > 2.5x the 30-day average AND
    - abs(sentiment_score) > 0.4
    """

    def __init__(
        self,
        article_spike_threshold: float = 2.5,
        sentiment_threshold: float = 0.4,
    ) -> None:
        self._spike_threshold = article_spike_threshold
        self._sent_threshold = sentiment_threshold

    def detect_shock(
        self,
        ticker: str,
        sentiment_signals: dict[str, Any] | None = None,
        sentiment_history: dict[str, Any] | None = None,
    ) -> NewsShock:
        """Check if a ticker is experiencing a news shock.

        Parameters
        ----------
        ticker:
            Stock ticker.
        sentiment_signals:
            Current sentiment data for the ticker. Expected keys:
            score (float), article_count (int), avg_article_count_30d (float).
        sentiment_history:
            Historical sentiment. Expected keys:
            sentiment_7d (float), sentiment_30d (float).
        """
        if not sentiment_signals:
            return NewsShock(
                is_shock=False, severity=0.0, direction="neutral",
                article_spike_ratio=0.0, sentiment_score=0.0,
                description="No sentiment data available",
            )

        score = float(sentiment_signals.get("score", 0.0))
        article_count = int(sentiment_signals.get("article_count", 0))
        avg_30d = float(sentiment_signals.get("avg_article_count_30d", max(article_count, 1)))

        # Also check sentiment history for momentum
        if sentiment_history:
            sent_7d = float(sentiment_history.get("sentiment_7d", score))
            sent_30d = float(sentiment_history.get("sentiment_30d", score))
        else:
            sent_7d = score
            sent_30d = score

        # Compute spike ratio
        spike_ratio = article_count / max(avg_30d, 1.0)

        # Determine if shock
        is_shock = spike_ratio > self._spike_threshold and abs(score) > self._sent_threshold

        # Severity: 0 to 1
        if is_shock:
            severity = min(1.0, (spike_ratio / 5.0) * abs(score))
        else:
            severity = 0.0

        # Direction
        if score > self._sent_threshold:
            direction = "positive"
        elif score < -self._sent_threshold:
            direction = "negative"
        else:
            direction = "neutral"

        # Description
        if is_shock:
            description = (
                f"NEWS SHOCK for {ticker}: {article_count} articles "
                f"({spike_ratio:.1f}x avg), sentiment={score:.2f} ({direction}), "
                f"severity={severity:.2f}"
            )
            logger.warning("news_shock_detected", ticker=ticker, severity=severity,
                          direction=direction, spike_ratio=spike_ratio, score=score)
        else:
            description = f"No shock: {article_count} articles ({spike_ratio:.1f}x avg), sentiment={score:.2f}"

        return NewsShock(
            is_shock=is_shock,
            severity=severity,
            direction=direction,
            article_spike_ratio=spike_ratio,
            sentiment_score=score,
            description=description,
        )
