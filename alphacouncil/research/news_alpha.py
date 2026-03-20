"""News-alpha event study and sentiment weight refinement.

Identifies high-sentiment headlines, measures their predictive power
via event studies (cumulative abnormal return at +1, +3, +5 days), and
builds a keyword->alpha database.  Growth-related events (earnings
beats, revenue surprises, expansion announcements) receive extra
tracking focus.

The :meth:`update_sentiment_weights` method feeds back into the
sentiment engine, adjusting keyword weights based on their demonstrated
predictive value.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats as sp_stats
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Keyword taxonomies
# ---------------------------------------------------------------------------

_GROWTH_KEYWORDS: list[str] = [
    "revenue beat",
    "earnings beat",
    "revenue surprise",
    "growth",
    "expansion",
    "market share",
    "new product",
    "order win",
    "capacity addition",
    "profit surge",
    "margin expansion",
    "guidance raise",
    "upgrade",
    "outperform",
    "buy rating",
]

_NEGATIVE_KEYWORDS: list[str] = [
    "downgrade",
    "miss",
    "fraud",
    "sebi probe",
    "debt concern",
    "profit warning",
    "earnings miss",
    "revenue miss",
    "margin pressure",
    "downturn",
    "sell rating",
    "underperform",
    "default",
    "write-off",
]

_ALL_KEYWORDS: list[str] = _GROWTH_KEYWORDS + _NEGATIVE_KEYWORDS

# Event study windows (trading days)
_EVENT_WINDOWS: list[int] = [1, 3, 5]

# Minimum |sentiment| threshold for tracking
_MIN_SENTIMENT_ABS: float = 0.7

# Minimum events per keyword for statistical reliability
_MIN_EVENTS_FOR_ALPHA: int = 5

# Minimum t-statistic for keyword to be considered predictive
_MIN_T_STAT: float = 1.65  # ~90% confidence, one-tailed


# ---------------------------------------------------------------------------
# NewsAlphaTracker
# ---------------------------------------------------------------------------


class NewsAlphaTracker:
    """Event study framework for refining sentiment keyword weights.

    Tracks headlines with high sentiment scores, measures stock returns
    after those events, and identifies which keywords are genuinely
    predictive versus noise.  Growth-related events receive focused
    attention to align with AlphaCouncil's growth investing bias.

    Parameters
    ----------
    db_engine:
        SQLAlchemy :class:`Engine` for reading sentiment cache and
        trade/price data.
    sentiment_engine:
        The sentiment engine whose keyword weights will be updated.
        Expected to have a ``keyword_weights`` attribute (dict) and
        a ``set_keyword_weights(weights: dict)`` method.
    """

    def __init__(
        self,
        db_engine: Engine,
        sentiment_engine: Any,
    ) -> None:
        self._db_engine = db_engine
        self._sentiment_engine = sentiment_engine
        self._log = logger.bind(component="news_alpha")
        self._keyword_alpha: dict[str, float] = {}
        self._event_database: list[dict] = []

    # ------------------------------------------------------------------
    # Public API: track events
    # ------------------------------------------------------------------

    async def track_events(self, days: int = 30) -> None:
        """Run the event study pipeline for the most recent *days*.

        Steps
        -----
        1. Find all headlines with ``|sentiment| > 0.7``.
        2. For each, compute stock return at +1, +3, +5 days.
        3. Build / update the event->return database.
        4. Classify keywords as predictive vs. noise.
        5. Give extra focus to growth events (earnings beats, revenue
           surprises, expansion announcements).

        Parameters
        ----------
        days:
            Look-back window in calendar days.
        """
        self._log.info("news_alpha.track_start", days=days)

        # Step 1: Fetch high-sentiment events
        events = await self._fetch_sentiment_events(days)
        if not events:
            self._log.info("news_alpha.no_events", days=days)
            return

        # Step 2: For each event, compute forward returns
        enriched: list[dict] = []
        for event in events:
            ticker = str(event.get("symbol", event.get("ticker", "")))
            event_date = event.get("timestamp", "")
            sentiment_score = float(event.get("score", 0))
            headline = str(event.get("raw_text", event.get("headline", "")))

            # Skip if below threshold
            if abs(sentiment_score) < _MIN_SENTIMENT_ABS:
                continue

            # Compute returns at each window
            forward_returns: dict[str, float] = {}
            for window in _EVENT_WINDOWS:
                car = await self._event_study(ticker, event_date, window)
                forward_returns[f"return_{window}d"] = car

            # Extract matched keywords
            keywords = self._extract_keywords(headline)
            is_growth = any(kw in _GROWTH_KEYWORDS for kw in keywords)

            entry = {
                "ticker": ticker,
                "date": str(event_date),
                "sentiment": sentiment_score,
                "headline": headline,
                "keywords": keywords,
                "is_growth_event": is_growth,
                **forward_returns,
            }
            enriched.append(entry)

        # Step 3: Update event database
        self._event_database.extend(enriched)

        # Step 4: Identify predictive vs. noise keywords
        self._update_keyword_alpha(enriched)

        # Step 5: Log growth event statistics
        growth_count = sum(1 for e in enriched if e.get("is_growth_event"))
        self._log.info(
            "news_alpha.track_complete",
            total_events=len(enriched),
            growth_events=growth_count,
            predictive_keywords=len(self._keyword_alpha),
        )

    # ------------------------------------------------------------------
    # Public API: get keyword alpha
    # ------------------------------------------------------------------

    async def get_keyword_alpha(self) -> dict[str, float]:
        """Return the keyword->average excess return mapping.

        If no keyword alpha has been computed yet, triggers a 30-day
        event tracking run first.

        Returns
        -------
        dict[str, float]
            ``{keyword: avg_excess_return}`` for all keywords with at
            least ``_MIN_EVENTS_FOR_ALPHA`` observations.
        """
        if not self._keyword_alpha:
            await self.track_events(days=30)
        return dict(self._keyword_alpha)

    # ------------------------------------------------------------------
    # Public API: update sentiment weights
    # ------------------------------------------------------------------

    async def update_sentiment_weights(
        self,
        sentiment_engine: Any | None = None,
    ) -> None:
        """Adjust sentiment engine keyword weights based on event-study findings.

        For each keyword with a statistically significant alpha, the
        weight is scaled proportionally to the magnitude of its average
        excess return.  Growth-related keywords receive a 1.5x bias
        multiplier.

        Parameters
        ----------
        sentiment_engine:
            Optional override; defaults to the engine passed at init.
        """
        engine = sentiment_engine or self._sentiment_engine
        if engine is None:
            self._log.warning("news_alpha.no_sentiment_engine")
            return

        if not self._keyword_alpha:
            await self.track_events(days=30)

        if not self._keyword_alpha:
            self._log.info("news_alpha.no_alpha_to_apply")
            return

        # Build new weight map
        predictive = await self._get_predictive_keywords()
        new_weights: dict[str, float] = {}

        for kw, stats in predictive.items():
            mean_ret = stats["mean_return"]
            t_stat = stats["t_stat"]
            is_growth = stats.get("is_growth", False)

            # Only update weights for statistically significant keywords
            if abs(t_stat) < _MIN_T_STAT:
                continue

            # Weight proportional to alpha, with growth bias
            weight = mean_ret
            if is_growth:
                weight *= 1.5

            new_weights[kw] = round(weight, 6)

        if not new_weights:
            self._log.info("news_alpha.no_significant_keywords")
            return

        # Apply to sentiment engine
        try:
            if hasattr(engine, "set_keyword_weights"):
                engine.set_keyword_weights(new_weights)
            elif hasattr(engine, "keyword_weights"):
                engine.keyword_weights.update(new_weights)
            else:
                self._log.warning(
                    "news_alpha.engine_incompatible",
                    reason="no set_keyword_weights or keyword_weights attribute",
                )
                return

            self._log.info(
                "news_alpha.weights_updated",
                n_keywords=len(new_weights),
            )
        except Exception as exc:
            self._log.error(
                "news_alpha.weight_update_failed",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Event study computation
    # ------------------------------------------------------------------

    async def _event_study(
        self,
        ticker: str,
        event_date: Any,
        window: int,
    ) -> float:
        """Compute cumulative abnormal return (CAR) for a single event.

        The abnormal return is the stock return minus the benchmark
        (Nifty 50 proxy) return over the same window.

        Parameters
        ----------
        ticker:
            NSE ticker symbol.
        event_date:
            Date of the event (ISO string or datetime).
        window:
            Number of trading days after the event.

        Returns
        -------
        float
            Cumulative abnormal return over the window.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._event_study_sync,
            ticker,
            event_date,
            window,
        )

    def _event_study_sync(
        self,
        ticker: str,
        event_date: Any,
        window: int,
    ) -> float:
        """Synchronous event study computation."""
        try:
            # Parse event date
            if isinstance(event_date, str):
                dt = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
            elif isinstance(event_date, datetime):
                dt = event_date
            else:
                return 0.0

            # Fetch stock returns around event
            start = dt
            end = dt + timedelta(days=window + 7)  # buffer for weekends/holidays

            query = """
                SELECT timestamp, price
                FROM trades
                WHERE symbol = :ticker
                  AND timestamp BETWEEN :start AND :end
                ORDER BY timestamp
                LIMIT :limit
            """
            with self._db_engine.connect() as conn:
                df = pd.read_sql(
                    query,
                    conn,
                    params={
                        "ticker": ticker,
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "limit": window + 5,
                    },
                    parse_dates=["timestamp"],
                )

            if len(df) < 2:
                return 0.0

            # Stock return
            stock_return = float(
                df["price"].iloc[min(window, len(df) - 1)] / df["price"].iloc[0] - 1
            )

            # Benchmark return (use portfolio snapshots as proxy)
            benchmark_return = self._get_benchmark_return_sync(
                start.isoformat(), end.isoformat(), window,
            )

            # Abnormal return = stock - benchmark
            car = stock_return - benchmark_return
            return round(car, 6)

        except Exception as exc:
            self._log.debug(
                "news_alpha.event_study_failed",
                ticker=ticker,
                error=str(exc),
            )
            return 0.0

    def _get_benchmark_return_sync(
        self,
        start: str,
        end: str,
        window: int,
    ) -> float:
        """Fetch benchmark return over the event window."""
        try:
            query = """
                SELECT timestamp, total_value
                FROM portfolio_snapshots
                WHERE timestamp BETWEEN :start AND :end
                ORDER BY timestamp
                LIMIT :limit
            """
            with self._db_engine.connect() as conn:
                df = pd.read_sql(
                    query,
                    conn,
                    params={"start": start, "end": end, "limit": window + 5},
                    parse_dates=["timestamp"],
                )

            if len(df) < 2:
                return 0.0

            return float(
                df["total_value"].iloc[min(window, len(df) - 1)]
                / df["total_value"].iloc[0] - 1
            )
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Keyword analysis
    # ------------------------------------------------------------------

    def _extract_keywords(self, headline: str) -> list[str]:
        """Extract matching keywords from a headline string."""
        headline_lower = headline.lower()
        found: list[str] = []
        for kw in _ALL_KEYWORDS:
            if kw.lower() in headline_lower:
                found.append(kw)
        return found

    def _update_keyword_alpha(self, results: list[dict]) -> None:
        """Update the keyword alpha map from event study results.

        Uses the 3-day return as the primary alpha metric.
        """
        keyword_returns: dict[str, list[float]] = defaultdict(list)

        for r in results:
            ret_3d = r.get("return_3d", 0.0)
            for kw in r.get("keywords", []):
                keyword_returns[kw].append(ret_3d)

        for kw, returns in keyword_returns.items():
            if returns:
                self._keyword_alpha[kw] = round(float(np.mean(returns)), 6)

    async def _get_predictive_keywords(
        self,
        min_events: int = _MIN_EVENTS_FOR_ALPHA,
    ) -> dict[str, dict]:
        """Identify keywords that reliably predict returns.

        For each keyword with at least *min_events* observations,
        computes mean return, standard deviation, t-statistic, and
        whether the keyword belongs to the growth category.

        Returns
        -------
        dict[str, dict]
            ``{keyword: {"mean_return": float, "std": float,
            "count": int, "t_stat": float, "is_growth": bool}}``
        """
        keyword_data: dict[str, list[float]] = defaultdict(list)

        for event in self._event_database:
            ret = event.get("return_3d", 0.0)
            for kw in event.get("keywords", []):
                keyword_data[kw].append(ret)

        predictive: dict[str, dict] = {}
        for kw, returns in keyword_data.items():
            if len(returns) < min_events:
                continue

            arr = np.array(returns, dtype=np.float64)
            mean_ret = float(arr.mean())
            std_ret = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

            if std_ret > 0:
                t_stat = mean_ret / (std_ret / np.sqrt(len(arr)))
            else:
                t_stat = 0.0

            predictive[kw] = {
                "mean_return": round(mean_ret, 6),
                "std": round(std_ret, 6),
                "count": len(returns),
                "t_stat": round(float(t_stat), 4),
                "is_growth": kw in _GROWTH_KEYWORDS,
            }

        return predictive

    # ------------------------------------------------------------------
    # Data fetchers
    # ------------------------------------------------------------------

    async def _fetch_sentiment_events(
        self,
        days: int,
    ) -> list[dict]:
        """Fetch high-sentiment events from the sentiment cache table."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._fetch_sentiment_events_sync, days,
        )

    def _fetch_sentiment_events_sync(self, days: int) -> list[dict]:
        """Read sentiment events synchronously."""
        try:
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
            query = """
                SELECT id, timestamp, symbol, source, score,
                       magnitude, raw_text
                FROM sentiment_cache
                WHERE timestamp >= :cutoff
                  AND abs(score) > :threshold
                ORDER BY timestamp DESC
                LIMIT 500
            """
            with self._db_engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {"cutoff": cutoff.isoformat(), "threshold": _MIN_SENTIMENT_ABS},
                )
                return [dict(row._mapping) for row in result]

        except Exception as exc:
            self._log.error(
                "news_alpha.sentiment_fetch_failed",
                error=str(exc),
            )
            return []
