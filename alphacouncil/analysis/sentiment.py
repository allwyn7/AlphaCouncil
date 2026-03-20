"""Sentiment analysis engine using FinBERT and RSS / Reddit feeds.

Aggregates headlines from Indian financial news RSS feeds and Reddit
communities, scores them with *ProsusAI/finbert* via the Hugging Face
``transformers`` pipeline, and produces frozen
:class:`~alphacouncil.core.models.SentimentSignal` models.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import feedparser
import structlog
from transformers import pipeline as hf_pipeline

from alphacouncil.core.models import SentimentSignal

if TYPE_CHECKING:
    from alphacouncil.core.cache import TieredCache

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_NS = "sentiment"
_CACHE_TTL_S = 900  # 15 minutes

# FinBERT batch inference settings
_FINBERT_MODEL = "ProsusAI/finbert"
_FINBERT_BATCH_SIZE = 32

# RSS feed sources for Indian financial news
RSS_FEEDS: dict[str, str] = {
    "moneycontrol": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "livemint": "https://www.livemint.com/rss/markets",
    "reuters_india": "https://feeds.reuters.com/reuters/INtopNews",
}

# Keyword boosters for growth-oriented headlines
_POSITIVE_KEYWORDS: list[str] = [
    "revenue beat",
    "growth",
    "expansion",
    "market share",
    "new product",
    "order win",
    "capacity addition",
]

_NEGATIVE_KEYWORDS: list[str] = [
    "downgrade",
    "miss",
    "fraud",
    "sebi probe",
    "debt concern",
]

# Default Reddit subreddits for Indian stock discussions
_DEFAULT_SUBREDDITS: list[str] = ["IndianStreetBets", "IndianStockMarket"]


# ---------------------------------------------------------------------------
# SentimentEngine
# ---------------------------------------------------------------------------


class SentimentEngine:
    """NLP-powered sentiment analysis for Indian equity markets.

    Parameters
    ----------
    cache:
        A :class:`TieredCache` instance for caching per-ticker sentiment.
    """

    def __init__(self, cache: TieredCache) -> None:
        self._cache = cache
        self._finbert: object | None = None  # lazy-loaded HF pipeline

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_finbert(self):
        """Lazy-load the FinBERT pipeline on first use."""
        if self._finbert is None:
            logger.info("loading_finbert", model=_FINBERT_MODEL)
            self._finbert = hf_pipeline(
                "sentiment-analysis",
                model=_FINBERT_MODEL,
                tokenizer=_FINBERT_MODEL,
                truncation=True,
                max_length=512,
            )
        return self._finbert

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def fetch_headlines(self) -> list[dict]:
        """Parse all configured RSS feeds and return headline dicts.

        Each dict contains: ``{"title", "link", "published", "source"}``.
        """
        all_headlines: list[dict] = []

        async def _parse_feed(source: str, url: str) -> list[dict]:
            """Parse a single RSS feed off the event loop."""
            try:
                feed = await asyncio.to_thread(feedparser.parse, url)
                items: list[dict] = []
                for entry in feed.entries:
                    items.append({
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": source,
                    })
                logger.debug("rss_fetched", source=source, count=len(items))
                return items
            except Exception:  # noqa: BLE001
                logger.warning("rss_fetch_failed", source=source, url=url)
                return []

        tasks = [_parse_feed(src, url) for src, url in RSS_FEEDS.items()]
        results = await asyncio.gather(*tasks)
        for batch in results:
            all_headlines.extend(batch)

        logger.info("headlines_fetched", total=len(all_headlines))
        return all_headlines

    async def fetch_reddit(
        self,
        subreddits: list[str] | None = None,
    ) -> list[dict]:
        """Fetch recent posts from Indian stock-market subreddits.

        Requires the ``asyncpraw`` package and valid Reddit API credentials
        set via environment variables (``REDDIT_CLIENT_ID``,
        ``REDDIT_CLIENT_SECRET``, ``REDDIT_USER_AGENT``).

        Returns a list of dicts: ``{"title", "link", "score", "source"}``.
        Falls back to an empty list if credentials or the library are
        unavailable.
        """
        subs = subreddits or _DEFAULT_SUBREDDITS

        try:
            import os

            import asyncpraw  # type: ignore[import-untyped]

            client_id = os.environ.get("REDDIT_CLIENT_ID", "")
            client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
            user_agent = os.environ.get("REDDIT_USER_AGENT", "alphacouncil:v0.1")

            if not client_id or not client_secret:
                logger.warning("reddit_credentials_missing")
                return []

            reddit = asyncpraw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )

            posts: list[dict] = []
            for sub_name in subs:
                try:
                    subreddit = await reddit.subreddit(sub_name)
                    async for submission in subreddit.hot(limit=25):
                        posts.append({
                            "title": submission.title,
                            "link": f"https://reddit.com{submission.permalink}",
                            "score": submission.score,
                            "source": f"reddit/{sub_name}",
                        })
                except Exception:  # noqa: BLE001
                    logger.warning("reddit_sub_failed", subreddit=sub_name)

            await reddit.close()
            logger.info("reddit_fetched", total=len(posts))
            return posts

        except ImportError:
            logger.warning("asyncpraw_not_installed")
            return []
        except Exception:  # noqa: BLE001
            logger.warning("reddit_fetch_failed")
            return []

    # ------------------------------------------------------------------
    # Ticker mapping
    # ------------------------------------------------------------------

    @staticmethod
    def map_ticker(headline: str, universe: list[str]) -> Optional[str]:
        """Map a headline string to a ticker from the universe.

        Uses case-insensitive substring matching on the ticker symbol and
        common variations (e.g. ``"Reliance"`` matches ``"RELIANCE"``).

        Returns ``None`` when no match is found.
        """
        headline_upper = headline.upper()
        for ticker in universe:
            # Match the raw ticker name (e.g. RELIANCE, TCS, INFY)
            clean = ticker.replace(".NS", "").upper()
            # Whole-word boundary check to avoid partial matches
            pattern = rf"\b{re.escape(clean)}\b"
            if re.search(pattern, headline_upper):
                return ticker
        return None

    # ------------------------------------------------------------------
    # FinBERT inference
    # ------------------------------------------------------------------

    async def analyze_batch(self, headlines: list[str]) -> list[float]:
        """Run FinBERT batch inference and return normalised sentiment scores.

        Each score is in ``[-1.0, 1.0]`` (negative / neutral / positive).

        Parameters
        ----------
        headlines:
            A list of headline strings to classify.
        """
        if not headlines:
            return []

        scores: list[float] = []

        # Process in batches of _FINBERT_BATCH_SIZE
        for start in range(0, len(headlines), _FINBERT_BATCH_SIZE):
            batch = headlines[start : start + _FINBERT_BATCH_SIZE]
            batch_scores = await asyncio.to_thread(self._infer_batch, batch)
            scores.extend(batch_scores)

        return scores

    def _infer_batch(self, texts: list[str]) -> list[float]:
        """Synchronous FinBERT inference for one batch of texts."""
        pipe = self._get_finbert()
        results = pipe(texts, batch_size=_FINBERT_BATCH_SIZE)

        scores: list[float] = []
        for res in results:
            label = res["label"].lower()
            raw_score = float(res["score"])

            if label == "positive":
                scores.append(raw_score)
            elif label == "negative":
                scores.append(-raw_score)
            else:
                # neutral -- contribute a small magnitude toward zero
                scores.append(0.0)

        return scores

    # ------------------------------------------------------------------
    # Per-ticker sentiment
    # ------------------------------------------------------------------

    async def get_ticker_sentiment(self, ticker: str) -> SentimentSignal:
        """Aggregate headline sentiment for a specific ticker.

        Fetches headlines, filters to those mentioning the ticker, runs
        FinBERT, applies keyword boosters, and returns a frozen signal.
        """
        cache_key = f"{_CACHE_NS}:ticker:{ticker}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug("cache_hit", key=cache_key)
            return cached

        headlines = await self.fetch_headlines()
        reddit_posts = await self.fetch_reddit()

        # Merge all text sources
        all_items = headlines + reddit_posts

        # Filter for this ticker
        relevant = [
            item for item in all_items
            if self.map_ticker(item.get("title", ""), [ticker]) is not None
        ]

        titles = [item["title"] for item in relevant]

        if not titles:
            signal = SentimentSignal(
                ticker=ticker,
                score=0.0,
                volume=0,
                trend=0.0,
                keywords=[],
                source="aggregate",
                timestamp=datetime.now(tz=timezone.utc),
            )
            await self._cache.set(cache_key, signal, ttl=_CACHE_TTL_S)
            return signal

        raw_scores = await self.analyze_batch(titles)

        # Apply keyword boosters
        adjusted_scores: list[float] = []
        detected_keywords: list[str] = []
        for title, score in zip(titles, raw_scores):
            boost = self._keyword_boost(title)
            adjusted_scores.append(_clamp(score + boost, -1.0, 1.0))
            detected_keywords.extend(self._extract_keywords(title))

        avg_score = sum(adjusted_scores) / len(adjusted_scores)

        # Deduplicate keywords, keep order
        seen: set[str] = set()
        unique_keywords: list[str] = []
        for kw in detected_keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        signal = SentimentSignal(
            ticker=ticker,
            score=round(_clamp(avg_score, -1.0, 1.0), 4),
            volume=len(titles),
            trend=0.0,  # requires historical comparison (future enhancement)
            keywords=unique_keywords[:10],  # cap at 10
            source="aggregate",
            timestamp=datetime.now(tz=timezone.utc),
        )

        await self._cache.set(cache_key, signal, ttl=_CACHE_TTL_S)
        logger.info(
            "ticker_sentiment_complete",
            ticker=ticker,
            score=signal.score,
            volume=signal.volume,
        )
        return signal

    # ------------------------------------------------------------------
    # Market-wide sentiment
    # ------------------------------------------------------------------

    async def get_market_sentiment(self) -> dict:
        """Compute an overall market fear/greed proxy.

        Analyses all fetched headlines (not filtered to a single ticker)
        and returns aggregate statistics.

        Returns
        -------
        dict
            ``{"score": float, "label": str, "volume": int,
              "positive_pct": float, "negative_pct": float,
              "neutral_pct": float, "timestamp": str}``
        """
        cache_key = f"{_CACHE_NS}:market"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        headlines = await self.fetch_headlines()
        titles = [h["title"] for h in headlines if h.get("title")]

        if not titles:
            result = {
                "score": 0.0,
                "label": "NEUTRAL",
                "volume": 0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 100.0,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
            return result

        scores = await self.analyze_batch(titles)

        positive = sum(1 for s in scores if s > 0.1)
        negative = sum(1 for s in scores if s < -0.1)
        neutral = len(scores) - positive - negative

        total = len(scores)
        avg_score = sum(scores) / total

        if avg_score > 0.15:
            label = "GREED"
        elif avg_score > 0.05:
            label = "MILD_GREED"
        elif avg_score < -0.15:
            label = "FEAR"
        elif avg_score < -0.05:
            label = "MILD_FEAR"
        else:
            label = "NEUTRAL"

        result = {
            "score": round(avg_score, 4),
            "label": label,
            "volume": total,
            "positive_pct": round(positive / total * 100, 2),
            "negative_pct": round(negative / total * 100, 2),
            "neutral_pct": round(neutral / total * 100, 2),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        await self._cache.set(cache_key, result, ttl=_CACHE_TTL_S)
        logger.info("market_sentiment_complete", label=label, score=result["score"])
        return result

    # ------------------------------------------------------------------
    # Keyword helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_boost(headline: str) -> float:
        """Return a sentiment boost based on domain-specific keywords.

        Positive keywords add +0.10, negative keywords add -0.10.
        Boosts are cumulative but the total is capped at +/-0.3.
        """
        text = headline.lower()
        boost = 0.0

        for kw in _POSITIVE_KEYWORDS:
            if kw in text:
                boost += 0.10

        for kw in _NEGATIVE_KEYWORDS:
            if kw in text:
                boost -= 0.10

        return _clamp(boost, -0.3, 0.3)

    @staticmethod
    def _extract_keywords(headline: str) -> list[str]:
        """Return any growth / risk keywords present in the headline."""
        text = headline.lower()
        found: list[str] = []
        for kw in _POSITIVE_KEYWORDS + _NEGATIVE_KEYWORDS:
            if kw in text:
                found.append(kw)
        return found


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the interval [lo, hi]."""
    return max(lo, min(hi, value))
