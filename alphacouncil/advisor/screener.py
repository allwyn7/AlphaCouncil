"""Stock screener with customizable filters and pre-built screening profiles.

Provides rapid screening of large stock universes using lightweight technical,
fundamental, and (optional) sentiment analysis. Designed for speed over depth
-- use the full InvestmentAdvisor.analyze() pipeline for individual deep dives.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import structlog
import yfinance as yf

from alphacouncil.advisor.engine import InvestmentAdvisor
from alphacouncil.advisor.models import (
    AdvisorAction,
    ScreenerFilter,
    ScreenerResult,
    ScreenerResultItem,
)
from alphacouncil.advisor.universes import get_universe  # noqa: F401 -- re-export convenience

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pre-built screening profiles
# ---------------------------------------------------------------------------

_PROFILES: dict[str, ScreenerFilter] = {
    "growth_picks": ScreenerFilter(
        min_revenue_growth=0.10,
        min_roe=0.12,
        positive_fcf=True,
        above_sma_200=True,
    ),
    "value_dips": ScreenerFilter(
        rsi_max=35,
        above_sma_200=False,
        max_pe=20,
        positive_fcf=True,
    ),
    "momentum_breakouts": ScreenerFilter(
        adx_min=25,
        above_sma_200=True,
        macd_bullish=True,
        rsi_min=50,
        rsi_max=70,
    ),
    "dividend_steady": ScreenerFilter(
        max_debt_to_equity=100,
        positive_fcf=True,
        min_roe=0.10,
    ),
    "turnaround_candidates": ScreenerFilter(
        rsi_max=40,
        min_revenue_growth=-0.10,
        max_pe=30,
    ),
}

# ---------------------------------------------------------------------------
# Lightweight indicator helpers (avoid pandas-ta full import for speed)
# ---------------------------------------------------------------------------


def _compute_rsi(closes: pd.Series, period: int = 14) -> float:
    """Wilder-smoothed RSI from a price Series."""
    if len(closes) < period + 1:
        return 50.0  # neutral fallback
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _compute_sma(closes: pd.Series, period: int = 200) -> Optional[float]:
    """Simple moving average (last value) or None if insufficient data."""
    if len(closes) < period:
        return None
    return float(closes.rolling(period).mean().iloc[-1])


def _compute_macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[Optional[float], Optional[float]]:
    """Return (macd_line, signal_line) or (None, None)."""
    if len(closes) < slow + signal:
        return None, None
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def _compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> float:
    """Average Directional Index (simplified Wilder)."""
    if len(close) < period * 2:
        return 0.0
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()
    val = adx.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _technical_score(
    rsi: float,
    above_200sma: bool,
    macd_bull: bool,
    adx: float,
) -> float:
    """Score 0-100 based on lightweight technicals."""
    score = 0.0

    # RSI contribution (25 pts) -- reward 40-60 neutral zone and slightly oversold
    if 40 <= rsi <= 60:
        score += 25.0
    elif 30 <= rsi < 40:
        score += 20.0  # potential bounce
    elif 60 < rsi <= 70:
        score += 15.0
    elif rsi < 30:
        score += 10.0  # deep oversold, risky
    else:
        score += 5.0  # overbought

    # Trend (SMA200) contribution (30 pts)
    if above_200sma:
        score += 30.0
    else:
        score += 5.0

    # MACD contribution (25 pts)
    if macd_bull:
        score += 25.0
    else:
        score += 5.0

    # ADX contribution (20 pts) -- stronger trend is better
    if adx >= 25:
        score += 20.0
    elif adx >= 15:
        score += 12.0
    else:
        score += 5.0

    return min(score, 100.0)


def _fundamental_score(
    revenue_growth: Optional[float],
    pe: Optional[float],
    roe: Optional[float],
    debt_to_equity: Optional[float],
    fcf_positive: bool,
) -> float:
    """Score 0-100 based on fundamental metrics."""
    score = 0.0
    items = 0

    # Revenue growth (20 pts)
    if revenue_growth is not None:
        items += 1
        if revenue_growth >= 0.20:
            score += 20.0
        elif revenue_growth >= 0.10:
            score += 15.0
        elif revenue_growth >= 0.0:
            score += 10.0
        else:
            score += 3.0

    # PE ratio (20 pts) -- lower is better (value lens)
    if pe is not None and pe > 0:
        items += 1
        if pe <= 15:
            score += 20.0
        elif pe <= 25:
            score += 15.0
        elif pe <= 40:
            score += 10.0
        else:
            score += 3.0

    # ROE (20 pts)
    if roe is not None:
        items += 1
        if roe >= 0.20:
            score += 20.0
        elif roe >= 0.12:
            score += 15.0
        elif roe >= 0.05:
            score += 10.0
        else:
            score += 3.0

    # Debt-to-equity (20 pts)
    if debt_to_equity is not None and debt_to_equity >= 0:
        items += 1
        if debt_to_equity <= 50:
            score += 20.0
        elif debt_to_equity <= 100:
            score += 15.0
        elif debt_to_equity <= 200:
            score += 8.0
        else:
            score += 2.0

    # FCF (20 pts)
    items += 1
    score += 20.0 if fcf_positive else 3.0

    # Normalize to 100 based on available items
    if items == 0:
        return 50.0  # neutral when no data
    return min(score * (5 / items), 100.0)


def _sentiment_score_default() -> float:
    """Neutral placeholder when sentiment analysis is skipped."""
    return 50.0


def _composite_score(
    tech: float,
    fund: float,
    sent: float,
) -> float:
    """Weighted composite: 40% tech + 35% fundamental + 25% sentiment."""
    return round(0.40 * tech + 0.35 * fund + 0.25 * sent, 2)


def _score_to_action(score: float) -> AdvisorAction:
    """Map composite score to an AdvisorAction."""
    if score >= 80:
        return AdvisorAction.STRONG_BUY
    if score >= 65:
        return AdvisorAction.BUY
    if score >= 45:
        return AdvisorAction.HOLD
    if score >= 30:
        return AdvisorAction.SELL
    return AdvisorAction.STRONG_SELL


def _score_to_conviction(score: float) -> int:
    """Map composite score to 0-100 conviction."""
    return max(0, min(100, int(score)))


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def _passes_filter(
    filters: ScreenerFilter,
    *,
    rsi: float,
    above_sma_200: bool,
    macd_bull: bool,
    adx: float,
    revenue_growth: Optional[float],
    pe: Optional[float],
    roe: Optional[float],
    debt_to_equity: Optional[float],
    fcf_positive: bool,
    sentiment_score: Optional[float],
    article_count: int,
) -> bool:
    """Return True if the stock passes all non-None filter criteria."""
    # Technical
    if filters.rsi_min is not None and rsi < filters.rsi_min:
        return False
    if filters.rsi_max is not None and rsi > filters.rsi_max:
        return False
    if filters.above_sma_200 is not None and above_sma_200 != filters.above_sma_200:
        return False
    if filters.macd_bullish is not None and macd_bull != filters.macd_bullish:
        return False
    if filters.adx_min is not None and adx < filters.adx_min:
        return False

    # Fundamental
    if filters.min_revenue_growth is not None:
        if revenue_growth is None or revenue_growth < filters.min_revenue_growth:
            return False
    if filters.max_pe is not None:
        if pe is None or pe <= 0 or pe > filters.max_pe:
            return False
    if filters.min_roe is not None:
        if roe is None or roe < filters.min_roe:
            return False
    if filters.max_debt_to_equity is not None:
        if debt_to_equity is not None and debt_to_equity > filters.max_debt_to_equity:
            return False
    if filters.positive_fcf is True and not fcf_positive:
        return False

    # Sentiment
    if filters.min_sentiment_score is not None:
        if sentiment_score is None or sentiment_score < filters.min_sentiment_score:
            return False
    if filters.min_article_count is not None:
        if article_count < filters.min_article_count:
            return False

    return True


# ---------------------------------------------------------------------------
# StockScreener
# ---------------------------------------------------------------------------

class StockScreener:
    """High-throughput stock screener with customizable filters.

    Uses lightweight indicator computation and batch data download for
    speed. For full deep-dive analysis on individual stocks use
    :pyclass:`InvestmentAdvisor` directly.

    Parameters
    ----------
    advisor:
        An ``InvestmentAdvisor`` instance (used for optional enhanced
        sentiment analysis on top picks, but *not* for the primary
        screening pass -- that would be too slow).
    """

    def __init__(self, advisor: InvestmentAdvisor) -> None:
        self._advisor = advisor
        self._log = logger.bind(component="StockScreener")

    # -- public helpers -----------------------------------------------------

    @staticmethod
    def get_profile(name: str) -> ScreenerFilter:
        """Return the :class:`ScreenerFilter` for a named profile.

        Raises
        ------
        KeyError
            If *name* is not a recognised profile.
        """
        key = name.lower().replace(" ", "_").replace("-", "_")
        if key not in _PROFILES:
            available = ", ".join(sorted(_PROFILES.keys()))
            raise KeyError(f"Unknown profile '{name}'. Available: {available}")
        return _PROFILES[key].model_copy()

    @staticmethod
    def list_profiles() -> list[str]:
        """Return a sorted list of available profile names."""
        return sorted(_PROFILES.keys())

    # -- primary screening entry point --------------------------------------

    async def screen(
        self,
        tickers: list[str],
        filters: Optional[ScreenerFilter] = None,
        profile: Optional[str] = None,
    ) -> ScreenerResult:
        """Screen a universe of tickers and return ranked results.

        Parameters
        ----------
        tickers:
            List of Yahoo Finance ticker symbols to screen.
        filters:
            Explicit :class:`ScreenerFilter` to apply. Mutually exclusive
            with *profile* -- if both are given, *filters* takes precedence.
        profile:
            Name of a pre-built profile (see :meth:`list_profiles`).

        Returns
        -------
        ScreenerResult
            Ranked list of stocks that pass all filter criteria, ordered
            by composite score (descending).
        """
        active_filters = filters or (self.get_profile(profile) if profile else ScreenerFilter())
        profile_label = profile or ("custom" if filters else "none")

        self._log.info(
            "screener.start",
            universe_size=len(tickers),
            profile=profile_label,
        )

        # ------------------------------------------------------------------
        # 1. Batch download price data for all tickers
        # ------------------------------------------------------------------
        price_data = await self._fetch_price_data(tickers)

        # ------------------------------------------------------------------
        # 2. Fetch fundamental info per-ticker (async via thread pool)
        # ------------------------------------------------------------------
        info_data = await self._fetch_fundamentals(tickers)

        # ------------------------------------------------------------------
        # 3. Score, filter, rank
        # ------------------------------------------------------------------
        items: list[ScreenerResultItem] = []

        for ticker in tickers:
            try:
                ohlcv = price_data.get(ticker)
                info = info_data.get(ticker, {})
                if ohlcv is None or ohlcv.empty:
                    self._log.debug("screener.skip_no_data", ticker=ticker)
                    continue

                closes: pd.Series = ohlcv["Close"]
                highs: pd.Series = ohlcv["High"]
                lows: pd.Series = ohlcv["Low"]

                # --- Technical indicators --------------------------------
                rsi = _compute_rsi(closes)
                sma200 = _compute_sma(closes, 200)
                last_close = float(closes.iloc[-1])
                above_sma_200 = (sma200 is not None) and (last_close > sma200)
                macd_line, macd_sig = _compute_macd(closes)
                macd_bull = (
                    macd_line is not None
                    and macd_sig is not None
                    and macd_line > macd_sig
                )
                adx = _compute_adx(highs, lows, closes)

                # --- Fundamental metrics ---------------------------------
                revenue_growth = info.get("revenueGrowth")
                pe = info.get("trailingPE") or info.get("forwardPE")
                roe = info.get("returnOnEquity")
                debt_to_equity = info.get("debtToEquity")
                fcf_raw = info.get("freeCashflow")
                fcf_positive = (fcf_raw is not None and fcf_raw > 0)
                company_name = info.get("shortName") or info.get("longName") or ticker

                # --- Filter gate -----------------------------------------
                if not _passes_filter(
                    active_filters,
                    rsi=rsi,
                    above_sma_200=above_sma_200,
                    macd_bull=macd_bull,
                    adx=adx,
                    revenue_growth=revenue_growth,
                    pe=pe,
                    roe=roe,
                    debt_to_equity=debt_to_equity,
                    fcf_positive=fcf_positive,
                    sentiment_score=None,
                    article_count=0,
                ):
                    continue

                # --- Scoring ---------------------------------------------
                tech = _technical_score(rsi, above_sma_200, macd_bull, adx)
                fund = _fundamental_score(revenue_growth, pe, roe, debt_to_equity, fcf_positive)
                sent = _sentiment_score_default()
                composite = _composite_score(tech, fund, sent)
                action = _score_to_action(composite)
                conviction = _score_to_conviction(composite)

                # Key factors for human readability
                key_factors = self._build_key_factors(
                    rsi=rsi,
                    above_sma_200=above_sma_200,
                    macd_bull=macd_bull,
                    adx=adx,
                    revenue_growth=revenue_growth,
                    pe=pe,
                    roe=roe,
                    fcf_positive=fcf_positive,
                )

                items.append(
                    ScreenerResultItem(
                        ticker=ticker,
                        name=company_name,
                        current_price=last_close,
                        composite_score=composite,
                        technical_score=round(tech, 2),
                        fundamental_score=round(fund, 2),
                        sentiment_score=round(sent, 2),
                        action=action,
                        conviction=conviction,
                        key_factors=key_factors,
                    )
                )
            except Exception:
                self._log.warning("screener.ticker_error", ticker=ticker, exc_info=True)

        # Sort descending by composite score
        items.sort(key=lambda it: it.composite_score, reverse=True)

        result = ScreenerResult(
            universe_name=profile_label,
            filter_profile=profile_label,
            total_screened=len(tickers),
            results=items,
            timestamp=datetime.now(tz=timezone.utc),
        )
        self._log.info(
            "screener.done",
            screened=len(tickers),
            passed=len(items),
            profile=profile_label,
        )
        return result

    # -- private helpers ----------------------------------------------------

    async def _fetch_price_data(
        self,
        tickers: list[str],
        period: str = "1y",
    ) -> dict[str, pd.DataFrame]:
        """Batch-download OHLCV via yfinance.download for speed."""
        if not tickers:
            return {}

        def _download() -> dict[str, pd.DataFrame]:
            try:
                raw = yf.download(
                    tickers,
                    period=period,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )
            except Exception:
                self._log.warning("screener.batch_download_failed", exc_info=True)
                return {}

            result: dict[str, pd.DataFrame] = {}
            if len(tickers) == 1:
                # Single ticker: raw is already a flat DataFrame
                t = tickers[0]
                if not raw.empty:
                    result[t] = raw
            else:
                for t in tickers:
                    try:
                        df = raw[t].dropna(how="all")
                        if not df.empty:
                            result[t] = df
                    except (KeyError, AttributeError):
                        pass
            return result

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _download)

    async def _fetch_fundamentals(
        self,
        tickers: list[str],
    ) -> dict[str, dict]:
        """Fetch yfinance .info for each ticker concurrently."""

        async def _single(ticker: str) -> tuple[str, dict]:
            def _get() -> dict:
                try:
                    return yf.Ticker(ticker).info or {}
                except Exception:
                    self._log.debug(
                        "screener.info_failed",
                        ticker=ticker,
                        exc_info=True,
                    )
                    return {}

            loop = asyncio.get_running_loop()
            info = await loop.run_in_executor(None, _get)
            return ticker, info

        tasks = [_single(t) for t in tickers]
        pairs = await asyncio.gather(*tasks, return_exceptions=True)

        data: dict[str, dict] = {}
        for item in pairs:
            if isinstance(item, tuple):
                data[item[0]] = item[1]
        return data

    @staticmethod
    def _build_key_factors(
        *,
        rsi: float,
        above_sma_200: bool,
        macd_bull: bool,
        adx: float,
        revenue_growth: Optional[float],
        pe: Optional[float],
        roe: Optional[float],
        fcf_positive: bool,
    ) -> list[str]:
        """Build human-readable list of key factors for a screened stock."""
        factors: list[str] = []

        # Technical
        if rsi < 30:
            factors.append(f"RSI deeply oversold ({rsi:.1f})")
        elif rsi < 40:
            factors.append(f"RSI approaching oversold ({rsi:.1f})")
        elif rsi > 70:
            factors.append(f"RSI overbought ({rsi:.1f})")

        if above_sma_200:
            factors.append("Trading above 200-SMA (uptrend)")
        else:
            factors.append("Trading below 200-SMA (downtrend)")

        if macd_bull:
            factors.append("MACD bullish crossover")

        if adx >= 25:
            factors.append(f"Strong trend (ADX {adx:.1f})")

        # Fundamental
        if revenue_growth is not None:
            factors.append(f"Revenue growth {revenue_growth:+.1%}")
        if pe is not None and pe > 0:
            factors.append(f"P/E {pe:.1f}")
        if roe is not None:
            factors.append(f"ROE {roe:.1%}")
        if fcf_positive:
            factors.append("Positive free cash flow")

        return factors
