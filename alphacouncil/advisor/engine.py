"""Investment Advisor engine — produces actionable stock recommendations.

Reuses existing TechnicalEngine, FundamentalEngine, and SentimentEngine
from alphacouncil.analysis, combining their outputs into scored
recommendations with entry/exit levels and human-readable reasoning.

Works for both Indian (.NS) and global tickers via yfinance.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import structlog
import yfinance as yf

from alphacouncil.advisor.models import (
    AdvisorAction,
    EntryExitLevels,
    FundamentalVerdict,
    HorizonRating,
    InvestmentHorizon,
    RiskAssessment,
    SentimentVerdict,
    StockRecommendation,
    TechnicalVerdict,
    ValuationVerdict,
)
from alphacouncil.core.cache_manager import TieredCache
from alphacouncil.core.models import FundamentalSignal, SentimentSignal, TechnicalSignal

logger = structlog.get_logger(__name__)


def _is_indian_ticker(ticker: str) -> bool:
    """Check if a ticker is an Indian stock (.NS or .BO suffix)."""
    return ticker.endswith(".NS") or ticker.endswith(".BO")


def _safe(val: Optional[float], default: float = 0.0) -> float:
    """Return val if it's a finite number, else default."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return default
    return float(val)


class InvestmentAdvisor:
    """Produces scored stock recommendations combining technical, fundamental, and sentiment analysis.

    Works independently of any broker. Uses only yfinance (free) and RSS feeds.
    Supports Indian (.NS) and global tickers.
    """

    def __init__(self, cache: Optional[TieredCache] = None) -> None:
        self._cache = cache or TieredCache()
        # Lazy-load engines to avoid heavy imports at init
        self._tech_engine = None
        self._fund_engine = None
        self._sent_engine = None

    def _get_tech_engine(self):
        if self._tech_engine is None:
            from alphacouncil.analysis.technical import TechnicalEngine
            self._tech_engine = TechnicalEngine(self._cache)
        return self._tech_engine

    def _get_fund_engine(self):
        if self._fund_engine is None:
            from alphacouncil.analysis.fundamental import FundamentalEngine
            self._fund_engine = FundamentalEngine(self._cache)
        return self._fund_engine

    def _get_sent_engine(self):
        if self._sent_engine is None:
            from alphacouncil.analysis.sentiment import SentimentEngine
            self._sent_engine = SentimentEngine(self._cache)
        return self._sent_engine

    async def analyze(self, ticker: str) -> StockRecommendation:
        """Produce a complete recommendation for a single stock.

        Parameters
        ----------
        ticker:
            Any yfinance-compatible ticker. Indian stocks should have .NS suffix.
            US stocks use plain symbols (AAPL, MSFT). European stocks use exchange suffix (.L, .SW).
            Bare Indian tickers (e.g. MOTHERSON) are auto-resolved to .NS.
        """
        # Auto-resolve bare tickers: if no dot and not a known US ticker, try .NS
        ticker = ticker.strip().upper()
        if "." not in ticker:
            resolved = await asyncio.to_thread(self._resolve_ticker, ticker)
            if resolved != ticker:
                logger.info("ticker_resolved", original=ticker, resolved=resolved)
            ticker = resolved

        logger.info("analyzing_stock", ticker=ticker)
        now = datetime.now(timezone.utc)

        # Fetch price data via yfinance
        df = await asyncio.to_thread(self._fetch_ohlcv, ticker)
        if df is None or df.empty or len(df) < 20:
            raise ValueError(f"Insufficient price data for {ticker}")

        current_price = float(df["Close"].iloc[-1])
        currency = "INR" if _is_indian_ticker(ticker) else "USD"

        # Get company info
        info = await asyncio.to_thread(self._fetch_info, ticker)
        name = info.get("shortName") or info.get("longName") or ticker.split(".")[0]
        exchange_name = info.get("exchange", "NSE" if _is_indian_ticker(ticker) else "NASDAQ")

        # Run analysis engines concurrently
        tech_signal, fund_signal, sent_signal = await asyncio.gather(
            self._run_technical(ticker, df),
            self._run_fundamental(ticker, info),
            self._run_sentiment(ticker),
            return_exceptions=True,
        )

        # Handle any engine failures gracefully
        if isinstance(tech_signal, Exception):
            logger.warning("technical_analysis_failed", ticker=ticker, error=str(tech_signal))
            tech_signal = None
        if isinstance(fund_signal, Exception):
            logger.warning("fundamental_analysis_failed", ticker=ticker, error=str(fund_signal))
            fund_signal = None
        if isinstance(sent_signal, Exception):
            logger.warning("sentiment_analysis_failed", ticker=ticker, error=str(sent_signal))
            sent_signal = None

        # Build verdicts
        tech_verdict = self._build_technical_verdict(tech_signal, df, current_price)
        fund_verdict = self._build_fundamental_verdict(fund_signal, current_price, info)
        sent_verdict = self._build_sentiment_verdict(sent_signal)
        risk_assessment = self._build_risk_assessment(tech_signal, df, current_price)

        # Score and decide
        tech_score = self._score_technical(tech_verdict)
        fund_score = self._score_fundamental(fund_verdict)
        sent_score = self._score_sentiment(sent_verdict)

        # Weighted composite: technical 40%, fundamental 35%, sentiment 25%
        composite = 0.40 * tech_score + 0.35 * fund_score + 0.25 * sent_score

        action = self._composite_to_action(composite)
        horizon = self._determine_horizon(tech_verdict, fund_verdict)
        conviction = max(0, min(100, int(composite)))

        # Entry/exit levels
        levels = self._compute_levels(current_price, tech_signal, df, action)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            ticker, name, action, conviction, tech_verdict, fund_verdict, sent_verdict, risk_assessment, current_price
        )

        # Compute per-horizon buy ratings
        horizon_ratings = self._compute_horizon_ratings(
            tech_score, fund_score, sent_score, tech_verdict, fund_verdict,
            levels, current_price,
        )

        return StockRecommendation(
            ticker=ticker,
            name=name,
            exchange=exchange_name,
            current_price=current_price,
            currency=currency,
            action=action,
            horizon=horizon,
            conviction=conviction,
            technical=tech_verdict,
            fundamental=fund_verdict,
            sentiment=sent_verdict,
            risk=risk_assessment,
            levels=levels,
            horizon_ratings=horizon_ratings,
            reasoning=reasoning,
            timestamp=now,
        )

    async def analyze_batch(self, tickers: list[str], max_concurrent: int = 5) -> list[StockRecommendation]:
        """Analyze multiple stocks with concurrency limit."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _analyze_one(t: str) -> Optional[StockRecommendation]:
            async with sem:
                try:
                    return await self.analyze(t)
                except Exception as e:
                    logger.warning("batch_analysis_failed", ticker=t, error=str(e))
                    return None

        tasks = [_analyze_one(t) for t in tickers]
        raw = await asyncio.gather(*tasks)
        return [r for r in raw if r is not None]

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_ohlcv(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch OHLCV data via yfinance. Auto-resolves bare Indian tickers."""
        # Try the ticker as-is first
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning("ohlcv_fetch_failed", ticker=ticker, error=str(e))

        # If bare ticker (no exchange suffix) fails, try .NS (NSE India)
        if "." not in ticker:
            ns_ticker = f"{ticker}.NS"
            try:
                stock = yf.Ticker(ns_ticker)
                df = stock.history(period=period)
                if df is not None and not df.empty:
                    logger.info("resolved_to_ns", original=ticker, resolved=ns_ticker)
                    return df
            except Exception:
                pass

        return None

    def _fetch_info(self, ticker: str) -> dict:
        """Fetch stock info via yfinance. Auto-resolves bare tickers to .NS."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            if info.get("regularMarketPrice") or info.get("currentPrice"):
                return info
        except Exception:
            pass

        # Try .NS if bare ticker returned nothing useful
        if "." not in ticker:
            try:
                stock = yf.Ticker(f"{ticker}.NS")
                return stock.info or {}
            except Exception:
                pass

        return {}

    # ------------------------------------------------------------------
    # Ticker resolution
    # ------------------------------------------------------------------

    def _resolve_ticker(self, ticker: str) -> str:
        """Resolve a bare ticker to the correct yfinance symbol.

        For bare tickers without an exchange suffix (e.g. 'MOTHERSON'), tries:
        1. The ticker as-is (works for US stocks like AAPL, MSFT)
        2. ticker.NS (NSE India)
        3. ticker.BO (BSE India)
        Returns the first one that has valid price data.
        """
        # If it already has a suffix, return as-is
        if "." in ticker:
            return ticker

        # Try as-is first (US stocks)
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            if info.get("regularMarketPrice") or info.get("currentPrice"):
                return ticker
        except Exception:
            pass

        # Try .NS (NSE)
        try:
            stock = yf.Ticker(f"{ticker}.NS")
            hist = stock.history(period="5d")
            if hist is not None and not hist.empty:
                return f"{ticker}.NS"
        except Exception:
            pass

        # Try .BO (BSE)
        try:
            stock = yf.Ticker(f"{ticker}.BO")
            hist = stock.history(period="5d")
            if hist is not None and not hist.empty:
                return f"{ticker}.BO"
        except Exception:
            pass

        # Fallback: return with .NS (most common for Indian stocks)
        return f"{ticker}.NS"

    # ------------------------------------------------------------------
    # Engine wrappers (handle Indian vs global)
    # ------------------------------------------------------------------

    async def _run_technical(self, ticker: str, df: pd.DataFrame) -> TechnicalSignal:
        """Run technical analysis via existing TechnicalEngine."""
        engine = self._get_tech_engine()
        return await engine.analyze(ticker, df)

    async def _run_fundamental(self, ticker: str, info: dict) -> Optional[FundamentalSignal]:
        """Run fundamental analysis. Handles global tickers gracefully."""
        engine = self._get_fund_engine()
        try:
            # FundamentalEngine auto-appends .NS, which is wrong for global tickers.
            # For Indian tickers, let the engine work normally.
            if _is_indian_ticker(ticker):
                return await engine.analyze(ticker)
            else:
                # For global tickers, build a FundamentalSignal directly from yfinance info
                return self._build_fundamental_from_info(ticker, info)
        except Exception as e:
            logger.warning("fundamental_failed", ticker=ticker, error=str(e))
            return None

    async def _run_sentiment(self, ticker: str) -> Optional[SentimentSignal]:
        """Run sentiment analysis."""
        engine = self._get_sent_engine()
        try:
            return await engine.get_ticker_sentiment(ticker)
        except Exception:
            # Sentiment may not be available for all tickers
            return None

    def _build_fundamental_from_info(self, ticker: str, info: dict) -> FundamentalSignal:
        """Build FundamentalSignal directly from yfinance info dict for global tickers."""
        now = datetime.now(timezone.utc)

        pe = _safe(info.get("trailingPE"))
        peg = _safe(info.get("pegRatio"))
        pb = _safe(info.get("priceToBook"))
        roe = _safe(info.get("returnOnEquity"))
        roa = _safe(info.get("returnOnAssets"))
        de = _safe(info.get("debtToEquity"))
        fcf = _safe(info.get("freeCashflow"))
        gm = _safe(info.get("grossMargins"))
        om = _safe(info.get("operatingMargins"))
        nm = _safe(info.get("profitMargins"))
        rg = _safe(info.get("revenueGrowth"))
        eg = _safe(info.get("earningsGrowth"))

        # Simple intrinsic value estimate
        price = _safe(info.get("currentPrice"), _safe(info.get("regularMarketPrice")))
        iv = price * 1.0  # default to current price
        if pe > 0 and eg > 0:
            fair_pe = min(eg * 100, 30)  # PEG = 1 fair value
            iv = (price / pe) * fair_pe if pe > 0 else price

        return FundamentalSignal(
            ticker=ticker,
            pe_ratio=pe, peg_ratio=peg, pb_ratio=pb,
            roe=roe, roa=roa, debt_to_equity=de, fcf=fcf,
            gross_margin=gm, operating_margin=om, net_margin=nm,
            revenue_growth=rg, eps_growth=eg,
            promoter_holding=0.0, fii_holding=0.0, dii_holding=0.0,
            intrinsic_value=iv,
            timestamp=now,
        )

    # ------------------------------------------------------------------
    # Verdict builders
    # ------------------------------------------------------------------

    def _build_technical_verdict(
        self, signal: Optional[TechnicalSignal], df: pd.DataFrame, price: float,
    ) -> TechnicalVerdict:
        if signal is None:
            return TechnicalVerdict(
                trend="NEUTRAL", rsi=50.0, rsi_signal="NEUTRAL",
                macd_signal="NEUTRAL", ma_alignment="MIXED",
                adx=0.0, adx_signal="NO_TREND",
                support=price * 0.95, resistance=price * 1.05,
                atr=price * 0.02, volume_signal="NORMAL",
                breakout=False, summary="Technical data unavailable.",
            )

        # RSI signal
        rsi = _safe(signal.rsi, 50.0)
        if rsi < 30:
            rsi_sig = "OVERSOLD"
        elif rsi > 70:
            rsi_sig = "OVERBOUGHT"
        else:
            rsi_sig = "NEUTRAL"

        # MACD signal
        macd_hist = _safe(signal.macd_hist)
        macd_sig = "BULLISH" if macd_hist > 0 else ("BEARISH" if macd_hist < 0 else "NEUTRAL")

        # MA alignment
        sma_20 = _safe(signal.sma_20, price)
        sma_50 = _safe(signal.sma_50, price)
        sma_200 = _safe(signal.sma_200, price)
        if price > sma_20 > sma_50 > sma_200:
            ma_align = "BULLISH"
        elif price < sma_20 < sma_50 < sma_200:
            ma_align = "BEARISH"
        else:
            ma_align = "MIXED"

        # ADX signal
        adx = _safe(signal.adx, 0.0)
        if adx > 25:
            adx_sig = "STRONG_TREND"
        elif adx > 15:
            adx_sig = "WEAK_TREND"
        else:
            adx_sig = "NO_TREND"

        # Trend
        if ma_align == "BULLISH" and macd_sig == "BULLISH":
            trend = "BULLISH"
        elif ma_align == "BEARISH" and macd_sig == "BEARISH":
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        # Support/resistance from Bollinger Bands
        support = _safe(signal.bollinger_lower, price * 0.95)
        resistance = _safe(signal.bollinger_upper, price * 1.05)

        # Also check SMA levels as support/resistance
        if sma_50 < price:
            support = max(support, sma_50)
        if sma_200 < price:
            support = max(_safe(signal.sma_200, support), support)

        atr = _safe(signal.atr, price * 0.02)

        # Volume
        vr = _safe(signal.volume_ratio, 1.0)
        vol_sig = "HIGH" if vr > 1.5 else ("LOW" if vr < 0.5 else "NORMAL")

        # Breakout detection
        try:
            tech_engine = self._get_tech_engine()
            breakout_info = tech_engine.detect_breakout(df)
            is_breakout = breakout_info.get("breakout", False)
        except Exception:
            is_breakout = False

        # Summary
        parts = []
        parts.append(f"Trend is {trend.lower()} with RSI at {rsi:.0f} ({rsi_sig.lower()}).")
        if macd_sig == "BULLISH":
            parts.append("MACD shows bullish momentum.")
        elif macd_sig == "BEARISH":
            parts.append("MACD shows bearish momentum.")
        if is_breakout:
            parts.append("Price is breaking out above resistance.")

        return TechnicalVerdict(
            trend=trend, rsi=rsi, rsi_signal=rsi_sig,
            macd_signal=macd_sig, ma_alignment=ma_align,
            adx=adx, adx_signal=adx_sig,
            support=round(support, 2), resistance=round(resistance, 2),
            atr=round(atr, 2), volume_signal=vol_sig,
            breakout=is_breakout,
            summary=" ".join(parts),
        )

    def _build_fundamental_verdict(
        self, signal: Optional[FundamentalSignal], price: float, info: dict,
    ) -> FundamentalVerdict:
        if signal is None:
            return FundamentalVerdict(
                valuation=ValuationVerdict.FAIR_VALUE,
                current_price=price,
                financial_health="UNKNOWN",
                summary="Fundamental data unavailable.",
            )

        # Valuation verdict
        iv = _safe(signal.intrinsic_value, price)
        margin_of_safety = (iv - price) / iv if iv > 0 else 0.0

        if margin_of_safety > 0.20:
            valuation = ValuationVerdict.UNDERVALUED
        elif margin_of_safety < -0.20:
            valuation = ValuationVerdict.OVERVALUED
        else:
            valuation = ValuationVerdict.FAIR_VALUE

        # Growth quality
        try:
            engine = self._get_fund_engine()
            gq_score = engine.growth_quality_score(signal)
        except Exception:
            gq_score = 50.0

        # Financial health
        de = _safe(signal.debt_to_equity, 0.0)
        roe = _safe(signal.roe, 0.0)
        fcf = _safe(signal.fcf, 0.0)

        if de < 50 and roe > 0.15 and fcf > 0:
            health = "STRONG"
        elif de < 100 and roe > 0.05:
            health = "MODERATE"
        elif de > 200 or roe < 0:
            health = "WEAK"
        else:
            health = "MODERATE"

        # Summary
        parts = []
        if valuation == ValuationVerdict.UNDERVALUED:
            parts.append(f"Stock appears undervalued with {margin_of_safety:.0%} margin of safety.")
        elif valuation == ValuationVerdict.OVERVALUED:
            parts.append(f"Stock appears overvalued, trading {abs(margin_of_safety):.0%} above intrinsic value.")
        else:
            parts.append("Stock is trading near fair value.")

        rg = _safe(signal.revenue_growth)
        if rg > 0.15:
            parts.append(f"Strong revenue growth of {rg:.0%}.")
        elif rg > 0:
            parts.append(f"Modest revenue growth of {rg:.0%}.")

        return FundamentalVerdict(
            valuation=valuation,
            pe_ratio=_safe(signal.pe_ratio) or None,
            peg_ratio=_safe(signal.peg_ratio) or None,
            pb_ratio=_safe(signal.pb_ratio) or None,
            roe=_safe(signal.roe) or None,
            debt_to_equity=_safe(signal.debt_to_equity) or None,
            revenue_growth=_safe(signal.revenue_growth) or None,
            eps_growth=_safe(signal.eps_growth) or None,
            fcf_positive=_safe(signal.fcf) > 0,
            growth_quality_score=gq_score,
            intrinsic_value=round(iv, 2) if iv > 0 else None,
            current_price=price,
            margin_of_safety=round(margin_of_safety, 4) if iv > 0 else None,
            financial_health=health,
            summary=" ".join(parts),
        )

    def _build_sentiment_verdict(self, signal: Optional[SentimentSignal]) -> SentimentVerdict:
        if signal is None:
            return SentimentVerdict(
                score=0.0, signal="NEUTRAL", article_count=0,
                social_buzz="NORMAL", trend=0.0, summary="No sentiment data available.",
            )

        score = _safe(signal.score, 0.0)
        if score > 0.3:
            sig = "BULLISH"
        elif score < -0.3:
            sig = "BEARISH"
        else:
            sig = "NEUTRAL"

        vol = signal.volume if signal.volume else 0
        buzz = "HIGH" if vol > 10 else ("LOW" if vol < 2 else "NORMAL")

        parts = []
        if sig == "BULLISH":
            parts.append(f"Sentiment is positive ({score:.2f}) based on {vol} articles.")
        elif sig == "BEARISH":
            parts.append(f"Sentiment is negative ({score:.2f}) based on {vol} articles.")
        else:
            parts.append(f"Sentiment is neutral ({score:.2f}).")

        if signal.keywords:
            parts.append(f"Key themes: {', '.join(signal.keywords[:3])}.")

        return SentimentVerdict(
            score=score, signal=sig, article_count=vol,
            social_buzz=buzz, trend=_safe(signal.trend),
            top_keywords=list(signal.keywords[:5]) if signal.keywords else [],
            recent_headlines=[],
            summary=" ".join(parts),
        )

    def _build_risk_assessment(
        self, signal: Optional[TechnicalSignal], df: pd.DataFrame, price: float,
    ) -> RiskAssessment:
        atr = _safe(signal.atr, price * 0.02) if signal else price * 0.02
        atr_pct = atr / price if price > 0 else 0.02

        # Volatility regime from ATR%
        if atr_pct < 0.015:
            vol_regime = "LOW"
        elif atr_pct < 0.03:
            vol_regime = "MEDIUM"
        else:
            vol_regime = "HIGH"

        # Beta estimation (vs market)
        try:
            close = df["Close"] if "Close" in df.columns else df["close"]
            returns = close.pct_change().dropna()
            vol_20 = float(returns.tail(20).std() * np.sqrt(252))
            beta = vol_20 / 0.15  # assume market vol ~15%
        except Exception:
            beta = 1.0

        # Max expected drawdown (rough: 3x ATR)
        max_dd = min(3 * atr_pct, 0.50)

        # Risk level
        if atr_pct < 0.015 and beta < 0.8:
            risk_level = "LOW"
        elif atr_pct < 0.025 and beta < 1.2:
            risk_level = "MODERATE"
        elif atr_pct < 0.04:
            risk_level = "HIGH"
        else:
            risk_level = "VERY_HIGH"

        # Risk-reward (simplified)
        rr = (2 * atr) / atr if atr > 0 else 1.0  # target 2x ATR, risk 1x ATR

        summary = f"Volatility is {vol_regime.lower()} (ATR {atr_pct:.1%}). Beta: {beta:.2f}. Max expected drawdown: {max_dd:.0%}."

        return RiskAssessment(
            volatility_regime=vol_regime,
            atr_pct=round(atr_pct, 4),
            beta=round(beta, 2),
            max_expected_drawdown=round(max_dd, 4),
            risk_reward_ratio=round(rr, 2),
            risk_level=risk_level,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_technical(self, verdict: TechnicalVerdict) -> float:
        """Score technical verdict 0-100."""
        score = 50.0
        # Trend
        if verdict.trend == "BULLISH":
            score += 15
        elif verdict.trend == "BEARISH":
            score -= 15

        # RSI
        if verdict.rsi < 30:
            score += 10  # oversold = buying opp
        elif verdict.rsi > 70:
            score -= 10  # overbought

        # MACD
        if verdict.macd_signal == "BULLISH":
            score += 10
        elif verdict.macd_signal == "BEARISH":
            score -= 10

        # MA alignment
        if verdict.ma_alignment == "BULLISH":
            score += 10
        elif verdict.ma_alignment == "BEARISH":
            score -= 10

        # ADX (trend strength amplifies)
        if verdict.adx_signal == "STRONG_TREND":
            if verdict.trend == "BULLISH":
                score += 5
            elif verdict.trend == "BEARISH":
                score -= 5

        # Breakout bonus
        if verdict.breakout:
            score += 10

        # Volume confirmation
        if verdict.volume_signal == "HIGH" and verdict.trend == "BULLISH":
            score += 5

        return max(0, min(100, score))

    def _score_fundamental(self, verdict: FundamentalVerdict) -> float:
        """Score fundamental verdict 0-100."""
        score = 50.0

        if verdict.valuation == ValuationVerdict.UNDERVALUED:
            score += 15
        elif verdict.valuation == ValuationVerdict.OVERVALUED:
            score -= 15

        # Growth quality (0-100 already)
        gq = verdict.growth_quality_score
        score += (gq - 50) * 0.3  # map 0-100 to +/-15

        # Financial health
        if verdict.financial_health == "STRONG":
            score += 10
        elif verdict.financial_health == "WEAK":
            score -= 10

        # Revenue growth
        rg = verdict.revenue_growth or 0
        if rg > 0.20:
            score += 10
        elif rg > 0.10:
            score += 5
        elif rg < 0:
            score -= 5

        # FCF
        if verdict.fcf_positive:
            score += 5
        else:
            score -= 5

        return max(0, min(100, score))

    def _score_sentiment(self, verdict: SentimentVerdict) -> float:
        """Score sentiment verdict 0-100."""
        score = 50.0
        # Score: -1 to +1 mapped to +/-25
        score += verdict.score * 25
        # Trend momentum
        score += verdict.trend * 10
        # Volume buzz
        if verdict.social_buzz == "HIGH" and verdict.score > 0:
            score += 5
        elif verdict.social_buzz == "HIGH" and verdict.score < 0:
            score -= 5
        return max(0, min(100, score))

    def _composite_to_action(self, composite: float) -> AdvisorAction:
        if composite >= 75:
            return AdvisorAction.STRONG_BUY
        elif composite >= 60:
            return AdvisorAction.BUY
        elif composite >= 40:
            return AdvisorAction.HOLD
        elif composite >= 25:
            return AdvisorAction.SELL
        else:
            return AdvisorAction.STRONG_SELL

    def _determine_horizon(
        self, tech: TechnicalVerdict, fund: FundamentalVerdict,
    ) -> InvestmentHorizon:
        # Strong technicals + weak fundamentals = short term
        # Weak technicals + strong fundamentals = long term
        if tech.trend == "BULLISH" and tech.adx_signal == "STRONG_TREND":
            if fund.valuation == ValuationVerdict.UNDERVALUED:
                return InvestmentHorizon.MID_TERM
            return InvestmentHorizon.SHORT_TERM
        if fund.valuation == ValuationVerdict.UNDERVALUED and fund.growth_quality_score > 70:
            return InvestmentHorizon.LONG_TERM
        return InvestmentHorizon.MID_TERM

    # ------------------------------------------------------------------
    # Entry/exit levels
    # ------------------------------------------------------------------

    def _compute_levels(
        self, price: float, signal: Optional[TechnicalSignal], df: pd.DataFrame, action: AdvisorAction,
    ) -> EntryExitLevels:
        atr = _safe(signal.atr, price * 0.02) if signal else price * 0.02
        sma_20 = _safe(signal.sma_20, price) if signal else price
        sma_50 = _safe(signal.sma_50, price) if signal else price
        bb_lower = _safe(signal.bollinger_lower, price * 0.95) if signal else price * 0.95

        if action in (AdvisorAction.STRONG_BUY, AdvisorAction.BUY):
            entry_low = max(bb_lower, price - atr)
            entry_high = price + atr * 0.3
            stop = price - 2.5 * atr
            target_st = price + 2 * atr
            target_mt = price * 1.15
            target_lt = price * 1.30
        elif action in (AdvisorAction.SELL, AdvisorAction.STRONG_SELL):
            entry_low = price - atr * 0.3
            entry_high = price
            stop = price + 2.5 * atr
            target_st = price - 2 * atr
            target_mt = price * 0.85
            target_lt = price * 0.70
        else:  # HOLD
            entry_low = price - atr
            entry_high = price + atr
            stop = price - 3 * atr
            target_st = price + 1.5 * atr
            target_mt = price * 1.10
            target_lt = price * 1.20

        def _rr(target: float, stop_: float) -> Optional[float]:
            risk = abs(price - stop_)
            reward = abs(target - price)
            return round(reward / risk, 2) if risk > 0 else None

        return EntryExitLevels(
            entry_zone_low=round(entry_low, 2),
            entry_zone_high=round(entry_high, 2),
            stop_loss=round(stop, 2),
            target_short_term=round(target_st, 2),
            target_mid_term=round(target_mt, 2),
            target_long_term=round(target_lt, 2),
            risk_reward_short=_rr(target_st, stop),
            risk_reward_mid=_rr(target_mt, stop),
            risk_reward_long=_rr(target_lt, stop),
        )

    # ------------------------------------------------------------------
    # Horizon ratings
    # ------------------------------------------------------------------

    def _compute_horizon_ratings(
        self, tech_score: float, fund_score: float, sent_score: float,
        tech: TechnicalVerdict, fund: FundamentalVerdict,
        levels: EntryExitLevels, price: float,
    ) -> list[HorizonRating]:
        """Compute buy/sell ratings for short, mid, and long term."""
        ratings = []

        # Short term: heavily weighted by technical (60%) + sentiment (30%) + fundamental (10%)
        st_score = 0.60 * tech_score + 0.30 * sent_score + 0.10 * fund_score
        st_action = self._composite_to_action(st_score)
        st_target = levels.target_short_term
        st_ret = ((st_target - price) / price * 100) if st_target and price > 0 else None
        st_reason = []
        if tech.trend == "BULLISH":
            st_reason.append("Bullish technical trend")
        elif tech.trend == "BEARISH":
            st_reason.append("Bearish technical trend")
        if tech.rsi < 30:
            st_reason.append("oversold on RSI")
        elif tech.rsi > 70:
            st_reason.append("overbought on RSI")
        if tech.breakout:
            st_reason.append("breakout detected")

        ratings.append(HorizonRating(
            horizon=InvestmentHorizon.SHORT_TERM,
            action=st_action,
            conviction=max(0, min(100, int(st_score))),
            target_price=st_target,
            expected_return_pct=round(st_ret, 1) if st_ret is not None else None,
            reasoning=". ".join(st_reason) + "." if st_reason else "Neutral short-term outlook.",
        ))

        # Mid term: balanced (35% tech + 40% fundamental + 25% sentiment)
        mt_score = 0.35 * tech_score + 0.40 * fund_score + 0.25 * sent_score
        mt_action = self._composite_to_action(mt_score)
        mt_target = levels.target_mid_term
        mt_ret = ((mt_target - price) / price * 100) if mt_target and price > 0 else None
        mt_reason = []
        if fund.valuation == ValuationVerdict.UNDERVALUED:
            mt_reason.append("Undervalued on fundamentals")
        elif fund.valuation == ValuationVerdict.OVERVALUED:
            mt_reason.append("Overvalued on fundamentals")
        if fund.growth_quality_score > 70:
            mt_reason.append("strong growth quality")
        if fund.financial_health == "STRONG":
            mt_reason.append("strong financial health")

        ratings.append(HorizonRating(
            horizon=InvestmentHorizon.MID_TERM,
            action=mt_action,
            conviction=max(0, min(100, int(mt_score))),
            target_price=mt_target,
            expected_return_pct=round(mt_ret, 1) if mt_ret is not None else None,
            reasoning=". ".join(mt_reason) + "." if mt_reason else "Neutral mid-term outlook.",
        ))

        # Long term: heavily fundamental (60%) + sentiment trend (15%) + technical (25%)
        lt_score = 0.25 * tech_score + 0.60 * fund_score + 0.15 * sent_score
        lt_action = self._composite_to_action(lt_score)
        lt_target = levels.target_long_term
        lt_ret = ((lt_target - price) / price * 100) if lt_target and price > 0 else None
        lt_reason = []
        if fund.revenue_growth and fund.revenue_growth > 0.15:
            lt_reason.append(f"Strong revenue growth ({fund.revenue_growth:.0%})")
        if fund.margin_of_safety and fund.margin_of_safety > 0.2:
            lt_reason.append(f"Attractive margin of safety ({fund.margin_of_safety:.0%})")
        if fund.roe and fund.roe > 0.15:
            lt_reason.append(f"High ROE ({fund.roe:.0%})")
        if fund.fcf_positive:
            lt_reason.append("positive free cash flow")

        ratings.append(HorizonRating(
            horizon=InvestmentHorizon.LONG_TERM,
            action=lt_action,
            conviction=max(0, min(100, int(lt_score))),
            target_price=lt_target,
            expected_return_pct=round(lt_ret, 1) if lt_ret is not None else None,
            reasoning=". ".join(lt_reason) + "." if lt_reason else "Neutral long-term outlook.",
        ))

        return ratings

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    def _generate_reasoning(
        self, ticker: str, name: str, action: AdvisorAction,
        conviction: int, tech: TechnicalVerdict, fund: FundamentalVerdict,
        sent: SentimentVerdict, risk: RiskAssessment, price: float,
    ) -> str:
        parts = []

        # Opening
        action_desc = {
            AdvisorAction.STRONG_BUY: "a strong buy",
            AdvisorAction.BUY: "a buy",
            AdvisorAction.HOLD: "a hold",
            AdvisorAction.SELL: "a sell",
            AdvisorAction.STRONG_SELL: "a strong sell",
        }
        parts.append(
            f"{name} ({ticker}) is rated {action_desc[action]} with {conviction}% conviction."
        )

        # Technical
        parts.append(tech.summary)

        # Fundamental
        parts.append(fund.summary)

        # Sentiment
        if sent.score != 0:
            parts.append(sent.summary)

        # Risk warning
        if risk.risk_level in ("HIGH", "VERY_HIGH"):
            parts.append(f"Caution: volatility is {risk.volatility_regime.lower()} — use tight stops.")

        return " ".join(parts)
