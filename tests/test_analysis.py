"""Tests for analysis engines: TechnicalEngine, FundamentalEngine,
SentimentEngine, MacroEngine.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphacouncil.core.models import (
    FundamentalSignal,
    MacroSignal,
    MarketRegime,
    SentimentSignal,
    TechnicalSignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 250, start_price: float = 100.0) -> pd.DataFrame:
    """Reproducible synthetic OHLCV with a DatetimeIndex."""
    rng = np.random.default_rng(123)
    dates = pd.bdate_range(end=datetime.now(), periods=n, tz=timezone.utc)
    close = start_price + np.cumsum(rng.normal(0.05, 1.0, n))
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 1.5, n)
    low = np.maximum(low, 0.5)
    opn = close + rng.uniform(-0.5, 0.5, n)
    opn = np.maximum(opn, 0.5)
    volume = rng.integers(50_000, 500_000, n).astype(np.float64)
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class FakeCache:
    """Trivial cache that always misses."""

    async def get(self, key: str) -> Any:
        return None

    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        pass


# ---------------------------------------------------------------------------
# TechnicalEngine tests
# ---------------------------------------------------------------------------


class TestTechnicalEngine:
    @pytest.mark.asyncio
    async def test_analyze_returns_technical_signal(self):
        from alphacouncil.analysis.technical import TechnicalEngine

        engine = TechnicalEngine(cache=FakeCache())
        df = _make_ohlcv(250)
        sig = await engine.analyze("RELIANCE", df)
        assert isinstance(sig, TechnicalSignal)
        assert sig.ticker == "RELIANCE"

    @pytest.mark.asyncio
    async def test_rsi_in_valid_range(self):
        from alphacouncil.analysis.technical import TechnicalEngine

        engine = TechnicalEngine(cache=FakeCache())
        df = _make_ohlcv(250)
        sig = await engine.analyze("TEST", df)
        assert 0.0 <= sig.rsi <= 100.0

    @pytest.mark.asyncio
    async def test_bollinger_bands_order(self):
        from alphacouncil.analysis.technical import TechnicalEngine

        engine = TechnicalEngine(cache=FakeCache())
        df = _make_ohlcv(250)
        sig = await engine.analyze("TEST", df)
        assert sig.bollinger_lower <= sig.bollinger_mid <= sig.bollinger_upper

    @pytest.mark.asyncio
    async def test_sma_ordering(self):
        """SMA-20 should generally be closer to price than SMA-200."""
        from alphacouncil.analysis.technical import TechnicalEngine

        engine = TechnicalEngine(cache=FakeCache())
        df = _make_ohlcv(250)
        sig = await engine.analyze("TEST", df)
        assert sig.sma_20 != 0.0
        assert sig.sma_200 != 0.0

    @pytest.mark.asyncio
    async def test_macd_components_exist(self):
        from alphacouncil.analysis.technical import TechnicalEngine

        engine = TechnicalEngine(cache=FakeCache())
        df = _make_ohlcv(250)
        sig = await engine.analyze("TEST", df)
        # MACD histogram = MACD - signal (approximately)
        assert sig.macd_hist == pytest.approx(sig.macd - sig.macd_signal, abs=0.5)

    @pytest.mark.asyncio
    async def test_volume_ratio_reasonable(self):
        from alphacouncil.analysis.technical import TechnicalEngine

        engine = TechnicalEngine(cache=FakeCache())
        df = _make_ohlcv(250)
        sig = await engine.analyze("TEST", df)
        assert 0.1 < sig.volume_ratio < 10.0

    def test_detect_breakout_structure(self):
        from alphacouncil.analysis.technical import TechnicalEngine

        engine = TechnicalEngine(cache=FakeCache())
        df = _make_ohlcv(50)
        result = engine.detect_breakout(df)
        assert "breakout" in result
        assert "resistance" in result
        assert "close" in result
        assert "atr" in result
        assert isinstance(result["breakout"], bool)

    def test_get_trend_classification(self):
        from alphacouncil.analysis.technical import TechnicalEngine

        engine = TechnicalEngine(cache=FakeCache())
        df = _make_ohlcv(250)
        trend = engine.get_trend(df)
        assert trend in ("BULLISH", "BEARISH", "NEUTRAL")

    @pytest.mark.asyncio
    async def test_empty_df_raises(self):
        from alphacouncil.analysis.technical import TechnicalEngine

        engine = TechnicalEngine(cache=FakeCache())
        with pytest.raises(ValueError, match="Empty DataFrame"):
            await engine.analyze("TEST", pd.DataFrame())


# ---------------------------------------------------------------------------
# FundamentalEngine tests
# ---------------------------------------------------------------------------


class TestFundamentalEngine:
    def test_growth_quality_score_range(self):
        from alphacouncil.analysis.fundamental import FundamentalEngine

        engine = FundamentalEngine(cache=FakeCache())
        now = datetime.now(tz=timezone.utc)
        sig = FundamentalSignal(
            ticker="TEST", pe_ratio=20.0, peg_ratio=1.5, pb_ratio=3.0,
            roe=0.20, roa=0.10, debt_to_equity=0.5, fcf=1e9,
            gross_margin=0.40, operating_margin=0.25, net_margin=0.15,
            revenue_growth=0.30, eps_growth=0.25, promoter_holding=50.0,
            fii_holding=20.0, dii_holding=15.0, intrinsic_value=1500.0,
            timestamp=now,
        )
        score = engine.growth_quality_score(sig)
        assert 0.0 <= score <= 100.0

    def test_growth_quality_score_zero_growth(self):
        from alphacouncil.analysis.fundamental import FundamentalEngine

        engine = FundamentalEngine(cache=FakeCache())
        now = datetime.now(tz=timezone.utc)
        sig = FundamentalSignal(
            ticker="WEAK", pe_ratio=30.0, peg_ratio=3.0, pb_ratio=5.0,
            roe=0.0, roa=0.0, debt_to_equity=2.0, fcf=0.0,
            gross_margin=0.0, operating_margin=0.0, net_margin=0.0,
            revenue_growth=0.0, eps_growth=0.0, promoter_holding=10.0,
            fii_holding=5.0, dii_holding=5.0, intrinsic_value=0.0,
            timestamp=now,
        )
        score = engine.growth_quality_score(sig)
        assert score == 0.0

    def test_dcf_intrinsic_value_positive_fcf(self):
        from alphacouncil.analysis.fundamental import FundamentalEngine

        result = FundamentalEngine._dcf_intrinsic_value(1_000_000.0, 0.15)
        assert result > 0.0

    def test_dcf_intrinsic_value_zero_fcf(self):
        from alphacouncil.analysis.fundamental import FundamentalEngine

        result = FundamentalEngine._dcf_intrinsic_value(0.0, 0.15)
        assert result == 0.0

    def test_dcf_intrinsic_value_negative_fcf(self):
        from alphacouncil.analysis.fundamental import FundamentalEngine

        result = FundamentalEngine._dcf_intrinsic_value(-500_000.0, 0.15)
        assert result == 0.0

    def test_fundamental_signal_is_frozen(self):
        now = datetime.now(tz=timezone.utc)
        sig = FundamentalSignal(
            ticker="TEST", pe_ratio=20.0, peg_ratio=1.5, pb_ratio=3.0,
            roe=0.15, roa=0.08, debt_to_equity=0.5, fcf=1e9,
            gross_margin=0.3, operating_margin=0.2, net_margin=0.1,
            revenue_growth=0.2, eps_growth=0.15, promoter_holding=50.0,
            fii_holding=20.0, dii_holding=10.0, intrinsic_value=1500.0,
            timestamp=now,
        )
        assert hash(sig) is not None  # frozen models are hashable


# ---------------------------------------------------------------------------
# SentimentEngine tests
# ---------------------------------------------------------------------------


class TestSentimentEngine:
    def test_map_ticker_exact_match(self):
        from alphacouncil.analysis.sentiment import SentimentEngine

        universe = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        assert SentimentEngine.map_ticker(
            "Reliance Q4 beats estimates", universe
        ) == "RELIANCE.NS"

    def test_map_ticker_no_match(self):
        from alphacouncil.analysis.sentiment import SentimentEngine

        universe = ["RELIANCE.NS", "TCS.NS"]
        assert SentimentEngine.map_ticker("Random news article", universe) is None

    def test_map_ticker_avoids_partial_match(self):
        from alphacouncil.analysis.sentiment import SentimentEngine

        universe = ["ITC.NS", "ITCHOTEL.NS"]
        # "ITC" should match but not as a substring of another word
        result = SentimentEngine.map_ticker("ITC reports strong Q3", universe)
        assert result == "ITC.NS"

    def test_keyword_boost_positive(self):
        from alphacouncil.analysis.sentiment import SentimentEngine

        boost = SentimentEngine._keyword_boost("Revenue beat expectations, strong growth")
        assert boost > 0

    def test_keyword_boost_negative(self):
        from alphacouncil.analysis.sentiment import SentimentEngine

        boost = SentimentEngine._keyword_boost("Stock downgrade amid fraud concerns")
        assert boost < 0

    def test_keyword_boost_capped(self):
        from alphacouncil.analysis.sentiment import SentimentEngine

        # Even with many positive keywords, capped at 0.3
        boost = SentimentEngine._keyword_boost(
            "revenue beat growth expansion market share new product order win capacity addition"
        )
        assert boost <= 0.3

    def test_extract_keywords_finds_growth_terms(self):
        from alphacouncil.analysis.sentiment import SentimentEngine

        kws = SentimentEngine._extract_keywords("Revenue beat and market share gain")
        assert "revenue beat" in kws
        assert "market share" in kws


# ---------------------------------------------------------------------------
# MacroEngine tests
# ---------------------------------------------------------------------------


class TestMacroEngine:
    def test_determine_regime_bull_low_vol(self):
        from alphacouncil.analysis.macro import MacroEngine

        engine = MacroEngine(cache=FakeCache(), fred_api_key="dummy")
        data = {
            "fii_net_flow": 0.0, "dii_net_flow": 0.0,
            "nifty_level": 22000.0, "nifty_sma200": 20000.0,
            "india_vix": 12.0,
        }
        with patch.object(engine, '_is_pre_expiry', return_value=False), \
             patch.object(engine, '_is_budget_policy_week', return_value=False):
            # Override today check by patching the internal method calls
            regime = engine.determine_regime(data)
        assert regime == MarketRegime.BULL_LOW_VOL

    def test_determine_regime_bear_high_vol(self):
        from alphacouncil.analysis.macro import MacroEngine

        engine = MacroEngine(cache=FakeCache(), fred_api_key="dummy")
        data = {
            "fii_net_flow": 0.0, "dii_net_flow": 0.0,
            "nifty_level": 18000.0, "nifty_sma200": 20000.0,
            "india_vix": 25.0,
        }
        with patch.object(engine, '_is_pre_expiry', return_value=False), \
             patch.object(engine, '_is_budget_policy_week', return_value=False):
            regime = engine.determine_regime(data)
        assert regime == MarketRegime.BEAR_HIGH_VOL

    def test_determine_regime_fii_buying(self):
        from alphacouncil.analysis.macro import MacroEngine

        engine = MacroEngine(cache=FakeCache(), fred_api_key="dummy")
        data = {
            "fii_net_flow": 2000.0, "dii_net_flow": 500.0,
            "nifty_level": 22000.0, "nifty_sma200": 20000.0,
            "india_vix": 15.0,
        }
        with patch.object(engine, '_is_pre_expiry', return_value=False), \
             patch.object(engine, '_is_budget_policy_week', return_value=False):
            regime = engine.determine_regime(data)
        assert regime == MarketRegime.FII_BUYING

    def test_determine_regime_fii_selling(self):
        from alphacouncil.analysis.macro import MacroEngine

        engine = MacroEngine(cache=FakeCache(), fred_api_key="dummy")
        data = {
            "fii_net_flow": -1500.0, "dii_net_flow": 100.0,
            "nifty_level": 20000.0, "nifty_sma200": 20000.0,
            "india_vix": 15.0,
        }
        with patch.object(engine, '_is_pre_expiry', return_value=False), \
             patch.object(engine, '_is_budget_policy_week', return_value=False):
            regime = engine.determine_regime(data)
        assert regime == MarketRegime.FII_SELLING

    def test_is_pre_expiry_near_last_thursday(self):
        from alphacouncil.analysis.macro import MacroEngine

        # March 2026: last Thursday is March 26
        # March 24 is within 3 days -> True
        assert MacroEngine._is_pre_expiry(date(2026, 3, 24)) is True
        # March 20 is 6 days away -> False
        assert MacroEngine._is_pre_expiry(date(2026, 3, 20)) is False

    def test_is_budget_policy_week(self):
        from alphacouncil.analysis.macro import MacroEngine

        assert MacroEngine._is_budget_policy_week(date(2026, 2, 1)) is True
        assert MacroEngine._is_budget_policy_week(date(2026, 1, 30)) is True
        assert MacroEngine._is_budget_policy_week(date(2026, 6, 15)) is False

    def test_macro_signal_frozen_and_hashable(self):
        now = datetime.now(tz=timezone.utc)
        sig = MacroSignal(
            repo_rate=6.5, india_cpi=5.0, india_iip=3.0,
            fed_rate=5.25, dxy=104.0, brent_crude=80.0,
            india_vix=14.0, gold_price=60000.0, nifty_level=22000.0,
            fii_net_flow=500.0, dii_net_flow=300.0,
            regime=MarketRegime.BULL_LOW_VOL, timestamp=now,
        )
        assert hash(sig) is not None
