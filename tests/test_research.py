"""Tests for the research pipeline: StrategyDiscovery, StrategyBacktester,
AgentParameterOptimizer, PerformanceAttribution, RegimeAdaptiveWeightLearner,
NewsAlphaTracker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# StrategyDiscovery tests
# ---------------------------------------------------------------------------


class TestStrategyDiscovery:
    def test_compute_ic_known_correlation(self):
        from alphacouncil.research.discovery import StrategyDiscovery

        rng = np.random.default_rng(42)
        n = 100
        factor = pd.Series(rng.standard_normal(n))
        forward = factor + 0.1 * rng.standard_normal(n)  # highly correlated
        ic = StrategyDiscovery._compute_ic(factor, forward)
        assert ic > 0.5  # strong positive correlation

    def test_compute_ic_uncorrelated(self):
        from alphacouncil.research.discovery import StrategyDiscovery

        rng = np.random.default_rng(42)
        factor = pd.Series(rng.standard_normal(200))
        forward = pd.Series(rng.standard_normal(200))
        ic = StrategyDiscovery._compute_ic(factor, forward)
        assert abs(ic) < 0.2  # near zero

    def test_compute_ic_too_few_observations(self):
        from alphacouncil.research.discovery import StrategyDiscovery

        factor = pd.Series([1.0, 2.0])
        forward = pd.Series([3.0, 4.0])
        ic = StrategyDiscovery._compute_ic(factor, forward)
        assert ic == 0.0

    def test_compute_ic_constant_series(self):
        from alphacouncil.research.discovery import StrategyDiscovery

        factor = pd.Series([5.0] * 50)
        forward = pd.Series(np.random.randn(50))
        ic = StrategyDiscovery._compute_ic(factor, forward)
        assert ic == 0.0

    def test_rsi_computation(self):
        from alphacouncil.research.discovery import StrategyDiscovery

        close = pd.Series(np.linspace(100, 130, 100))
        rsi = StrategyDiscovery._rsi(close, period=14)
        # Monotonically increasing series -> RSI should be high
        last_rsi = rsi.dropna().iloc[-1]
        assert last_rsi > 60

    def test_describe_feature_interaction(self):
        from alphacouncil.research.discovery import StrategyDiscovery

        desc = StrategyDiscovery._describe_feature("return_5_x_volume_ratio_5")
        assert "Interaction" in desc

    def test_describe_feature_single(self):
        from alphacouncil.research.discovery import StrategyDiscovery

        desc = StrategyDiscovery._describe_feature("return_20")
        assert "20" in desc

    def test_check_robustness_insufficient_data(self):
        from alphacouncil.research.discovery import StrategyDiscovery

        ic_series = pd.Series(np.random.randn(50))  # too short for 252-window
        score = StrategyDiscovery._check_robustness("test", ic_series)
        assert score == 0.0

    def test_growth_bias_positive_correlation(self):
        from alphacouncil.research.discovery import StrategyDiscovery

        rng = np.random.default_rng(42)
        n = 200
        close = pd.Series(100 + np.cumsum(rng.normal(0.1, 1.0, n)))
        factor = close.pct_change(10).fillna(0)
        price_data = pd.DataFrame({"close": close, "volume": [1000] * n})

        bonus = StrategyDiscovery._compute_growth_bias("momentum_10", factor, price_data)
        assert bonus >= 0.0  # should be non-negative


# ---------------------------------------------------------------------------
# StrategyBacktester cost calculation tests
# ---------------------------------------------------------------------------


class TestBacktesterCosts:
    def test_buy_cost_components(self):
        from alphacouncil.research.backtester import StrategyBacktester

        cost = StrategyBacktester._compute_costs(100, 1000.0, "BUY")
        turnover = 100 * 1000.0
        # Components for BUY: brokerage + GST + SEBI + stamp + slippage
        brokerage = turnover * 0.0005
        gst = brokerage * 0.18
        sebi = turnover * 0.000001
        stamp = turnover * 0.00015
        slippage = turnover * 0.001
        expected = brokerage + gst + sebi + stamp + slippage
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_sell_cost_components(self):
        from alphacouncil.research.backtester import StrategyBacktester

        cost = StrategyBacktester._compute_costs(100, 1000.0, "SELL")
        turnover = 100 * 1000.0
        brokerage = turnover * 0.0005
        gst = brokerage * 0.18
        sebi = turnover * 0.000001
        stt = turnover * 0.001
        slippage = turnover * 0.001
        expected = brokerage + gst + sebi + stt + slippage
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_round_trip_cost_percentage(self):
        from alphacouncil.research.backtester import StrategyBacktester

        price = 1000.0
        qty = 100
        buy_cost = StrategyBacktester._compute_costs(qty, price, "BUY")
        sell_cost = StrategyBacktester._compute_costs(qty, price, "SELL")
        total = buy_cost + sell_cost
        turnover = qty * price
        pct = total / turnover * 100
        # Round-trip should be roughly 0.3-0.5% given all charges
        assert 0.2 < pct < 1.0

    def test_cost_scales_linearly(self):
        from alphacouncil.research.backtester import StrategyBacktester

        c1 = StrategyBacktester._compute_costs(10, 1000.0, "BUY")
        c2 = StrategyBacktester._compute_costs(100, 1000.0, "BUY")
        assert c2 == pytest.approx(c1 * 10, rel=1e-9)

    def test_zero_quantity_returns_zero(self):
        """Edge case: cost for zero shares should be zero charges."""
        from alphacouncil.research.backtester import StrategyBacktester

        cost = StrategyBacktester._compute_costs(0, 1000.0, "BUY")
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Backtester window generation
# ---------------------------------------------------------------------------


class TestBacktesterWindows:
    def test_build_windows_structure(self):
        from alphacouncil.research.backtester import StrategyBacktester

        windows = StrategyBacktester._build_windows("2020-01-01", "2025-12-31")
        assert len(windows) > 0
        for train_start, train_end, test_start, test_end in windows:
            # Each is an ISO date string
            assert len(train_start) == 10
            # test_start should be after train_end
            assert test_start > train_end

    def test_build_windows_short_range_empty(self):
        from alphacouncil.research.backtester import StrategyBacktester

        windows = StrategyBacktester._build_windows("2025-01-01", "2025-06-01")
        # Range too short for train + test
        assert len(windows) == 0


# ---------------------------------------------------------------------------
# Backtester aggregation
# ---------------------------------------------------------------------------


class TestBacktesterAggregation:
    def test_aggregate_positive_returns(self):
        from alphacouncil.research.backtester import StrategyBacktester

        daily = [0.001] * 252  # consistent small positive
        result = StrategyBacktester._aggregate_metrics(daily, [], "2023-01-01", "2024-01-01")
        assert result["sharpe"] > 0
        assert result["max_dd"] < 0.05
        assert result["win_rate"] == 1.0

    def test_aggregate_empty_returns(self):
        from alphacouncil.research.backtester import StrategyBacktester

        result = StrategyBacktester._aggregate_metrics([], [], "2023-01-01", "2024-01-01")
        assert result["sharpe"] == 0.0

    def test_pass_fail_criteria(self):
        """Sharpe >0.5, DD <20%, 3/4 profitable."""
        from alphacouncil.research.backtester import (
            _MAX_DRAWDOWN,
            _MIN_PROFITABLE_PERIODS_RATIO,
            _MIN_WALK_FORWARD_SHARPE,
        )

        assert _MIN_WALK_FORWARD_SHARPE == 0.5
        assert _MAX_DRAWDOWN == 0.20
        assert _MIN_PROFITABLE_PERIODS_RATIO == 0.75


# ---------------------------------------------------------------------------
# AgentParameterOptimizer tests
# ---------------------------------------------------------------------------


class TestParameterOptimizer:
    def test_create_search_space_float(self):
        from alphacouncil.research.optimizer import AgentParameterOptimizer

        params = {"threshold": 0.5, "multiplier": 2.0}
        space = AgentParameterOptimizer._create_search_space(params)
        assert "threshold" in space
        assert space["threshold"]["type"] == "float"
        assert space["threshold"]["low"] == pytest.approx(0.5 * 0.7)
        assert space["threshold"]["high"] == pytest.approx(0.5 * 1.3)

    def test_create_search_space_int(self):
        from alphacouncil.research.optimizer import AgentParameterOptimizer

        params = {"max_positions": 15}
        space = AgentParameterOptimizer._create_search_space(params)
        assert space["max_positions"]["type"] == "int"
        assert space["max_positions"]["low"] < 15
        assert space["max_positions"]["high"] > 15

    def test_create_search_space_skips_bool(self):
        from alphacouncil.research.optimizer import AgentParameterOptimizer

        params = {"enabled": True, "threshold": 0.5}
        space = AgentParameterOptimizer._create_search_space(params)
        assert "enabled" not in space
        assert "threshold" in space

    def test_create_search_space_zero_float(self):
        from alphacouncil.research.optimizer import AgentParameterOptimizer

        params = {"offset": 0.0}
        space = AgentParameterOptimizer._create_search_space(params)
        assert space["offset"]["low"] == -0.1
        assert space["offset"]["high"] == 0.1

    def test_growth_allocation_penalty(self):
        from alphacouncil.research.optimizer import AgentParameterOptimizer

        current = {"growth_weight": 0.30, "momentum_factor": 0.20, "other": 1.0}
        new = {"growth_weight": 0.20, "momentum_factor": 0.15, "other": 1.0}
        penalty = AgentParameterOptimizer._growth_allocation_penalty(current, new)
        assert penalty > 0  # reducing growth/momentum params -> penalty

    def test_no_penalty_when_growth_increases(self):
        from alphacouncil.research.optimizer import AgentParameterOptimizer

        current = {"growth_weight": 0.20}
        new = {"growth_weight": 0.30}
        penalty = AgentParameterOptimizer._growth_allocation_penalty(current, new)
        assert penalty == 0.0


# ---------------------------------------------------------------------------
# PerformanceAttribution tests
# ---------------------------------------------------------------------------


class TestPerformanceAttribution:
    def test_factor_decomposition_sync(self):
        from alphacouncil.research.attribution import PerformanceAttribution

        trades = [
            {"side": "BUY", "price": 100.0, "factor_scores": {"momentum": 0.5, "value": 0.3}},
            {"side": "SELL", "price": 100.0, "factor_scores": {"momentum": -0.2, "value": 0.8}},
        ]
        result = PerformanceAttribution._factor_decomposition_sync(trades, pd.DataFrame())
        assert "momentum" in result
        assert "value" in result
        assert isinstance(result["momentum"], float)

    def test_empty_result_structure(self):
        from alphacouncil.research.attribution import PerformanceAttribution

        result = PerformanceAttribution._empty_result()
        assert result["total_alpha"] == 0.0
        assert result["factor_contrib"] == {}


# ---------------------------------------------------------------------------
# RegimeAdaptiveWeightLearner tests
# ---------------------------------------------------------------------------


class TestRegimeWeightLearner:
    def test_default_weights_sum_to_one(self):
        from alphacouncil.research.regime_learner import DEFAULT_WEIGHTS

        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_init_sets_all_regimes(self):
        from alphacouncil.research.regime_learner import (
            REGIMES,
            RegimeAdaptiveWeightLearner,
        )

        engine = MagicMock()
        learner = RegimeAdaptiveWeightLearner(db_engine=engine)
        for regime in REGIMES:
            weights = learner.get_weights(regime)
            assert abs(sum(weights.values()) - 1.0) < 0.05  # may have been adjusted

    def test_sharpe_to_weights_normalization(self):
        from alphacouncil.research.regime_learner import RegimeAdaptiveWeightLearner

        engine = MagicMock()
        learner = RegimeAdaptiveWeightLearner(db_engine=engine)
        sharpes = {
            "GrowthMomentumAgent": 1.5,
            "MeanReversionAgent": 0.8,
            "MultiFactorRankingAgent": 1.0,
        }
        weights = learner._sharpe_to_weights(sharpes)
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert all(w > 0 for w in weights.values())

    def test_sharpe_to_weights_growth_bias(self):
        from alphacouncil.research.regime_learner import RegimeAdaptiveWeightLearner

        engine = MagicMock()
        learner = RegimeAdaptiveWeightLearner(db_engine=engine)
        # Equal sharpes -> growth agents should get more weight due to 1.5x bias
        sharpes = {
            "GrowthMomentumAgent": 1.0,
            "MeanReversionAgent": 1.0,
        }
        weights = learner._sharpe_to_weights(sharpes)
        assert weights["GrowthMomentumAgent"] > weights["MeanReversionAgent"]

    def test_classify_regime_pre_expiry(self):
        from alphacouncil.research.regime_learner import RegimeAdaptiveWeightLearner

        engine = MagicMock()
        learner = RegimeAdaptiveWeightLearner(db_engine=engine)
        regime = learner.classify_regime(
            nifty_return_3m=0.05, vix=15.0, fii_flow=0.0,
            days_to_expiry=2, month=3,
        )
        assert regime == "PRE_EXPIRY"

    def test_classify_regime_earnings_season(self):
        from alphacouncil.research.regime_learner import RegimeAdaptiveWeightLearner

        engine = MagicMock()
        learner = RegimeAdaptiveWeightLearner(db_engine=engine)
        regime = learner.classify_regime(
            nifty_return_3m=0.05, vix=15.0, fii_flow=0.0,
            days_to_expiry=20, month=7,
        )
        assert regime == "EARNINGS_SEASON"

    def test_classify_regime_fii_buying(self):
        from alphacouncil.research.regime_learner import RegimeAdaptiveWeightLearner

        engine = MagicMock()
        learner = RegimeAdaptiveWeightLearner(db_engine=engine)
        regime = learner.classify_regime(
            nifty_return_3m=0.05, vix=15.0, fii_flow=1500.0,
            days_to_expiry=20, month=3,
        )
        assert regime == "FII_BUYING"


# ---------------------------------------------------------------------------
# NewsAlphaTracker tests
# ---------------------------------------------------------------------------


class TestNewsAlphaTracker:
    def test_extract_keywords_growth(self):
        from alphacouncil.research.news_alpha import NewsAlphaTracker

        tracker = NewsAlphaTracker(db_engine=MagicMock())
        kws = tracker._extract_keywords("Revenue beat expectations, strong growth and expansion")
        assert "revenue beat" in kws
        assert "growth" in kws
        assert "expansion" in kws

    def test_extract_keywords_negative(self):
        from alphacouncil.research.news_alpha import NewsAlphaTracker

        tracker = NewsAlphaTracker(db_engine=MagicMock())
        kws = tracker._extract_keywords("Company faces downgrade and fraud allegations")
        assert "downgrade" in kws
        assert "fraud" in kws

    def test_extract_keywords_empty(self):
        from alphacouncil.research.news_alpha import NewsAlphaTracker

        tracker = NewsAlphaTracker(db_engine=MagicMock())
        kws = tracker._extract_keywords("Regular market update")
        assert len(kws) == 0

    def test_update_keyword_alpha(self):
        from alphacouncil.research.news_alpha import NewsAlphaTracker

        tracker = NewsAlphaTracker(db_engine=MagicMock())
        results = [
            {"keywords": ["growth"], "return_3d": 0.05},
            {"keywords": ["growth"], "return_3d": 0.03},
            {"keywords": ["downgrade"], "return_3d": -0.04},
        ]
        tracker._update_keyword_alpha(results)
        assert "growth" in tracker.keyword_alpha
        assert tracker.keyword_alpha["growth"] == pytest.approx(0.04)
        assert tracker.keyword_alpha["downgrade"] == pytest.approx(-0.04)
