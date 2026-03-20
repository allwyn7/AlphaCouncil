"""Tests for the AlphaCouncil quant agents.

Covers BaseAgent interface, GrowthMomentumAgent, MeanReversionAgent,
MultiFactorRankingAgent, SentimentAlphaAgent, PortfolioOptimizationAgent.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from alphacouncil.agents.base import BaseAgent, MessageBus
from alphacouncil.core.models import Action, AgentSignal, AgentStatus


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200, start_price: float = 100.0, ticker: str = "TEST") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with ``n`` rows."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=datetime.now(), periods=n)
    close = start_price + np.cumsum(rng.normal(0.0, 1.0, n))
    close = np.maximum(close, 1.0)  # floor at 1
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    low = np.maximum(low, 0.5)
    opn = close + rng.uniform(-1.0, 1.0, n)
    opn = np.maximum(opn, 0.5)
    volume = rng.integers(100_000, 1_000_000, n).astype(np.float64)
    df = pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return df


def _make_mock_bus() -> MessageBus:
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


class ConcreteTestAgent(BaseAgent):
    """Minimal concrete implementation for testing the abstract base."""

    def __init__(self, name: str = "test_agent", **kw: Any):
        super().__init__(
            name=name,
            config=MagicMock(),
            cache=MagicMock(),
            bus=_make_mock_bus(),
            db_engine=MagicMock(),
        )
        self._parameters = {"param_a": 1.0, "param_b": "hello"}

    async def generate_signals(
        self, universe: list[str], market_data: dict[str, Any]
    ) -> list[AgentSignal]:
        now = datetime.now(tz=timezone.utc)
        return [
            AgentSignal(
                ticker=t,
                action=Action.BUY,
                conviction=75,
                target_weight=0.05,
                stop_loss=90.0,
                take_profit=120.0,
                factor_scores={"alpha": 1.0},
                reasoning="test signal",
                holding_period_days=10,
                agent_name=self._name,
                timestamp=now,
            )
            for t in universe
        ]

    def get_parameters(self) -> dict[str, Any]:
        return dict(self._parameters)

    def set_parameters(self, params: dict[str, Any]) -> None:
        self._parameters.update(params)


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------


class TestBaseAgent:
    def test_initial_status_is_backtest(self):
        agent = ConcreteTestAgent()
        assert agent.status == AgentStatus.BACKTEST

    def test_name_property(self):
        agent = ConcreteTestAgent(name="my_agent")
        assert agent.name == "my_agent"

    def test_set_status(self):
        agent = ConcreteTestAgent()
        agent.set_status(AgentStatus.PAPER)
        assert agent.status == AgentStatus.PAPER
        assert agent.get_status() == AgentStatus.PAPER

    def test_get_set_parameters(self):
        agent = ConcreteTestAgent()
        assert agent.get_parameters()["param_a"] == 1.0
        agent.set_parameters({"param_a": 2.5})
        assert agent.get_parameters()["param_a"] == 2.5

    @pytest.mark.asyncio
    async def test_generate_signals_returns_agent_signals(self):
        agent = ConcreteTestAgent()
        signals = await agent.generate_signals(["INFY", "TCS"], {})
        assert len(signals) == 2
        for sig in signals:
            assert isinstance(sig, AgentSignal)

    @pytest.mark.asyncio
    async def test_run_cycle_publishes_to_bus(self):
        agent = ConcreteTestAgent()
        signals = await agent.run_cycle(["RELIANCE"], {})
        assert len(signals) == 1
        agent._bus.publish.assert_called()

    def test_zscore_identical_values_returns_zeros(self):
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = BaseAgent._zscore(s)
        assert all(result == 0.0)

    def test_zscore_known_distribution(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = BaseAgent._zscore(s)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std(ddof=0) - 1.0) < 1e-10

    def test_compute_conviction_clamps_to_0_100(self):
        assert BaseAgent._compute_conviction(-5.0) == 0
        assert BaseAgent._compute_conviction(10.0) == 100
        assert 0 <= BaseAgent._compute_conviction(1.0) <= 100

    def test_compute_conviction_linear_interpolation(self):
        mid = BaseAgent._compute_conviction(1.0, min_score=0.0, max_score=2.0)
        assert mid == 50


# ---------------------------------------------------------------------------
# GrowthMomentumAgent tests
# ---------------------------------------------------------------------------


class TestGrowthMomentumAgent:
    @pytest.mark.asyncio
    async def test_signal_generation_with_mock_data(self):
        from alphacouncil.agents.growth_momentum import GrowthMomentumAgent

        agent = GrowthMomentumAgent(
            config=MagicMock(),
            cache=MagicMock(),
            bus=_make_mock_bus(),
            db_engine=MagicMock(),
        )
        universe = ["RELIANCE", "TCS"]
        df_rel = _make_ohlcv(200, 2500.0)
        df_tcs = _make_ohlcv(200, 3500.0)

        market_data = {
            "prices": {"RELIANCE": df_rel, "TCS": df_tcs},
            "fundamentals": {
                "RELIANCE": {"revenueGrowth": 0.25, "earningsGrowth": 0.30,
                              "quarterly_revenue_growth": [0.15, 0.20, 0.28]},
                "TCS": {"revenueGrowth": 0.10, "earningsGrowth": 0.12},
            },
            "sentiment": {
                "RELIANCE": {"sentiment_7d": 0.3, "sentiment_30d": 0.1},
                "TCS": {"sentiment_7d": -0.1, "sentiment_30d": 0.0},
            },
            "nifty_prices": _make_ohlcv(200, 20000.0),
        }

        signals = await agent.generate_signals(universe, market_data)
        for sig in signals:
            assert isinstance(sig, AgentSignal)
            assert 0 <= sig.conviction <= 100
            assert sig.agent_name == "growth_momentum"

    def test_default_parameters_loaded(self):
        from alphacouncil.agents.growth_momentum import GrowthMomentumAgent

        agent = GrowthMomentumAgent(
            config=MagicMock(), cache=MagicMock(),
            bus=_make_mock_bus(), db_engine=MagicMock(),
        )
        params = agent.get_parameters()
        assert params["buy_threshold"] == 0.70
        assert params["atr_multiplier"] == 2.5
        assert abs(sum(params["factor_weights"].values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# MeanReversionAgent tests
# ---------------------------------------------------------------------------


class TestMeanReversionAgent:
    @pytest.mark.asyncio
    async def test_signal_generation(self):
        from alphacouncil.agents.mean_reversion import MeanReversionAgent

        agent = MeanReversionAgent(
            config=MagicMock(), cache=MagicMock(),
            bus=_make_mock_bus(), db_engine=MagicMock(),
        )

        universe = ["INFY", "WIPRO"]
        market_data = {
            "prices": {
                "INFY": _make_ohlcv(200, 1500.0),
                "WIPRO": _make_ohlcv(200, 400.0),
            },
            "fundamentals": {
                "INFY": {"revenueGrowth": 0.20, "returnOnEquity": 0.25},
                "WIPRO": {"revenueGrowth": 0.05, "returnOnEquity": 0.08},
            },
        }

        signals = await agent.generate_signals(universe, market_data)
        for sig in signals:
            assert isinstance(sig, AgentSignal)
            assert 0 <= sig.conviction <= 100

    def test_growth_quality_filter(self):
        from alphacouncil.agents.mean_reversion import MeanReversionAgent

        fund = {"GOOD": {"revenueGrowth": 0.20, "returnOnEquity": 0.15},
                "BAD": {"revenueGrowth": 0.03, "returnOnEquity": 0.05}}

        good = MeanReversionAgent._growth_quality_score(fund, "GOOD", 0.10, 0.12)
        bad = MeanReversionAgent._growth_quality_score(fund, "BAD", 0.10, 0.12)
        assert good == 1.0
        assert bad == -2.0

    def test_rsi_zscore_mapping(self):
        from alphacouncil.agents.mean_reversion import MeanReversionAgent

        assert MeanReversionAgent._rsi_zscore(50.0) == 0.0
        assert MeanReversionAgent._rsi_zscore(30.0) < 0
        assert MeanReversionAgent._rsi_zscore(70.0) > 0

    def test_bollinger_zscore_mapping(self):
        from alphacouncil.agents.mean_reversion import MeanReversionAgent

        assert MeanReversionAgent._bollinger_zscore(0.5) == 0.0
        assert MeanReversionAgent._bollinger_zscore(0.0) == -2.0
        assert MeanReversionAgent._bollinger_zscore(1.0) == 2.0

    def test_ou_halflife_zscore_non_reverting(self):
        from alphacouncil.agents.mean_reversion import MeanReversionAgent

        assert MeanReversionAgent._ou_halflife_zscore(np.inf, 20.0) == -2.0
        assert MeanReversionAgent._ou_halflife_zscore(25.0, 20.0) == -2.0


# ---------------------------------------------------------------------------
# MultiFactorRankingAgent tests
# ---------------------------------------------------------------------------


class TestMultiFactorRankingAgent:
    def test_ranking_composite_score(self):
        from alphacouncil.agents.multifactor import MultiFactorRankingAgent

        agent = MultiFactorRankingAgent(
            config=MagicMock(), cache=MagicMock(),
            bus=_make_mock_bus(), db_engine=MagicMock(),
        )
        factor_df = pd.DataFrame({
            "revenue_growth": [0.3, 0.1, 0.2],
            "eps_growth": [0.2, 0.05, 0.15],
            "roe": [0.2, 0.1, 0.15],
            "fcf_yield": [0.05, 0.03, 0.04],
            "momentum_6m": [0.15, -0.05, 0.10],
            "low_volatility": [2.0, 1.5, 1.8],
            "delivery_pct": [0.6, 0.4, 0.5],
            "fii_dii_net_flow": [100, -50, 30],
            "gross_margin_expansion": [0.02, -0.01, 0.01],
        }, index=["STOCK_A", "STOCK_B", "STOCK_C"])

        composite = agent._compute_composite(factor_df)
        assert len(composite) == 3
        assert composite.index.tolist() == ["STOCK_A", "STOCK_B", "STOCK_C"]
        # STOCK_A has best metrics across the board -- should rank highest
        assert composite["STOCK_A"] > composite["STOCK_B"]

    def test_factor_weights_sum_to_one(self):
        from alphacouncil.agents.multifactor import _DEFAULT_FACTOR_WEIGHTS

        assert abs(sum(_DEFAULT_FACTOR_WEIGHTS.values()) - 1.0) < 1e-9

    def test_set_parameters_rejects_bad_weights(self):
        from alphacouncil.agents.multifactor import MultiFactorRankingAgent

        agent = MultiFactorRankingAgent(
            config=MagicMock(), cache=MagicMock(),
            bus=_make_mock_bus(), db_engine=MagicMock(),
        )
        bad_weights = {"revenue_growth": 0.5, "eps_growth": 0.6}
        agent.set_parameters({"factor_weights": bad_weights})
        # Weights should NOT have been updated (sum != 1)
        params = agent.get_parameters()
        assert abs(sum(params["factor_weights"].values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# SentimentAlphaAgent tests
# ---------------------------------------------------------------------------


class TestSentimentAlphaAgent:
    def test_compute_sentiment_score_empty(self):
        from alphacouncil.agents.sentiment import SentimentAlphaAgent

        assert SentimentAlphaAgent._compute_sentiment_score([]) == 0.0

    def test_compute_sentiment_score_with_signals(self):
        from alphacouncil.agents.sentiment import SentimentAlphaAgent
        from types import SimpleNamespace

        sigs = [SimpleNamespace(score=0.8), SimpleNamespace(score=0.4)]
        result = SentimentAlphaAgent._compute_sentiment_score(sigs)
        assert abs(result - 0.6) < 1e-9

    def test_keyword_score_positive(self):
        from alphacouncil.agents.sentiment import SentimentAlphaAgent
        from types import SimpleNamespace

        sigs = [SimpleNamespace(keywords=["revenue beat", "expansion"])]
        score = SentimentAlphaAgent._compute_keyword_score(sigs)
        assert score > 0

    def test_keyword_score_negative(self):
        from alphacouncil.agents.sentiment import SentimentAlphaAgent
        from types import SimpleNamespace

        sigs = [SimpleNamespace(keywords=["downgrade", "fraud"])]
        score = SentimentAlphaAgent._compute_keyword_score(sigs)
        assert score < 0

    def test_normalise_flow(self):
        from alphacouncil.agents.sentiment import SentimentAlphaAgent

        assert SentimentAlphaAgent._normalise_flow(500.0) == 1.0
        assert SentimentAlphaAgent._normalise_flow(-500.0) == -1.0
        assert abs(SentimentAlphaAgent._normalise_flow(250.0) - 0.5) < 1e-9

    def test_compute_buzz_score(self):
        from alphacouncil.agents.sentiment import SentimentAlphaAgent

        assert SentimentAlphaAgent._compute_buzz_score({}) == 0.0
        result = SentimentAlphaAgent._compute_buzz_score(
            {"mention_velocity": 3.0, "avg_velocity": 1.0}
        )
        assert result == 1.0  # (3-1)/2 = 1.0

    def test_has_growth_keywords(self):
        from alphacouncil.agents.sentiment import SentimentAlphaAgent
        from types import SimpleNamespace

        positive = [SimpleNamespace(keywords=["revenue beat"])]
        negative = [SimpleNamespace(keywords=["random news"])]
        assert SentimentAlphaAgent._has_growth_keywords(positive) is True
        assert SentimentAlphaAgent._has_growth_keywords(negative) is False


# ---------------------------------------------------------------------------
# PortfolioOptimizationAgent tests
# ---------------------------------------------------------------------------


class TestPortfolioOptimizationAgent:
    def test_select_candidates_with_agreement(self):
        from alphacouncil.agents.portfolio_optimizer import PortfolioOptimizationAgent

        agent = PortfolioOptimizationAgent(
            config=MagicMock(), cache=MagicMock(),
            bus=_make_mock_bus(), db_engine=MagicMock(),
        )
        now = datetime.now(tz=timezone.utc)
        sig = lambda t, a: AgentSignal(
            ticker=t, action=a, conviction=70, target_weight=0.05,
            stop_loss=90.0, take_profit=120.0, factor_scores={},
            reasoning="test", holding_period_days=7,
            agent_name="growth_momentum", timestamp=now,
        )

        agent_signals = {
            "growth_momentum": [sig("INFY", Action.BUY), sig("TCS", Action.BUY)],
            "mean_reversion": [sig("INFY", Action.BUY)],
            "sentiment_alpha": [sig("INFY", Action.BUY), sig("RELIANCE", Action.BUY)],
        }

        candidates, counts, growth_tilt = agent._select_candidates(agent_signals)
        # INFY has 3 BUY signals (>= min_agent_agreement of 2)
        assert "INFY" in candidates
        assert "INFY" in growth_tilt
        assert counts["INFY"] == 3

    def test_equal_weight_fallback(self):
        from alphacouncil.agents.portfolio_optimizer import PortfolioOptimizationAgent

        agent = PortfolioOptimizationAgent(
            config=MagicMock(), cache=MagicMock(),
            bus=_make_mock_bus(), db_engine=MagicMock(),
        )
        fallback = agent._equal_weight_fallback(["A", "B", "C"])
        assert len(fallback) == 3
        assert sum(fallback.values()) <= 0.80 + 1e-9  # max_deployed
        assert all(w <= 0.05 for w in fallback.values())  # max_weight

    def test_incremental_trading_dampens_changes(self):
        from alphacouncil.agents.portfolio_optimizer import PortfolioOptimizationAgent

        agent = PortfolioOptimizationAgent(
            config=MagicMock(), cache=MagicMock(),
            bus=_make_mock_bus(), db_engine=MagicMock(),
        )
        agent._current_weights = {"INFY": 0.04, "TCS": 0.03}
        optimal = {"INFY": 0.05, "TCS": 0.0, "RELIANCE": 0.04}
        result = agent._apply_incremental_trading(optimal)
        # INFY should move 30% toward 0.05 from 0.04 => ~0.043
        assert result.get("INFY", 0) < 0.05
        assert result.get("INFY", 0) > 0.04


# ---------------------------------------------------------------------------
# Cross-cutting: all agents return valid AgentSignal objects
# ---------------------------------------------------------------------------


class TestAgentSignalValidity:
    def test_conviction_bounds_in_model(self):
        now = datetime.now(tz=timezone.utc)
        sig = AgentSignal(
            ticker="TEST", action=Action.BUY, conviction=50,
            target_weight=0.05, stop_loss=90.0, take_profit=120.0,
            factor_scores={}, reasoning="test", holding_period_days=5,
            agent_name="test", timestamp=now,
        )
        assert 0 <= sig.conviction <= 100

    def test_conviction_above_100_raises(self):
        from pydantic import ValidationError

        now = datetime.now(tz=timezone.utc)
        with pytest.raises(ValidationError):
            AgentSignal(
                ticker="TEST", action=Action.BUY, conviction=101,
                target_weight=0.05, stop_loss=90.0, take_profit=120.0,
                factor_scores={}, reasoning="test", holding_period_days=5,
                agent_name="test", timestamp=now,
            )

    def test_agent_signal_is_frozen(self):
        now = datetime.now(tz=timezone.utc)
        sig = AgentSignal(
            ticker="TEST", action=Action.HOLD, conviction=50,
            target_weight=0.05, stop_loss=90.0, take_profit=120.0,
            factor_scores={}, reasoning="test", holding_period_days=5,
            agent_name="test", timestamp=now,
        )
        with pytest.raises(Exception):
            sig.conviction = 99  # type: ignore[misc]
