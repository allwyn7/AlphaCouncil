"""Research pipeline for AlphaCouncil.

This package implements the offline / nightly research loop that drives
strategy discovery, walk-forward backtesting, parameter optimisation,
performance attribution, regime-adaptive weight learning, and news-alpha
tracking.

Modules
-------
discovery
    Weekly candidate-feature generation and information-coefficient screening.
backtester
    Walk-forward validation with realistic Indian equity transaction costs.
optimizer
    Nightly Bayesian parameter optimisation via Optuna.
attribution
    Brinson--Fachler return decomposition for factor, timing, and selection
    contributions.
regime_learner
    Learns optimal agent weights conditioned on Indian market regimes.
news_alpha
    Event-study framework for refining sentiment keyword weights.
"""

from __future__ import annotations

from alphacouncil.research.attribution import PerformanceAttribution
from alphacouncil.research.backtester import StrategyBacktester
from alphacouncil.research.discovery import StrategyDiscovery
from alphacouncil.research.news_alpha import NewsAlphaTracker
from alphacouncil.research.optimizer import AgentParameterOptimizer
from alphacouncil.research.regime_learner import RegimeAdaptiveWeightLearner

__all__ = [
    "StrategyDiscovery",
    "StrategyBacktester",
    "AgentParameterOptimizer",
    "PerformanceAttribution",
    "RegimeAdaptiveWeightLearner",
    "NewsAlphaTracker",
]
