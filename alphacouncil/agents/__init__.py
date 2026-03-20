"""Trading agents for the AlphaCouncil multi-agent system.

Each agent encapsulates a distinct quant strategy (factor model + signal
logic) and exposes a uniform interface via :class:`BaseAgent`.  Agents are
composed at runtime by the orchestrator and communicate through the shared
:class:`~alphacouncil.core.bus.MessageBus`.

Available agents
----------------
- **GrowthMomentumAgent** -- hunts high-growth stocks with strong price
  momentum.  Primary alpha source.
- **MeanReversionAgent** -- catches oversold *growth* stocks (buy-the-dip on
  quality).  Complements the momentum agent by harvesting short-term
  dislocations.
- **SentimentAlphaAgent** -- trades on news and social narratives with an
  explicit growth bias.  Uses FinBERT-driven sentiment scoring, keyword
  analysis, and institutional flow confirmation.
- **PortfolioOptimizationAgent** -- meta-allocator that optimises risk-
  adjusted allocation across all signals from other agents via mean-variance
  optimisation with Ledoit-Wolf shrinkage.
- **MultiFactorRankingAgent** -- cross-sectional multi-factor ranking with a
  heavy growth tilt (35 % growth vs 8 % value).  Rebalances weekly.
- **VolatilityRegimeAgent** -- detects LOW / MEDIUM / HIGH / TRANSITION
  volatility regimes and emits meta-signals to modulate system-wide risk.
"""

from alphacouncil.agents.base import BaseAgent
from alphacouncil.agents.growth_momentum import GrowthMomentumAgent
from alphacouncil.agents.mean_reversion import MeanReversionAgent
from alphacouncil.agents.multifactor import MultiFactorRankingAgent
from alphacouncil.agents.portfolio_optimizer import PortfolioOptimizationAgent
from alphacouncil.agents.sentiment import SentimentAlphaAgent
from alphacouncil.agents.volatility import VolatilityRegimeAgent

__all__ = [
    "BaseAgent",
    "GrowthMomentumAgent",
    "MeanReversionAgent",
    "MultiFactorRankingAgent",
    "PortfolioOptimizationAgent",
    "SentimentAlphaAgent",
    "VolatilityRegimeAgent",
]
