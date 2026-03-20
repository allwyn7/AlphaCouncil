"""Analysis engines for the AlphaCouncil trading system.

Provides four specialised analysis engines that produce frozen Pydantic signal
models consumed by downstream agents and the meta-controller:

* :class:`TechnicalEngine`   -- pandas-ta based technical indicators
* :class:`FundamentalEngine` -- yfinance-backed valuation & growth metrics
* :class:`SentimentEngine`   -- FinBERT NLP over news & social feeds
* :class:`MacroEngine`       -- macro-economic regime classification
"""

from alphacouncil.analysis.fundamental import FundamentalEngine
from alphacouncil.analysis.macro import MacroEngine
from alphacouncil.analysis.sentiment import SentimentEngine
from alphacouncil.analysis.technical import TechnicalEngine

__all__ = [
    "FundamentalEngine",
    "MacroEngine",
    "SentimentEngine",
    "TechnicalEngine",
]
