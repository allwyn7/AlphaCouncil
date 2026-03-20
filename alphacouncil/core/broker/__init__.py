"""Broker adapter layer for AlphaCouncil.

Provides a uniform async interface for paper trading, Angel One, and Fyers
brokers.  All concrete adapters inherit from :class:`BrokerAdapter` and are
interchangeable at runtime.

Usage::

    from alphacouncil.core.broker import BrokerAdapter, PaperBroker

    broker: BrokerAdapter = PaperBroker(initial_capital=1_000_000)
    order_id = await broker.place_order(order)
"""

from alphacouncil.core.broker.angelone import AngelOneBroker
from alphacouncil.core.broker.base import BrokerAdapter
from alphacouncil.core.broker.fyers import FyersBroker
from alphacouncil.core.broker.paper import PaperBroker

__all__ = [
    "BrokerAdapter",
    "PaperBroker",
    "AngelOneBroker",
    "FyersBroker",
]
