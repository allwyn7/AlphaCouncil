"""In-process asynchronous message bus for AlphaCouncil.

All inter-agent communication happens through :class:`MessageBus` using pure
``asyncio.Queue`` instances -- no HTTP, no external broker.  Each subscriber
receives its *own* queue for a given topic, so slow consumers cannot block
fast publishers.

Supported topics::

    price_update  -- real-time / delayed price ticks
    signal        -- agent trade signals
    sentiment     -- sentiment scores
    macro         -- macro-economic data updates
    order         -- order lifecycle events
    risk_alert    -- risk-limit breaches / warnings
    system        -- health-checks, heartbeats, admin commands

Usage::

    from alphacouncil.core.message_bus import MessageBus

    bus = MessageBus()

    q = bus.subscribe("signal")
    await bus.publish("signal", {"symbol": "INFY", "action": "buy"})
    msg = await q.get()          # -> {"symbol": "INFY", "action": "buy"}

    bus.unsubscribe("signal", q)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical topics
# ---------------------------------------------------------------------------

TOPICS: Final[frozenset[str]] = frozenset(
    {
        "price_update",
        "signal",
        "sentiment",
        "macro",
        "order",
        "risk_alert",
        "system",
    }
)

# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Envelope:
    """Thin wrapper around a published message.

    Attributes
    ----------
    topic:
        The topic the message was published to.
    payload:
        The actual message content (arbitrary Python object).
    timestamp:
        Monotonic time (``time.monotonic()``) when the message was published.
    publisher:
        Optional identifier for the publishing agent / component.
    """

    topic: str
    payload: Any
    timestamp: float = field(default_factory=time.monotonic)
    publisher: str = ""


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------


class MessageBus:
    """Async, in-process publish / subscribe message bus.

    The bus maintains a ``dict[str, list[asyncio.Queue]]`` mapping topics to
    subscriber queues.  Publishing fans-out each message to every queue
    registered for that topic.

    Parameters
    ----------
    max_queue_size:
        Upper bound on each subscriber queue.  ``0`` means unbounded.
    strict_topics:
        When ``True`` (the default), only topics in :data:`TOPICS` are
        accepted; publishing or subscribing to an unknown topic raises
        ``ValueError``.  Set to ``False`` during testing to allow ad-hoc
        topics.
    """

    def __init__(
        self,
        max_queue_size: int = 0,
        strict_topics: bool = True,
    ) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[Envelope]]] = {}
        self._max_queue_size = max_queue_size
        self._strict_topics = strict_topics

        # Counters for observability
        self._published: dict[str, int] = {}
        self._dropped: dict[str, int] = {}

        logger.info(
            "MessageBus initialised  strict_topics=%s  max_queue_size=%d",
            strict_topics,
            max_queue_size,
        )

    # ------------------------------------------------------------------
    # Topic validation
    # ------------------------------------------------------------------

    def _validate_topic(self, topic: str) -> None:
        """Raise ``ValueError`` if *topic* is not in the canonical set."""
        if self._strict_topics and topic not in TOPICS:
            raise ValueError(
                f"Unknown topic {topic!r}.  "
                f"Valid topics: {sorted(TOPICS)}.  "
                f"Set strict_topics=False to allow arbitrary topics."
            )

    # ------------------------------------------------------------------
    # Subscribe / unsubscribe
    # ------------------------------------------------------------------

    def subscribe(self, topic: str) -> asyncio.Queue[Envelope]:
        """Create and return a new subscriber queue for *topic*.

        Parameters
        ----------
        topic:
            The topic to subscribe to.

        Returns
        -------
        asyncio.Queue[Envelope]
            A dedicated queue that will receive every message published
            to *topic* after this call.
        """
        self._validate_topic(topic)
        queue: asyncio.Queue[Envelope] = asyncio.Queue(
            maxsize=self._max_queue_size,
        )
        self._subscribers.setdefault(topic, []).append(queue)
        logger.debug(
            "New subscriber for topic=%s  total_subscribers=%d",
            topic,
            len(self._subscribers[topic]),
        )
        return queue

    def unsubscribe(self, topic: str, queue: asyncio.Queue[Envelope]) -> None:
        """Remove a previously-registered subscriber queue.

        It is safe to call this even if the queue has already been removed.

        Parameters
        ----------
        topic:
            The topic the queue was subscribed to.
        queue:
            The queue returned by :meth:`subscribe`.
        """
        self._validate_topic(topic)
        queues = self._subscribers.get(topic)
        if queues is None:
            return
        try:
            queues.remove(queue)
            logger.debug(
                "Unsubscribed from topic=%s  remaining=%d",
                topic,
                len(queues),
            )
        except ValueError:
            pass  # already removed -- harmless

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(
        self,
        topic: str,
        message: Any,
        *,
        publisher: str = "",
    ) -> int:
        """Fan-out *message* to every subscriber queue for *topic*.

        The message is wrapped in an :class:`Envelope` before delivery.

        Parameters
        ----------
        topic:
            The topic to publish on.
        message:
            Arbitrary payload (dict, dataclass, primitive, ...).
        publisher:
            Optional name / id of the publishing agent.

        Returns
        -------
        int
            Number of subscriber queues that received the message.
        """
        self._validate_topic(topic)

        envelope = Envelope(
            topic=topic,
            payload=message,
            publisher=publisher,
        )

        queues = self._subscribers.get(topic, [])
        delivered = 0

        for q in queues:
            try:
                q.put_nowait(envelope)
                delivered += 1
            except asyncio.QueueFull:
                self._dropped[topic] = self._dropped.get(topic, 0) + 1
                logger.warning(
                    "Dropped message on topic=%s (queue full)", topic
                )

        self._published[topic] = self._published.get(topic, 0) + delivered
        return delivered

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def subscriber_count(self, topic: str) -> int:
        """Return the number of active subscribers for *topic*."""
        return len(self._subscribers.get(topic, []))

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Return per-topic publish/drop counters.

        Returns
        -------
        dict
            ``{"signal": {"published": 42, "dropped": 0, "subscribers": 3}, ...}``
        """
        all_topics = sorted(
            set(self._published) | set(self._dropped) | set(self._subscribers)
        )
        stats: dict[str, dict[str, int]] = {}
        for topic in all_topics:
            stats[topic] = {
                "published": self._published.get(topic, 0),
                "dropped": self._dropped.get(topic, 0),
                "subscribers": len(self._subscribers.get(topic, [])),
            }
        return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all subscribers and reset counters."""
        self._subscribers.clear()
        self._published.clear()
        self._dropped.clear()
        logger.info("MessageBus cleared -- all subscribers removed")
