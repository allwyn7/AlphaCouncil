"""Tests for performance: profiler, cache, message bus latency, frozen model hashing."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alphacouncil.core.models import (
    Action,
    AgentSignal,
    AgentStatus,
    FundamentalSignal,
    MacroSignal,
    MarketRegime,
    SentimentSignal,
    TechnicalSignal,
)


# ---------------------------------------------------------------------------
# Profiler context manager tests
# ---------------------------------------------------------------------------


class TestProfiler:
    @pytest.mark.asyncio
    async def test_measure_records_time(self):
        from alphacouncil.core.profiler import Profiler

        profiler = Profiler(flush_every=1000, db_url=None)
        async with profiler.measure("test_stage"):
            await asyncio.sleep(0.01)

        stats = profiler.get_stats("test_stage")
        assert stats["count"] == 1.0
        assert stats["p50"] > 0  # should have measured something
        # Should be at least ~10ms = 10,000,000 ns
        assert stats["p50"] > 1_000_000

    @pytest.mark.asyncio
    async def test_measure_multiple_records(self):
        from alphacouncil.core.profiler import Profiler

        profiler = Profiler(flush_every=1000, db_url=None)
        for _ in range(10):
            async with profiler.measure("repeated"):
                pass  # near-zero latency

        stats = profiler.get_stats("repeated")
        assert stats["count"] == 10.0

    @pytest.mark.asyncio
    async def test_measure_accuracy(self):
        """Verify that measured time is close to actual sleep time."""
        from alphacouncil.core.profiler import Profiler

        profiler = Profiler(flush_every=1000, db_url=None)
        async with profiler.measure("accuracy_test"):
            await asyncio.sleep(0.05)  # 50ms

        stats = profiler.get_stats("accuracy_test")
        measured_ms = stats["p50"] / 1_000_000  # ns -> ms
        # Should be roughly 50ms +/- 30ms (async sleep is not perfectly precise)
        assert 20 < measured_ms < 200

    @pytest.mark.asyncio
    async def test_get_all_stats(self):
        from alphacouncil.core.profiler import Profiler

        profiler = Profiler(flush_every=1000, db_url=None)
        async with profiler.measure("stage_a"):
            pass
        async with profiler.measure("stage_b"):
            pass

        all_stats = profiler.get_all_stats()
        assert "stage_a" in all_stats
        assert "stage_b" in all_stats

    @pytest.mark.asyncio
    async def test_unknown_stage_returns_empty(self):
        from alphacouncil.core.profiler import Profiler

        profiler = Profiler(flush_every=1000, db_url=None)
        stats = profiler.get_stats("nonexistent")
        assert stats == {}

    @pytest.mark.asyncio
    async def test_pending_count(self):
        from alphacouncil.core.profiler import Profiler

        profiler = Profiler(flush_every=1000, db_url=None)
        assert profiler.pending == 0
        async with profiler.measure("test"):
            pass
        assert profiler.pending == 1

    def test_compute_percentiles(self):
        from alphacouncil.core.profiler import Profiler

        values = list(range(1, 101))  # 1 to 100
        stats = Profiler._compute_percentiles(values)
        assert stats["min"] == 1.0
        assert stats["max"] == 100.0
        assert stats["count"] == 100.0
        assert 45 <= stats["p50"] <= 55


# ---------------------------------------------------------------------------
# Cache L0 (cachetools) response time tests
# ---------------------------------------------------------------------------


class TestCacheL0:
    def test_cachetools_ttl_cache_speed(self):
        """L0 cache (cachetools TTLCache) should respond in under 1ms."""
        from cachetools import TTLCache

        cache = TTLCache(maxsize=1000, ttl=300)
        # Pre-populate
        for i in range(100):
            cache[f"key_{i}"] = f"value_{i}"

        # Time 1000 reads
        start = time.perf_counter_ns()
        for _ in range(1000):
            _ = cache.get("key_50")
        elapsed_ns = time.perf_counter_ns() - start

        avg_ns = elapsed_ns / 1000
        avg_ms = avg_ns / 1_000_000
        # Each read should be well under 1ms
        assert avg_ms < 1.0, f"Average L0 cache read took {avg_ms:.4f}ms, expected <1ms"

    def test_cachetools_lru_cache_speed(self):
        """LRU cache should also be sub-millisecond."""
        from cachetools import LRUCache

        cache = LRUCache(maxsize=500)
        for i in range(200):
            cache[f"k{i}"] = {"data": i, "values": list(range(10))}

        start = time.perf_counter_ns()
        for _ in range(1000):
            _ = cache.get("k100")
        elapsed_ns = time.perf_counter_ns() - start

        avg_ms = elapsed_ns / 1000 / 1_000_000
        assert avg_ms < 1.0

    def test_cache_hit_returns_correct_value(self):
        from cachetools import TTLCache

        cache = TTLCache(maxsize=100, ttl=300)
        cache["ticker:INFY"] = {"rsi": 45.0, "macd": 1.2}
        result = cache.get("ticker:INFY")
        assert result["rsi"] == 45.0


# ---------------------------------------------------------------------------
# MessageBus publish/subscribe latency tests
# ---------------------------------------------------------------------------


class TestMessageBusLatency:
    @pytest.mark.asyncio
    async def test_publish_subscribe_roundtrip(self):
        from alphacouncil.core.message_bus import MessageBus

        bus = MessageBus(strict_topics=False)
        queue = bus.subscribe("test_topic")
        await bus.publish("test_topic", {"key": "value"})

        envelope = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert envelope.payload == {"key": "value"}
        assert envelope.topic == "test_topic"

    @pytest.mark.asyncio
    async def test_publish_latency_under_1ms(self):
        """Publishing a message should take well under 1ms."""
        from alphacouncil.core.message_bus import MessageBus

        bus = MessageBus(strict_topics=False)
        _q = bus.subscribe("perf_topic")

        latencies = []
        for _ in range(100):
            start = time.perf_counter_ns()
            await bus.publish("perf_topic", {"data": 42})
            elapsed = time.perf_counter_ns() - start
            latencies.append(elapsed)

        avg_ms = np.mean(latencies) / 1_000_000
        assert avg_ms < 1.0, f"Average publish latency {avg_ms:.4f}ms exceeds 1ms"

    @pytest.mark.asyncio
    async def test_fanout_to_multiple_subscribers(self):
        from alphacouncil.core.message_bus import MessageBus

        bus = MessageBus(strict_topics=False)
        q1 = bus.subscribe("multi")
        q2 = bus.subscribe("multi")
        q3 = bus.subscribe("multi")

        delivered = await bus.publish("multi", "hello")
        assert delivered == 3

        for q in [q1, q2, q3]:
            env = await asyncio.wait_for(q.get(), timeout=1.0)
            assert env.payload == "hello"

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self):
        from alphacouncil.core.message_bus import MessageBus

        bus = MessageBus(strict_topics=False)
        q = bus.subscribe("unsub_test")
        bus.unsubscribe("unsub_test", q)

        delivered = await bus.publish("unsub_test", "after_unsub")
        assert delivered == 0

    @pytest.mark.asyncio
    async def test_strict_topic_rejects_unknown(self):
        from alphacouncil.core.message_bus import MessageBus

        bus = MessageBus(strict_topics=True)
        with pytest.raises(ValueError, match="Unknown topic"):
            bus.subscribe("invalid_topic_xyz")

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        from alphacouncil.core.message_bus import MessageBus

        bus = MessageBus(strict_topics=False)
        bus.subscribe("tracked")
        await bus.publish("tracked", "msg1")
        await bus.publish("tracked", "msg2")

        stats = bus.get_stats()
        assert stats["tracked"]["published"] == 2
        assert stats["tracked"]["subscribers"] == 1

    @pytest.mark.asyncio
    async def test_clear_removes_all(self):
        from alphacouncil.core.message_bus import MessageBus

        bus = MessageBus(strict_topics=False)
        bus.subscribe("topic1")
        bus.subscribe("topic2")
        bus.clear()
        assert bus.subscriber_count("topic1") == 0
        assert bus.subscriber_count("topic2") == 0


# ---------------------------------------------------------------------------
# Frozen Pydantic models hashability tests
# ---------------------------------------------------------------------------


class TestFrozenModelHashing:
    def test_agent_signal_hashable(self):
        now = datetime.now(tz=timezone.utc)
        sig = AgentSignal(
            ticker="INFY", action=Action.BUY, conviction=75,
            target_weight=0.05, stop_loss=90.0, take_profit=120.0,
            factor_scores={"alpha": 1.0}, reasoning="test",
            holding_period_days=10, agent_name="test",
            timestamp=now,
        )
        h = hash(sig)
        assert isinstance(h, int)

    def test_agent_signal_usable_as_dict_key(self):
        now = datetime.now(tz=timezone.utc)
        sig = AgentSignal(
            ticker="TCS", action=Action.SELL, conviction=60,
            target_weight=0.03, stop_loss=3000.0, take_profit=4000.0,
            factor_scores={}, reasoning="test",
            holding_period_days=5, agent_name="test",
            timestamp=now,
        )
        d = {sig: "some_value"}
        assert d[sig] == "some_value"

    def test_agent_signal_usable_in_set(self):
        now = datetime.now(tz=timezone.utc)
        sig1 = AgentSignal(
            ticker="INFY", action=Action.BUY, conviction=50,
            target_weight=0.05, stop_loss=90.0, take_profit=120.0,
            factor_scores={}, reasoning="a",
            holding_period_days=5, agent_name="test",
            timestamp=now,
        )
        sig2 = AgentSignal(
            ticker="TCS", action=Action.SELL, conviction=60,
            target_weight=0.03, stop_loss=3000.0, take_profit=4000.0,
            factor_scores={}, reasoning="b",
            holding_period_days=5, agent_name="test",
            timestamp=now,
        )
        s = {sig1, sig2}
        assert len(s) == 2

    def test_technical_signal_hashable(self):
        now = datetime.now(tz=timezone.utc)
        sig = TechnicalSignal(
            ticker="TEST", rsi=50.0, macd=1.0, macd_signal=0.8,
            macd_hist=0.2, roc=5.0, bollinger_upper=110.0,
            bollinger_lower=90.0, bollinger_mid=100.0,
            sma_20=100.0, sma_50=99.0, sma_200=95.0,
            ema_20=100.5, ema_50=99.5, ema_200=95.5,
            adx=25.0, atr=2.0, obv=1000000.0, vwap=100.5,
            volume_ratio=1.2, timestamp=now,
        )
        assert isinstance(hash(sig), int)

    def test_fundamental_signal_hashable(self):
        now = datetime.now(tz=timezone.utc)
        sig = FundamentalSignal(
            ticker="TEST", pe_ratio=20.0, peg_ratio=1.5, pb_ratio=3.0,
            roe=0.15, roa=0.08, debt_to_equity=0.5, fcf=1e9,
            gross_margin=0.3, operating_margin=0.2, net_margin=0.1,
            revenue_growth=0.2, eps_growth=0.15,
            promoter_holding=50.0, fii_holding=20.0, dii_holding=10.0,
            intrinsic_value=1500.0, timestamp=now,
        )
        assert isinstance(hash(sig), int)

    def test_sentiment_signal_hashable(self):
        now = datetime.now(tz=timezone.utc)
        sig = SentimentSignal(
            ticker="TEST", score=0.5, volume=100, trend=0.1,
            keywords=["growth"], source="test",
            timestamp=now,
        )
        assert isinstance(hash(sig), int)

    def test_macro_signal_hashable(self):
        now = datetime.now(tz=timezone.utc)
        sig = MacroSignal(
            repo_rate=6.5, india_cpi=5.0, india_iip=3.0,
            fed_rate=5.25, dxy=104.0, brent_crude=80.0,
            india_vix=14.0, gold_price=60000.0, nifty_level=22000.0,
            fii_net_flow=500.0, dii_net_flow=300.0,
            regime=MarketRegime.BULL_LOW_VOL,
            timestamp=now,
        )
        assert isinstance(hash(sig), int)

    def test_equal_signals_have_same_hash(self):
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        sig1 = AgentSignal(
            ticker="X", action=Action.HOLD, conviction=50,
            target_weight=0.05, stop_loss=1.0, take_profit=2.0,
            factor_scores={}, reasoning="r",
            holding_period_days=1, agent_name="a",
            timestamp=now,
        )
        sig2 = AgentSignal(
            ticker="X", action=Action.HOLD, conviction=50,
            target_weight=0.05, stop_loss=1.0, take_profit=2.0,
            factor_scores={}, reasoning="r",
            holding_period_days=1, agent_name="a",
            timestamp=now,
        )
        assert hash(sig1) == hash(sig2)
        assert sig1 == sig2
