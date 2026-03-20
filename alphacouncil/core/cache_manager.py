"""Tiered caching system for AlphaCouncil.

Four cache tiers with increasing latency:

* **L0** -- in-process LRU (``cachetools.TTLCache``), sub-millisecond.
* **L1** -- on-disk ``diskcache.Cache``, typically < 5 ms.
* **L2** -- SQLite via SQLAlchemy, typically < 20 ms.
* **L3** -- upstream API calls (200--2000 ms), executed only on full miss.

The :class:`TieredCache` class exposes an ``async get / set`` interface so
callers can treat it as a single logical cache and never worry about which
tier served the value.

Usage::

    from alphacouncil.core.cache_manager import TieredCache

    cache = TieredCache()
    await cache.set("INFY:price", 1456.30, "price")
    val = await cache.get("INFY:price", "price")        # -> 1456.30
    cache.invalidate("INFY:price")
    stats = cache.get_stats()
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Final

from cachetools import TTLCache
from diskcache import Cache as DiskCache
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, Text, func
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL categories (seconds)
# ---------------------------------------------------------------------------

TTL_CATEGORIES: Final[dict[str, int]] = {
    "price": 60,           # real-time / near-real-time prices
    "fundamental": 86_400, # daily fundamentals
    "macro": 21_600,       # macro data refreshes ~4x/day
    "sentiment": 900,      # sentiment refreshes every 15 min
}

DEFAULT_TTL: Final[int] = 300  # fallback if category unknown

# ---------------------------------------------------------------------------
# L2 SQLite table (private to this module)
# ---------------------------------------------------------------------------

_l2_metadata = MetaData()

_cache_table = Table(
    "cache_store",
    _l2_metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cache_key", String(512), nullable=False, unique=True, index=True),
    Column("value_blob", Text, nullable=False),      # base-64 encoded pickle
    Column("ttl_category", String(32), nullable=False),
    Column("expires_at", Float, nullable=False),      # epoch seconds
    Column("created_at", DateTime, server_default=func.now(), nullable=False),
)


def _ttl_for(category: str) -> int:
    """Resolve a TTL category name to seconds."""
    return TTL_CATEGORIES.get(category, DEFAULT_TTL)


# ---------------------------------------------------------------------------
# TieredCache
# ---------------------------------------------------------------------------

class TieredCache:
    """Multi-tier cache: L0 (memory) -> L1 (disk) -> L2 (SQLite).

    Parameters
    ----------
    cache_dir:
        Root directory for on-disk caches.  ``L1`` lives under
        ``<cache_dir>/diskcache`` and ``L2`` uses an SQLite file at
        ``<cache_dir>/l2_cache.db``.
    l0_maxsize:
        Maximum number of entries in the in-memory LRU cache.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        l0_maxsize: int = 1_000,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # -- L0: in-memory TTL caches (one per category) --
        self._l0: dict[str, TTLCache[str, Any]] = {}
        self._l0_maxsize = l0_maxsize
        for category, ttl in TTL_CATEGORIES.items():
            self._l0[category] = TTLCache(maxsize=l0_maxsize, ttl=ttl)

        # -- L1: diskcache on-disk --
        l1_path = self._cache_dir / "diskcache"
        l1_path.mkdir(parents=True, exist_ok=True)
        self._l1: DiskCache = DiskCache(
            directory=str(l1_path),
            disk_pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )

        # -- L2: SQLite via SQLAlchemy --
        l2_db_path = self._cache_dir / "l2_cache.db"
        self._l2_engine: Engine = create_engine(
            f"sqlite:///{l2_db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        _l2_metadata.create_all(self._l2_engine)

        # -- Stats tracking --
        self._hits: dict[str, int] = {"l0": 0, "l1": 0, "l2": 0}
        self._misses: dict[str, int] = {"l0": 0, "l1": 0, "l2": 0}

        # Lock to serialise L2 writes (SQLite is not great with concurrency).
        self._l2_lock = asyncio.Lock()

        logger.info(
            "TieredCache initialised  cache_dir=%s  l0_maxsize=%d",
            self._cache_dir,
            l0_maxsize,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _infer_category(self, key: str) -> str:
        """Infer ttl_category from a cache key prefix like 'technical:...'."""
        prefix = key.split(":")[0].lower() if ":" in key else ""
        # Map common prefixes to known categories
        _PREFIX_MAP = {
            "price": "price",
            "technical": "price",      # technical analysis uses price-speed TTL
            "fundamental": "fundamental",
            "macro": "macro",
            "sentiment": "sentiment",
        }
        return _PREFIX_MAP.get(prefix, "price")

    async def get(self, key: str, ttl_category: str | None = None) -> Any | None:
        """Look up *key* across L0 -> L1 -> L2.

        Returns the cached value or ``None`` on a full miss (which signals
        the caller to perform the L3 API call and then ``set`` the result).

        Parameters
        ----------
        key:
            Unique cache key (e.g. ``"INFY:price:2024-01-15"``).
        ttl_category:
            One of ``"price"``, ``"fundamental"``, ``"macro"``,
            ``"sentiment"``.  If ``None``, inferred from the key prefix.
        """
        if ttl_category is None:
            ttl_category = self._infer_category(key)
        # --- L0 ---
        l0_cache = self._l0.get(ttl_category)
        if l0_cache is not None:
            try:
                value = l0_cache[key]
                self._hits["l0"] += 1
                return value
            except KeyError:
                self._misses["l0"] += 1

        # --- L1 ---
        raw = self._l1.get(key)
        if raw is not None:
            try:
                entry: dict[str, Any] = pickle.loads(raw)  # noqa: S301
                if entry["expires_at"] > time.time():
                    self._hits["l1"] += 1
                    # Promote to L0
                    self._promote_l0(key, entry["value"], ttl_category)
                    return entry["value"]
                else:
                    # Expired -- evict stale entry
                    self._l1.delete(key)
            except Exception:
                logger.debug("L1 deserialisation failed for key=%s", key)
            self._misses["l1"] += 1
        else:
            self._misses["l1"] += 1

        # --- L2 ---
        value = await self._l2_get(key, ttl_category)
        if value is not None:
            self._hits["l2"] += 1
            # Promote to L0 + L1
            self._promote_l0(key, value, ttl_category)
            self._store_l1(key, value, ttl_category)
            return value
        self._misses["l2"] += 1

        return None  # full miss -> caller should do L3 API call

    async def set(self, key: str, value: Any, ttl_category: str | None = None, *, ttl: int | None = None) -> None:
        """Store *value* in L0 and L1 (and optionally L2).

        L2 is written asynchronously to avoid blocking the hot path.

        Parameters
        ----------
        key:
            Unique cache key.
        value:
            The object to cache (must be picklable).
        ttl_category:
            TTL category name.  If ``None``, inferred from the key prefix.
        ttl:
            Explicit TTL in seconds.  If provided, overrides the category-based
            TTL for this entry.  If ``None``, the TTL is determined by
            *ttl_category* as before.
        """
        if ttl_category is None:
            ttl_category = self._infer_category(key)

        if ttl is not None:
            # Explicit TTL: use a one-off L0 entry and explicit L1/L2 expiry
            self._promote_l0_with_ttl(key, value, ttl_category, ttl)
            self._store_l1_with_ttl(key, value, ttl)
            asyncio.ensure_future(self._l2_set_with_ttl(key, value, ttl_category, ttl))
        else:
            self._promote_l0(key, value, ttl_category)
            self._store_l1(key, value, ttl_category)
            asyncio.ensure_future(self._l2_set(key, value, ttl_category))

    def invalidate(self, key: str) -> None:
        """Remove *key* from all tiers (L0, L1, L2)."""
        # L0 -- scan all category caches
        for cache in self._l0.values():
            cache.pop(key, None)

        # L1
        self._l1.delete(key)

        # L2
        with self._l2_engine.begin() as conn:
            conn.execute(
                _cache_table.delete().where(_cache_table.c.cache_key == key)
            )

        logger.debug("Invalidated key=%s from all tiers", key)

    def invalidate_prefix(self, prefix: str) -> None:
        """Remove all keys matching a prefix from all tiers."""
        # L0
        for cache in self._l0.values():
            to_del = [k for k in cache if k.startswith(prefix)]
            for k in to_del:
                cache.pop(k, None)
        # L1
        for k in list(self._l1):
            if isinstance(k, str) and k.startswith(prefix):
                self._l1.delete(k)
        # L2
        try:
            with self._l2_engine.begin() as conn:
                conn.execute(
                    _cache_table.delete().where(
                        _cache_table.c.cache_key.like(f"{prefix}%")
                    )
                )
        except Exception:
            pass
        logger.debug("Invalidated prefix=%s from all tiers", prefix)

    def clear_all(self) -> None:
        """Purge all entries from L0, L1, and L2."""
        # L0
        for cache in self._l0.values():
            cache.clear()
        # L1
        self._l1.clear()
        # L2
        try:
            with self._l2_engine.begin() as conn:
                conn.execute(_cache_table.delete())
        except Exception:
            pass
        # Reset stats
        for tier in self._hits:
            self._hits[tier] = 0
            self._misses[tier] = 0
        logger.info("All cache tiers cleared")

    def get_stats(self) -> dict[str, dict[str, int | float]]:
        """Return per-tier hit/miss counts and hit rates.

        Returns
        -------
        dict
            ``{"l0": {"hits": N, "misses": M, "hit_rate": R}, ...}``
        """
        stats: dict[str, dict[str, int | float]] = {}
        for tier in ("l0", "l1", "l2"):
            hits = self._hits[tier]
            misses = self._misses[tier]
            total = hits + misses
            stats[tier] = {
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
            }
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _promote_l0(self, key: str, value: Any, ttl_category: str) -> None:
        """Write *value* into the L0 in-memory cache."""
        cache = self._l0.get(ttl_category)
        if cache is None:
            # Unknown category -- create an ad-hoc cache with the default TTL
            ttl = _ttl_for(ttl_category)
            cache = TTLCache(maxsize=self._l0_maxsize, ttl=ttl)
            self._l0[ttl_category] = cache
        cache[key] = value

    def _store_l1(self, key: str, value: Any, ttl_category: str) -> None:
        """Serialise and store *value* in L1 diskcache."""
        ttl = _ttl_for(ttl_category)
        expires_at = time.time() + ttl
        blob = pickle.dumps({"value": value, "expires_at": expires_at})
        self._l1.set(key, blob, expire=ttl)

    def _promote_l0_with_ttl(self, key: str, value: Any, ttl_category: str, ttl: int) -> None:
        """Write to L0 with an explicit TTL (creates a one-off TTLCache if needed)."""
        # Use the category cache but the item will naturally expire via L0's category TTL.
        # For precise TTL control we just use the category cache — close enough for L0.
        cache = self._l0.get(ttl_category)
        if cache is None:
            cache = TTLCache(maxsize=self._l0_maxsize, ttl=ttl)
            self._l0[ttl_category] = cache
        cache[key] = value

    def _store_l1_with_ttl(self, key: str, value: Any, ttl: int) -> None:
        """Store in L1 with an explicit TTL in seconds."""
        expires_at = time.time() + ttl
        blob = pickle.dumps({"value": value, "expires_at": expires_at})
        self._l1.set(key, blob, expire=ttl)

    async def _l2_set_with_ttl(self, key: str, value: Any, ttl_category: str, ttl: int) -> None:
        """Write to L2 with an explicit TTL."""
        import base64
        expires_at = time.time() + ttl
        blob_b64 = base64.b64encode(pickle.dumps(value)).decode("ascii")

        async with self._l2_lock:
            loop = asyncio.get_running_loop()

            def _upsert() -> None:
                with self._l2_engine.begin() as conn:
                    conn.execute(
                        _cache_table.delete().where(_cache_table.c.cache_key == key)
                    )
                    conn.execute(
                        _cache_table.insert().values(
                            cache_key=key,
                            value_blob=blob_b64,
                            ttl_category=ttl_category,
                            expires_at=expires_at,
                        )
                    )

            await loop.run_in_executor(None, _upsert)

    async def _l2_get(self, key: str, ttl_category: str) -> Any | None:
        """Retrieve a non-expired value from the L2 SQLite cache."""
        now = time.time()
        loop = asyncio.get_running_loop()

        def _query() -> Any | None:
            with self._l2_engine.connect() as conn:
                row = conn.execute(
                    _cache_table.select().where(
                        _cache_table.c.cache_key == key,
                        _cache_table.c.expires_at > now,
                    )
                ).fetchone()
                if row is None:
                    return None
                try:
                    return pickle.loads(row.value_blob.encode("latin-1"))  # noqa: S301
                except Exception:
                    logger.debug("L2 unpickle failed for key=%s", key)
                    return None

        return await loop.run_in_executor(None, _query)

    async def _l2_set(self, key: str, value: Any, ttl_category: str) -> None:
        """Upsert a value into the L2 SQLite cache."""
        ttl = _ttl_for(ttl_category)
        expires_at = time.time() + ttl
        blob_str = pickle.dumps(value).decode("latin-1")  # store as text

        async with self._l2_lock:
            loop = asyncio.get_running_loop()

            def _upsert() -> None:
                with self._l2_engine.begin() as conn:
                    # Try update first
                    result = conn.execute(
                        _cache_table.update()
                        .where(_cache_table.c.cache_key == key)
                        .values(
                            value_blob=blob_str,
                            ttl_category=ttl_category,
                            expires_at=expires_at,
                        )
                    )
                    if result.rowcount == 0:
                        conn.execute(
                            _cache_table.insert().values(
                                cache_key=key,
                                value_blob=blob_str,
                                ttl_category=ttl_category,
                                expires_at=expires_at,
                            )
                        )

            await loop.run_in_executor(None, _upsert)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources held by L1 and L2."""
        self._l1.close()
        self._l2_engine.dispose()
        logger.info("TieredCache closed")
