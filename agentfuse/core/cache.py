"""
TwoTierCacheMiddleware — Redis L1 exact match + FAISS L2 semantic search.

L1: SHA-256 hash lookup in Redis (sub-ms). Falls back to local TTLCache if Redis unavailable.
L2: FAISS IndexFlatIP with redis/langcache-embed-v2 embeddings (2-5ms).

CRITICAL RULES:
- tool_use queries NEVER go through L2 semantic search
- L2 post-filter MUST check model_prefix matches (no cross-model)
- temperature > 0.5 → never cache
- Side-effect tools → never cache
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agentfuse.core.keys import build_l1_cache_key, extract_semantic_content, build_l2_metadata_filter

logger = logging.getLogger(__name__)

SIDE_EFFECT_TOOLS = frozenset({
    "send_email", "send_message", "create_ticket", "execute_trade",
    "delete_file", "update_database", "post_comment", "deploy",
    "publish", "transfer", "cancel", "submit_order",
})

NEVER_CACHE_CONDITIONS = [
    "temperature > 0.5",
    "side-effect tools present",
]


@dataclass
class CacheHit:
    tier: int       # 1 = L1 exact, 2 = L2 semantic
    response: str
    cost: float = 0.0
    similarity: float = 1.0


@dataclass
class CacheMiss:
    reason: str = ""


@dataclass
class _L2Entry:
    """Metadata stored alongside each FAISS vector."""
    cache_key: str
    model: str
    model_prefix: str
    has_tools: bool
    response: str
    stored_at: float = field(default_factory=time.time)


class TwoTierCacheMiddleware:
    """
    Two-tier cache: Redis L1 (exact) + FAISS L2 (semantic).
    Falls back to local TTLCache when Redis is unavailable.
    """

    TIER2_HIGH_SIM_THRESHOLD = 0.92
    TIER2_ADAPT_THRESHOLD = 0.88
    DEFAULT_TTL = 86400  # 24 hours
    TTL_JITTER_PCT = 0.10  # ±10%

    def __init__(
        self,
        redis_url: Optional[str] = None,
        embedding_model: str = "redis/langcache-embed-v2",
        faiss_index_size: int = 768,
        max_l2_entries: int = 100_000,
    ):
        self._embedding_model_name = embedding_model
        self._faiss_dim = faiss_index_size
        self._max_l2_entries = max_l2_entries
        self._embedder = None
        self._embedder_lock = threading.Lock()

        # L1: Redis primary, local TTLCache fallback
        self._redis = None
        if redis_url:
            try:
                import redis
                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except Exception as e:
                logger.warning("Redis unavailable, using local L1 cache: %s", e)
                self._redis = None

        # Local L1 fallback
        from cachetools import TTLCache
        self._local_l1 = TTLCache(maxsize=10_000, ttl=self.DEFAULT_TTL)
        self._local_l1_lock = threading.Lock()

        # L2: FAISS index
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(self._faiss_dim)
        except ImportError:
            logger.warning("faiss-cpu not installed, L2 semantic cache disabled")
            self._faiss_index = None

        self._faiss_metadata: list[_L2Entry] = []
        self._faiss_vectors: list[np.ndarray] = []  # store vectors for re-indexing on eviction
        self._faiss_lock = threading.Lock()

        # Thresholds for backward-compatible check() method
        self._direct_threshold = 0.85
        self._adapt_threshold = 0.70

    def _get_embedder(self):
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    from sentence_transformers import SentenceTransformer
                    self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _embed(self, text: str) -> np.ndarray:
        """Returns L2-normalized numpy vector."""
        model = self._get_embedder()
        vec = model.encode([text], normalize_embeddings=True)
        return vec[0].astype(np.float32)

    def _should_skip_cache(self, temperature: float, tools: Optional[list] = None) -> Optional[str]:
        """Returns skip reason if this request should never be cached."""
        if temperature > 0.5:
            return "temperature > 0.5"
        if tools:
            tool_names = set()
            for t in tools:
                if isinstance(t, dict):
                    func = t.get("function", {})
                    name = func.get("name", "") if isinstance(func, dict) else ""
                    tool_names.add(name)
            if tool_names & SIDE_EFFECT_TOOLS:
                return "side-effect tools present"
        return None

    def lookup(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        tools: Optional[list] = None,
        tenant_id: Optional[str] = None,
    ) -> CacheHit | CacheMiss:
        """Check L1 then L2. Returns CacheHit or CacheMiss."""
        skip = self._should_skip_cache(temperature, tools)
        if skip:
            return CacheMiss(reason=skip)

        # L1 exact match
        l1_key = build_l1_cache_key(model, messages, temperature, tools, tenant_id=tenant_id)
        l1_result = self._l1_get(l1_key)
        if l1_result is not None:
            return CacheHit(tier=1, response=l1_result, similarity=1.0)

        # L2 semantic search — NEVER for tool_use queries
        if tools:
            return CacheMiss(reason="tool_use queries skip L2")

        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return CacheMiss(reason="L2 index empty")

        try:
            semantic_text = extract_semantic_content(messages)
            if not semantic_text.strip():
                return CacheMiss(reason="no semantic content")

            meta_filter = build_l2_metadata_filter(model, tools)
            vec = self._embed(semantic_text).reshape(1, -1)

            with self._faiss_lock:
                k = min(10, self._faiss_index.ntotal)
                scores, indices = self._faiss_index.search(vec, k)

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._faiss_metadata):
                    continue
                entry = self._faiss_metadata[idx]

                # Post-filter: model prefix must match
                if entry.model_prefix != meta_filter["model_prefix"]:
                    continue

                if score >= self.TIER2_HIGH_SIM_THRESHOLD:
                    return CacheHit(tier=2, response=entry.response, similarity=float(score))

            return CacheMiss(reason="no L2 match above threshold")

        except Exception as e:
            logger.warning("L2 cache lookup failed: %s", e)
            return CacheMiss(reason=f"L2 error: {e}")

    def store(
        self,
        model: str,
        messages: list[dict],
        response: str,
        temperature: float = 0.0,
        tools: Optional[list] = None,
        tenant_id: Optional[str] = None,
    ):
        """Store in L1 and (if no tools) L2."""
        if not response or not response.strip():
            return  # Never cache empty responses

        skip = self._should_skip_cache(temperature, tools)
        if skip:
            return

        # L1 store
        l1_key = build_l1_cache_key(model, messages, temperature, tools, tenant_id=tenant_id)
        ttl = self._jittered_ttl()
        self._l1_set(l1_key, response, ttl)

        # L2 store — skip for tool_use
        if tools:
            return

        if self._faiss_index is None:
            return

        try:
            semantic_text = extract_semantic_content(messages)
            if not semantic_text.strip():
                return

            meta_filter = build_l2_metadata_filter(model, tools)
            vec = self._embed(semantic_text).reshape(1, -1)

            entry = _L2Entry(
                cache_key=l1_key,
                model=model,
                model_prefix=meta_filter["model_prefix"],
                has_tools=meta_filter["has_tools"],
                response=response,
            )

            with self._faiss_lock:
                # LRU eviction if at capacity
                if len(self._faiss_metadata) >= self._max_l2_entries:
                    self._evict_oldest_l2()

                self._faiss_index.add(vec)
                self._faiss_metadata.append(entry)
                self._faiss_vectors.append(vec.flatten())

        except Exception as e:
            logger.warning("L2 cache store failed: %s", e)

    def _evict_oldest_l2(self):
        """Remove oldest 10% of L2 entries and rebuild FAISS index."""
        n = max(1, len(self._faiss_metadata) // 10)
        import faiss

        self._faiss_metadata = self._faiss_metadata[n:]
        self._faiss_vectors = self._faiss_vectors[n:]

        new_index = faiss.IndexFlatIP(self._faiss_dim)
        if self._faiss_vectors:
            vecs = np.stack(self._faiss_vectors).astype(np.float32)
            new_index.add(vecs)
        self._faiss_index = new_index

    # --- L1 helpers ---

    def _l1_get(self, key: str) -> Optional[str]:
        """Try Redis first, fall back to local cache."""
        if self._redis:
            try:
                val = self._redis.get(key)
                if val is not None:
                    return val
            except Exception:
                pass

        with self._local_l1_lock:
            return self._local_l1.get(key)

    def _l1_set(self, key: str, value: str, ttl: int = None):
        """Store in Redis and local cache."""
        ttl = ttl or self.DEFAULT_TTL
        if self._redis:
            try:
                self._redis.setex(key, ttl, value)
            except Exception:
                pass

        with self._local_l1_lock:
            self._local_l1[key] = value

    def _jittered_ttl(self) -> int:
        """24h TTL with ±10% jitter to prevent thundering herd."""
        jitter = random.uniform(-self.TTL_JITTER_PCT, self.TTL_JITTER_PCT)
        return int(self.DEFAULT_TTL * (1 + jitter))

    def get_stats(self) -> dict:
        """Return cache statistics for observability."""
        return {
            "l1_local_size": len(self._local_l1),
            "l2_entries": len(self._faiss_metadata),
            "l2_index_total": self._faiss_index.ntotal if self._faiss_index else 0,
            "redis_connected": self._redis is not None,
            "embedding_model": self._embedding_model_name,
        }

    def save_l2_index(self, path: str):
        """Persist FAISS index and metadata to disk for recovery."""
        import faiss
        import json
        if self._faiss_index is None:
            return
        with self._faiss_lock:
            faiss.write_index(self._faiss_index, path + ".faiss")
            meta = [
                {"cache_key": e.cache_key, "model": e.model,
                 "model_prefix": e.model_prefix, "has_tools": e.has_tools,
                 "response": e.response, "stored_at": e.stored_at}
                for e in self._faiss_metadata
            ]
            with open(path + ".meta.json", "w") as f:
                json.dump(meta, f)

    def load_l2_index(self, path: str):
        """Load FAISS index and metadata from disk."""
        import faiss
        import json
        import os
        faiss_path = path + ".faiss"
        meta_path = path + ".meta.json"
        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            return False
        with self._faiss_lock:
            self._faiss_index = faiss.read_index(faiss_path)
            with open(meta_path) as f:
                meta_list = json.load(f)
            self._faiss_metadata = [
                _L2Entry(**m) for m in meta_list
            ]
            self._faiss_vectors = []  # Vectors are in the FAISS index already
        return True


    # --- Backward compatibility API ---

    def check(self, cache_key: str, model: str, budget_engine=None) -> CacheHit | CacheMiss:
        """Old API: check by pre-built cache key string."""
        # L1 check using the provided key directly
        l1_result = self._l1_get(cache_key)
        if l1_result is not None:
            return CacheHit(tier=1, response=l1_result, cost=0.0)

        # L2 semantic search using the cache_key as text
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return CacheMiss(reason="L2 index empty")

        try:
            meta_filter = build_l2_metadata_filter(model)
            vec = self._embed(cache_key).reshape(1, -1)

            with self._faiss_lock:
                k = min(10, self._faiss_index.ntotal)
                scores, indices = self._faiss_index.search(vec, k)

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._faiss_metadata):
                    continue
                entry = self._faiss_metadata[idx]
                if entry.model_prefix != meta_filter["model_prefix"]:
                    continue
                if score >= self._direct_threshold:
                    return CacheHit(tier=1, response=entry.response, cost=0.0)
                if score >= self._adapt_threshold:
                    return CacheHit(tier=2, response=entry.response, cost=0.0)

            return CacheMiss(reason="no L2 match")
        except Exception as e:
            logger.warning("Cache check failed: %s", e)
            return CacheMiss(reason=str(e))

    def store_compat(self, cache_key: str, response: str, model: str):
        """Old API: store by pre-built cache key string."""
        if not response or not response.strip():
            return  # Never cache empty responses
        # L1 store
        self._l1_set(cache_key, response)

        # L2 store
        if self._faiss_index is None:
            return
        try:
            meta_filter = build_l2_metadata_filter(model)
            vec = self._embed(cache_key).reshape(1, -1)
            entry = _L2Entry(
                cache_key=cache_key,
                model=model,
                model_prefix=meta_filter["model_prefix"],
                has_tools=False,
                response=response,
            )
            with self._faiss_lock:
                self._faiss_index.add(vec)
                self._faiss_metadata.append(entry)
                self._faiss_vectors.append(vec.flatten())
        except Exception as e:
            logger.warning("Cache store failed: %s", e)

    # Class-level defaults (also set in __init__ for safety)
    _direct_threshold = 0.85
    _adapt_threshold = 0.70


# Make store_compat available as the old .store(key, response, model) signature
_orig_store = TwoTierCacheMiddleware.store

def _flexible_store(self, *args, **kwargs):
    """Dispatch to new or old store API based on argument types."""
    if args and isinstance(args[0], str) and len(args) >= 2 and isinstance(args[1], str):
        # Old API: store(cache_key: str, response: str, model: str)
        return self.store_compat(*args[:3])
    return _orig_store(self, *args, **kwargs)

TwoTierCacheMiddleware.store = _flexible_store


# Backward compatibility alias
CacheMiddleware = TwoTierCacheMiddleware
