"""
Phase 2 — Two-tier cache middleware unit tests (no Redis required).
"""

from unittest.mock import MagicMock, patch
import numpy as np

from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit, CacheMiss


def _make_cache(embedding_dim=768):
    """Create a cache with a mock embedder to avoid loading real model."""
    cache = TwoTierCacheMiddleware.__new__(TwoTierCacheMiddleware)
    cache._embedding_model_name = "mock"
    cache._faiss_dim = embedding_dim
    cache._max_l2_entries = 100_000
    cache._embedder = None
    cache._embedder_lock = __import__("threading").Lock()
    cache._redis = None

    from cachetools import TTLCache
    cache._local_l1 = TTLCache(maxsize=10_000, ttl=86400)
    cache._local_l1_lock = __import__("threading").Lock()

    import faiss
    cache._faiss_index = faiss.IndexFlatIP(embedding_dim)
    cache._faiss_metadata = []
    cache._faiss_lock = __import__("threading").Lock()

    # Mock embedder that returns deterministic vectors
    mock_embedder = MagicMock()
    cache._embedder = mock_embedder
    return cache, mock_embedder


def _random_vec(dim=768, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def test_l1_hit_on_identical_request():
    """L1 must hit on byte-identical request."""
    cache, _ = _make_cache()
    model = "gpt-4o"
    msgs = [{"role": "user", "content": "hello"}]

    # Store
    cache._l1_set(
        cache.lookup.__code__  # just need a key
        and __import__("agentfuse.core.keys", fromlist=["build_l1_cache_key"]).build_l1_cache_key(model, msgs),
        "response text"
    )
    # Actually let's use the store method properly
    cache2, mock = _make_cache()
    mock.encode.return_value = np.array([_random_vec()])
    cache2.store(model, msgs, "hello response")
    result = cache2.lookup(model, msgs)
    assert isinstance(result, CacheHit)
    assert result.tier == 1
    assert result.response == "hello response"


def test_l2_miss_on_different_model_prefix():
    """Same text but different model must miss on L2 (model prefix filter)."""
    cache, mock = _make_cache()
    vec = _random_vec(seed=1)
    mock.encode.return_value = np.array([vec])

    # Store with gpt-4o
    cache.store("gpt-4o", [{"role": "user", "content": "capital of France?"}], "Paris")

    # Look up with claude — should miss even though embedding is identical
    result = cache.lookup("claude-sonnet-4-6", [{"role": "user", "content": "capital of France?"}])
    # L1 should miss (different model in key), L2 should miss (different prefix)
    assert isinstance(result, CacheMiss)


def test_tool_use_never_reaches_l2():
    """Queries with tools must skip L2 semantic search."""
    cache, mock = _make_cache()
    vec = _random_vec(seed=2)
    mock.encode.return_value = np.array([vec])

    tools = [{"function": {"name": "search_db"}}]
    # Store without tools first
    cache.store("gpt-4o", [{"role": "user", "content": "find info"}], "result")

    # Lookup with tools — should not hit L2
    result = cache.lookup("gpt-4o", [{"role": "user", "content": "find info"}], tools=tools)
    # L1 miss (tools in key), and L2 skipped
    assert isinstance(result, CacheMiss)
    assert "tool_use" in result.reason or "L2" not in result.reason


def test_high_temperature_skips_cache():
    """temperature=0.8 must always miss (never cache)."""
    cache, mock = _make_cache()
    mock.encode.return_value = np.array([_random_vec()])

    # Store at temp=0.8 — should be skipped
    cache.store("gpt-4o", [{"role": "user", "content": "hi"}], "response", temperature=0.8)

    # Lookup at temp=0.8 — should be skipped
    result = cache.lookup("gpt-4o", [{"role": "user", "content": "hi"}], temperature=0.8)
    assert isinstance(result, CacheMiss)
    assert "temperature" in result.reason


def test_side_effect_tool_skips_cache():
    """send_email in tools must skip cache entirely."""
    cache, mock = _make_cache()
    mock.encode.return_value = np.array([_random_vec()])

    tools = [{"function": {"name": "send_email"}}]
    cache.store("gpt-4o", [{"role": "user", "content": "send"}], "sent", tools=tools)
    result = cache.lookup("gpt-4o", [{"role": "user", "content": "send"}], tools=tools)
    assert isinstance(result, CacheMiss)
    assert "side-effect" in result.reason


def test_jittered_ttl_varies():
    """Consecutive TTL calculations should differ (jitter applied)."""
    cache, _ = _make_cache()
    ttls = {cache._jittered_ttl() for _ in range(20)}
    # With 10% jitter, we should get variation
    assert len(ttls) > 1


def test_l1_local_fallback_works():
    """When Redis is None, local TTLCache must serve as L1."""
    cache, _ = _make_cache()
    assert cache._redis is None

    cache._l1_set("test_key", "test_value")
    assert cache._l1_get("test_key") == "test_value"
    assert cache._l1_get("nonexistent") is None
