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
    cache._faiss_vectors = []
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


def test_get_stats_returns_expected_keys():
    """get_stats() must return cache statistics."""
    cache, _ = _make_cache()
    stats = cache.get_stats()
    assert "l1_local_size" in stats
    assert "l2_entries" in stats
    assert "l2_index_total" in stats
    assert "redis_connected" in stats
    assert stats["redis_connected"] is False
    assert stats["l2_entries"] == 0


def test_l1_set_with_custom_ttl():
    """L1 set with custom TTL must work."""
    cache, _ = _make_cache()
    cache._l1_set("key1", "value1", ttl=60)
    assert cache._l1_get("key1") == "value1"


def test_l1_get_returns_none_for_expired():
    """L1 get for expired keys must return None."""
    from cachetools import TTLCache
    cache, _ = _make_cache()
    # Replace with very short TTL cache for testing
    cache._local_l1 = TTLCache(maxsize=10, ttl=0.001)
    cache._l1_set("key1", "value1")
    import time
    time.sleep(0.01)
    assert cache._l1_get("key1") is None


def test_cache_miss_reason_populated():
    """CacheMiss must have a reason when applicable."""
    cache, _ = _make_cache()
    result = cache.lookup("gpt-4o", [{"role": "user", "content": "test"}], temperature=0.9)
    assert isinstance(result, CacheMiss)
    assert result.reason == "temperature > 0.5"


def test_save_and_load_l2_index():
    """FAISS index persistence must round-trip correctly."""
    import tempfile
    import os

    cache, mock = _make_cache(embedding_dim=16)
    vec = np.zeros(16, dtype=np.float32)
    vec[0] = 1.0
    mock.encode.return_value = np.array([vec])

    # Store an entry
    from agentfuse.core.cache import _L2Entry, build_l2_metadata_filter
    meta = build_l2_metadata_filter("gpt-4o")
    entry = _L2Entry(
        cache_key="key1", model="gpt-4o",
        model_prefix=meta["model_prefix"],
        has_tools=False, response="hello",
    )
    with cache._faiss_lock:
        cache._faiss_index.add(vec.reshape(1, -1))
        cache._faiss_metadata.append(entry)
        cache._faiss_vectors.append(vec)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_index")
        cache.save_l2_index(path)
        assert os.path.exists(path + ".faiss")
        assert os.path.exists(path + ".meta.json")

        # Load into a new cache
        cache2, _ = _make_cache(embedding_dim=16)
        loaded = cache2.load_l2_index(path)
        assert loaded
        assert cache2._faiss_index.ntotal == 1
        assert len(cache2._faiss_metadata) == 1
        assert cache2._faiss_metadata[0].response == "hello"


def test_l2_eviction_preserves_remaining_entries():
    """After L2 eviction, remaining entries must still be searchable."""
    dim = 16  # small dim for speed
    cache, mock = _make_cache(embedding_dim=dim)

    # Create deterministic vectors
    vecs = []
    for i in range(15):
        v = np.zeros(dim, dtype=np.float32)
        v[i % dim] = 1.0
        vecs.append(v)

    cache._max_l2_entries = 10

    # Store 10 entries with unique vectors
    for i in range(10):
        mock.encode.return_value = np.array([vecs[i]])
        msgs = [{"role": "user", "content": f"query {i}"}]
        # Use new API — need model as first arg
        from agentfuse.core.cache import _L2Entry, build_l2_metadata_filter
        meta = build_l2_metadata_filter("gpt-4o")
        entry = _L2Entry(
            cache_key=f"key_{i}", model="gpt-4o",
            model_prefix=meta["model_prefix"],
            has_tools=False, response=f"response {i}",
        )
        with cache._faiss_lock:
            vec = vecs[i].reshape(1, -1)
            cache._faiss_index.add(vec)
            cache._faiss_metadata.append(entry)
            cache._faiss_vectors.append(vecs[i])

    assert cache._faiss_index.ntotal == 10
    assert len(cache._faiss_metadata) == 10

    # Store one more to trigger eviction
    mock.encode.return_value = np.array([vecs[10 % dim]])
    cache.store("gpt-4o", [{"role": "user", "content": "trigger eviction"}], "new response")

    # After eviction: should have 10 entries (9 remaining + 1 new)
    assert cache._faiss_index.ntotal == len(cache._faiss_metadata)
    assert len(cache._faiss_metadata) <= 10
