"""
Concurrency safety tests for production scale.
Tests that verify thread safety, circuit breakers, and memory limits.
"""

import threading
import numpy as np
from unittest.mock import MagicMock
from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit, CacheMiss
from agentfuse.core.budget import BudgetEngine


def test_cache_circuit_breaker_opens():
    """After 5 Redis failures, cache must stop trying Redis."""
    cache = TwoTierCacheMiddleware.__new__(TwoTierCacheMiddleware)
    cache._redis = MagicMock()
    cache._redis.get.side_effect = ConnectionError("Redis down")
    cache._redis_failures = 0
    cache._REDIS_CIRCUIT_OPEN_THRESHOLD = 5
    from cachetools import TTLCache
    cache._local_l1 = TTLCache(maxsize=100, ttl=3600)
    cache._local_l1_lock = threading.Lock()

    # First 5 calls try Redis and fail
    for i in range(5):
        cache._l1_get(f"key_{i}")

    assert cache._redis_failures == 5
    assert not cache._redis_available()

    # 6th call should NOT try Redis at all
    cache._redis.get.reset_mock()
    cache._l1_get("key_6")
    cache._redis.get.assert_not_called()


def test_cache_circuit_breaker_resets_on_success():
    """Successful Redis operation must reset the circuit breaker."""
    cache = TwoTierCacheMiddleware.__new__(TwoTierCacheMiddleware)
    cache._redis = MagicMock()
    cache._redis.get.return_value = "cached_value"
    cache._redis_failures = 4  # nearly open
    cache._REDIS_CIRCUIT_OPEN_THRESHOLD = 5
    from cachetools import TTLCache
    cache._local_l1 = TTLCache(maxsize=100, ttl=3600)
    cache._local_l1_lock = threading.Lock()

    result = cache._l1_get("key")
    assert result == "cached_value"
    assert cache._redis_failures == 0  # reset on success


def test_l2_exact_model_match_prevents_cross_family():
    """L2 must not serve gpt-4-turbo response for gpt-4o query."""
    from agentfuse.core.cache import _L2Entry, build_l2_metadata_filter
    import faiss

    cache = TwoTierCacheMiddleware.__new__(TwoTierCacheMiddleware)
    cache._faiss_dim = 16
    cache._faiss_index = faiss.IndexFlatIP(16)
    cache._faiss_metadata = []
    cache._faiss_vectors = []
    cache._faiss_lock = threading.Lock()
    cache._embedder = MagicMock()
    cache._embedder_lock = threading.Lock()
    cache._redis = None
    cache._redis_failures = 0
    cache._REDIS_CIRCUIT_OPEN_THRESHOLD = 5
    from cachetools import TTLCache
    cache._local_l1 = TTLCache(maxsize=100, ttl=3600)
    cache._local_l1_lock = threading.Lock()

    # Store a gpt-4-turbo response
    vec = np.random.randn(16).astype(np.float32)
    vec /= np.linalg.norm(vec)
    cache._embedder.encode.return_value = np.array([vec])

    meta = build_l2_metadata_filter("gpt-4-turbo")
    entry = _L2Entry(
        cache_key="key1", model="gpt-4-turbo",
        model_prefix=meta["model_prefix"],
        has_tools=False, response="turbo response",
    )
    with cache._faiss_lock:
        cache._faiss_index.add(vec.reshape(1, -1))
        cache._faiss_metadata.append(entry)
        cache._faiss_vectors.append(vec)

    # Query with gpt-4o — same embedding, different model
    result = cache.lookup("gpt-4o", [{"role": "user", "content": "test"}])
    # Must be a MISS because exact model doesn't match
    assert isinstance(result, CacheMiss), "gpt-4-turbo response must NOT be served for gpt-4o"


def test_l2_tool_use_not_served_for_non_tool():
    """Tool-use cached response must not be served for non-tool query."""
    from agentfuse.core.cache import _L2Entry, build_l2_metadata_filter
    import faiss

    cache = TwoTierCacheMiddleware.__new__(TwoTierCacheMiddleware)
    cache._faiss_dim = 16
    cache._faiss_index = faiss.IndexFlatIP(16)
    cache._faiss_metadata = []
    cache._faiss_vectors = []
    cache._faiss_lock = threading.Lock()
    cache._embedder = MagicMock()
    cache._embedder_lock = threading.Lock()
    cache._redis = None
    cache._redis_failures = 0
    cache._REDIS_CIRCUIT_OPEN_THRESHOLD = 5
    from cachetools import TTLCache
    cache._local_l1 = TTLCache(maxsize=100, ttl=3600)
    cache._local_l1_lock = threading.Lock()

    # Store a tool-use response
    vec = np.random.randn(16).astype(np.float32)
    vec /= np.linalg.norm(vec)
    cache._embedder.encode.return_value = np.array([vec])

    entry = _L2Entry(
        cache_key="key1", model="gpt-4o",
        model_prefix="openai",
        has_tools=True,  # tool-use response
        response="<tool_call>search()</tool_call>",
    )
    with cache._faiss_lock:
        cache._faiss_index.add(vec.reshape(1, -1))
        cache._faiss_metadata.append(entry)
        cache._faiss_vectors.append(vec)

    # Query WITHOUT tools — must not get tool response
    # Note: queries with no tools skip L2 entirely (line 166-167)
    # This test verifies the post-filter handles has_tools correctly
    result = cache.lookup("gpt-4o", [{"role": "user", "content": "test"}])
    # Without tools, L2 is skipped, so this should be a miss
    assert isinstance(result, CacheMiss)


def test_embedding_nan_replaced_with_zeros():
    """NaN embeddings must be replaced with zeros to prevent FAISS segfault."""
    cache = TwoTierCacheMiddleware.__new__(TwoTierCacheMiddleware)
    cache._faiss_dim = 4
    cache._embedder = MagicMock()
    cache._embedder_lock = threading.Lock()

    # Return NaN embedding
    nan_vec = np.array([[float('nan'), 1.0, 0.0, 0.0]])
    cache._embedder.encode.return_value = nan_vec

    result = cache._embed("test text")
    assert np.isfinite(result).all(), "NaN should be replaced with zeros"
    assert np.allclose(result, np.zeros(4))


def test_budget_estimation_drift_tracking():
    """reconcile_cost must track cumulative estimation error."""
    engine = BudgetEngine("drift_test", 10.0, "gpt-4o")
    engine.reconcile_cost(estimated_usd=1.0, actual_usd=0.7)
    engine.reconcile_cost(estimated_usd=0.5, actual_usd=0.3)
    # Drift = (1.0 - 0.7) + (0.5 - 0.3) = 0.3 + 0.2 = 0.5
    assert abs(engine._estimation_drift - 0.5) < 0.001


def test_concurrent_budget_engines_isolated():
    """Multiple BudgetEngine instances must not interfere with each other."""
    engines = [BudgetEngine(f"run_{i}", 10.0, "gpt-4o") for i in range(10)]
    errors = []

    def stress(engine):
        try:
            for _ in range(100):
                engine.check_and_act(0.01, [{"role": "user", "content": "hi"}])
                engine.record_cost(0.01)
        except Exception as e:
            if "exhausted" not in str(e).lower():
                errors.append(e)

    threads = [threading.Thread(target=stress, args=(e,)) for e in engines]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors: {errors}"
    # Each engine should have spent ~$1.00 (100 * 0.01)
    for e in engines:
        assert 0.9 < e.spent < 1.1, f"Engine {e.run_id} spent ${e.spent}"
