"""
Phase 2 — Integration tests for two-tier cache (requires Redis on localhost:6379).

Skip these tests if Redis is not available.
"""

import pytest

try:
    import redis
    _r = redis.Redis(host="localhost", port=6379)
    _r.ping()
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available on localhost:6379")


def _make_redis_cache():
    from agentfuse.core.cache import TwoTierCacheMiddleware
    return TwoTierCacheMiddleware(redis_url="redis://localhost:6379")


def test_hit_rate_above_87_5_pct():
    """Store 8 prompts, check all 8, assert >= 7/8 hits."""
    cache = _make_redis_cache()
    prompts = [
        ("What is Python?", "Python is a programming language."),
        ("What is an API?", "An API is an application programming interface."),
        ("What is machine learning?", "ML is a subset of AI."),
        ("What is a neural network?", "A neural network mimics the brain."),
        ("What is gradient descent?", "An optimization algorithm."),
        ("What is a transformer?", "An attention-based architecture."),
        ("What is tokenization?", "Splitting text into tokens."),
        ("What is an embedding?", "A numeric vector representation."),
    ]

    for prompt, response in prompts:
        msgs = [{"role": "user", "content": prompt}]
        cache.store("gpt-4o", msgs, response)

    hits = 0
    for prompt, _ in prompts:
        msgs = [{"role": "user", "content": prompt}]
        result = cache.lookup("gpt-4o", msgs)
        from agentfuse.core.cache import CacheHit
        if isinstance(result, CacheHit):
            hits += 1

    assert hits >= 7, f"Hit rate {hits}/8 is below 87.5% threshold"


def test_cross_model_never_hits():
    """Store gpt-4o response, query with claude → must be a miss."""
    cache = _make_redis_cache()
    msgs = [{"role": "user", "content": "What is the capital of Japan?"}]
    cache.store("gpt-4o", msgs, "Tokyo")

    from agentfuse.core.cache import CacheMiss
    result = cache.lookup("claude-sonnet-4-6", msgs)
    assert isinstance(result, CacheMiss), "Cross-model contamination detected"


def test_ttl_jitter_applied():
    """Consecutive stores should have different TTLs."""
    cache = _make_redis_cache()
    ttls = {cache._jittered_ttl() for _ in range(20)}
    assert len(ttls) > 1, "No TTL jitter detected"
