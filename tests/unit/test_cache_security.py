"""
Tests for cache security — dual-threshold, tenant isolation, poisoning defense.
"""

from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit, CacheMiss


def test_tenant_isolation_l1():
    """L1 cache must be isolated per tenant_id."""
    cache = TwoTierCacheMiddleware()
    msgs = [{"role": "user", "content": "shared question for tenant test"}]

    # Tenant A stores a response
    cache.store(model="gpt-4o", messages=msgs, response="Tenant A response",
                tenant_id="tenant_a")

    # Tenant A gets their response
    result_a = cache.lookup(model="gpt-4o", messages=msgs, tenant_id="tenant_a")
    assert isinstance(result_a, CacheHit)
    assert result_a.response == "Tenant A response"

    # Tenant B must NOT get Tenant A's response (different L1 key due to tenant_id)
    result_b = cache.lookup(model="gpt-4o", messages=msgs, tenant_id="tenant_b")
    assert isinstance(result_b, CacheMiss)


def test_high_temperature_never_cached():
    """Requests with temperature > 0.5 must never be cached."""
    cache = TwoTierCacheMiddleware()
    msgs = [{"role": "user", "content": "creative writing prompt"}]

    cache.store(model="gpt-4o", messages=msgs, response="Creative response",
                temperature=0.8)

    result = cache.lookup(model="gpt-4o", messages=msgs, temperature=0.8)
    assert isinstance(result, CacheMiss)
    assert "temperature" in result.reason


def test_side_effect_tools_never_cached():
    """Side-effect tools (send_email, etc.) must never be cached."""
    cache = TwoTierCacheMiddleware()
    msgs = [{"role": "user", "content": "send an email"}]
    tools = [{"function": {"name": "send_email", "parameters": {}}}]

    cache.store(model="gpt-4o", messages=msgs, response="Email sent",
                tools=tools)

    result = cache.lookup(model="gpt-4o", messages=msgs, tools=tools)
    assert isinstance(result, CacheMiss)
    assert "side-effect" in result.reason


def test_cross_model_l1_isolation():
    """Different models must not share L1 cache entries."""
    cache = TwoTierCacheMiddleware()
    msgs = [{"role": "user", "content": "cross model test query"}]

    cache.store(model="gpt-4o", messages=msgs, response="GPT-4o answer")

    result = cache.lookup(model="claude-sonnet-4-6", messages=msgs)
    assert isinstance(result, CacheMiss)


def test_empty_response_not_cached():
    """Empty or whitespace responses must not be cached."""
    cache = TwoTierCacheMiddleware()
    msgs = [{"role": "user", "content": "empty test"}]

    cache.store(model="gpt-4o", messages=msgs, response="")
    cache.store(model="gpt-4o", messages=msgs, response="   ")

    result = cache.lookup(model="gpt-4o", messages=msgs)
    assert isinstance(result, CacheMiss)


def test_write_threshold_higher_than_read():
    """Write threshold (0.95) must be higher than read threshold (0.90)."""
    cache = TwoTierCacheMiddleware()
    assert cache.TIER2_WRITE_SIM_THRESHOLD > cache.TIER2_HIGH_SIM_THRESHOLD


def test_jittered_ttl_varies():
    """Consecutive TTLs must have jitter (not all identical)."""
    cache = TwoTierCacheMiddleware()
    ttls = [cache._jittered_ttl() for _ in range(20)]
    unique_ttls = set(ttls)
    assert len(unique_ttls) > 1, "TTL jitter must produce different values"


def test_cache_stats():
    """get_stats must return a dict with expected keys."""
    cache = TwoTierCacheMiddleware()
    stats = cache.get_stats()
    assert "l1_local_size" in stats
    assert "l2_entries" in stats
    assert "l2_index_total" in stats
    assert "redis_connected" in stats
    assert "embedding_model" in stats
