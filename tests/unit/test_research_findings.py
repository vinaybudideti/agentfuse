"""
Tests based on production research findings from LiteLLM, Portkey, Helicone.
These tests verify the fixes and features identified through deep research.
"""

import time
import threading
from agentfuse.core.cache_quality import CacheQualityTracker
from agentfuse.core.gcra_limiter import GCRARateLimiter
from agentfuse.core.cache import TwoTierCacheMiddleware, CacheMiss


# --- Cache Quality (research: cache poisoning defense) ---

def test_quality_score_increases_with_hits():
    tracker = CacheQualityTracker()
    tracker.record_hit("key1", model="gpt-4o")
    tracker.record_hit("key1", model="gpt-4o")
    tracker.record_hit("key1", model="gpt-4o")
    assert tracker.get_score("key1") > 0.5


def test_negative_feedback_lowers_score():
    tracker = CacheQualityTracker()
    tracker.record_hit("key1", model="gpt-4o")
    tracker.record_feedback("key1", positive=False)
    tracker.record_feedback("key1", positive=False)
    assert tracker.get_score("key1") < 0.5


def test_should_invalidate_low_quality():
    tracker = CacheQualityTracker(min_quality=0.3)
    tracker.record_hit("bad_key", model="gpt-4o")
    tracker.record_feedback("bad_key", positive=False)
    tracker.record_feedback("bad_key", positive=False)
    tracker.record_feedback("bad_key", positive=False)
    assert tracker.should_invalidate("bad_key")


def test_invalidate_by_model():
    """Model update must invalidate all entries for that model."""
    tracker = CacheQualityTracker()
    tracker.record_hit("key1", model="gpt-4o")
    tracker.record_hit("key2", model="gpt-4o")
    tracker.record_hit("key3", model="claude-sonnet-4-6")

    removed = tracker.invalidate_by_model("gpt-4o")
    assert removed == 2
    assert tracker.get_score("key3") > 0  # claude entries still exist


def test_cleanup_low_quality():
    tracker = CacheQualityTracker(min_quality=0.3)
    tracker.record_hit("good", model="gpt-4o")
    tracker.record_feedback("good", positive=True)

    tracker.record_hit("bad", model="gpt-4o")
    tracker.record_feedback("bad", positive=False)
    tracker.record_feedback("bad", positive=False)
    tracker.record_feedback("bad", positive=False)

    removed = tracker.cleanup_low_quality()
    assert removed == 1


def test_quality_stats():
    tracker = CacheQualityTracker()
    tracker.record_hit("k1", model="gpt-4o")
    tracker.record_hit("k2", model="gpt-4o")
    stats = tracker.get_stats()
    assert stats["total_entries"] == 2
    assert stats["total_hits"] == 2


# --- GCRA Rate Limiter (research: Helicone uses GCRA) ---

def test_gcra_allows_within_rate():
    """Requests within rate must be allowed."""
    limiter = GCRARateLimiter(rate=100, burst_tolerance=5)
    for _ in range(5):
        assert limiter.check("tenant_1")


def test_gcra_rejects_over_rate():
    """Rapid requests exceeding rate+burst must be rejected."""
    limiter = GCRARateLimiter(rate=10, burst_tolerance=2)
    allowed = 0
    for _ in range(20):
        if limiter.check("tenant_1"):
            allowed += 1
    # Should allow burst + some, but not all 20
    assert allowed < 20
    assert allowed >= 2  # at least burst tolerance


def test_gcra_recovers_over_time():
    """After rate exceeded, should recover after waiting."""
    limiter = GCRARateLimiter(rate=100, burst_tolerance=2)
    # Exhaust burst
    for _ in range(10):
        limiter.check("tenant_1")
    time.sleep(0.05)  # wait for rate recovery
    assert limiter.check("tenant_1")


def test_gcra_per_tenant_isolation():
    """Different tenants must have independent limits."""
    limiter = GCRARateLimiter(rate=5, burst_tolerance=1)
    # Exhaust tenant_a
    for _ in range(10):
        limiter.check("tenant_a")
    # tenant_b should still be fine
    assert limiter.check("tenant_b")


def test_gcra_get_wait_time():
    """Wait time must be non-negative."""
    limiter = GCRARateLimiter(rate=10, burst_tolerance=2)
    for _ in range(10):
        limiter.check("tenant_1")
    wait = limiter.get_wait_time("tenant_1")
    assert wait >= 0


def test_gcra_reset():
    limiter = GCRARateLimiter(rate=5, burst_tolerance=1)
    for _ in range(10):
        limiter.check("tenant_1")
    limiter.reset("tenant_1")
    assert limiter.check("tenant_1")


# --- Cache threshold change (research: 0.90 is optimal) ---

def test_cache_default_threshold_is_090():
    """Default L2 threshold must be 0.90 (research-backed)."""
    cache = TwoTierCacheMiddleware()
    assert cache.TIER2_HIGH_SIM_THRESHOLD == 0.90


def test_cache_skips_l2_for_long_content():
    """Requests with >32K chars must skip L2 semantic search."""
    cache = TwoTierCacheMiddleware()
    long_content = "x" * 33_000
    msgs = [{"role": "user", "content": long_content}]
    result = cache.lookup("gpt-4o", msgs)
    assert isinstance(result, CacheMiss)
    assert "too long" in result.reason


def test_cache_skips_l2_for_many_messages():
    """Requests with >10 messages must skip L2."""
    cache = TwoTierCacheMiddleware()
    msgs = [{"role": "user", "content": f"msg {i}"} for i in range(15)]
    result = cache.lookup("gpt-4o", msgs)
    assert isinstance(result, CacheMiss)
    assert "too many" in result.reason
