"""
Tests for ModelLoadBalancer.
"""

import time
from agentfuse.core.load_balancer import ModelLoadBalancer


def test_round_robin():
    lb = ModelLoadBalancer(strategy="round_robin")
    lb.add_endpoint("gpt-4o", api_key="key1")
    lb.add_endpoint("gpt-4o", api_key="key2")
    lb.add_endpoint("gpt-4o", api_key="key3")

    keys = [lb.get_endpoint("gpt-4o").api_key for _ in range(6)]
    # Should rotate: key1, key2, key3, key1, key2, key3
    assert keys == ["key1", "key2", "key3", "key1", "key2", "key3"]


def test_random_strategy():
    lb = ModelLoadBalancer(strategy="random")
    lb.add_endpoint("gpt-4o", api_key="key1")
    lb.add_endpoint("gpt-4o", api_key="key2")

    endpoints = [lb.get_endpoint("gpt-4o") for _ in range(20)]
    keys = {e.api_key for e in endpoints}
    assert len(keys) > 1  # should use both keys


def test_unhealthy_endpoint_skipped():
    lb = ModelLoadBalancer(strategy="round_robin")
    lb.add_endpoint("gpt-4o", api_key="key1")
    lb.add_endpoint("gpt-4o", api_key="key2")

    # Mark key1 as unhealthy
    ep1 = lb.get_endpoint("gpt-4o")
    lb.report_failure(ep1)

    # Next call should skip key1
    ep2 = lb.get_endpoint("gpt-4o")
    assert ep2.api_key == "key2"


def test_unhealthy_endpoint_recovers():
    lb = ModelLoadBalancer(strategy="round_robin")
    lb.HEALTH_COOLDOWN = 0.1  # fast recovery for test
    lb.add_endpoint("gpt-4o", api_key="key1")
    lb.add_endpoint("gpt-4o", api_key="key2")

    ep1 = lb.get_endpoint("gpt-4o")
    lb.report_failure(ep1)

    time.sleep(0.15)  # wait for cooldown

    # key1 should be available again
    endpoints = {lb.get_endpoint("gpt-4o").api_key for _ in range(10)}
    assert "key1" in endpoints


def test_success_updates_latency():
    lb = ModelLoadBalancer(strategy="least_latency")
    lb.add_endpoint("gpt-4o", api_key="fast_key")
    lb.add_endpoint("gpt-4o", api_key="slow_key")

    fast = lb.get_endpoint("gpt-4o")
    lb.report_success(fast, latency_ms=50)

    slow = lb.get_endpoint("gpt-4o")
    lb.report_success(slow, latency_ms=500)

    # least_latency should prefer fast_key
    chosen = lb.get_endpoint("gpt-4o")
    assert chosen.api_key == "fast_key"


def test_no_endpoints_returns_none():
    lb = ModelLoadBalancer()
    assert lb.get_endpoint("nonexistent") is None


def test_per_model_isolation():
    lb = ModelLoadBalancer()
    lb.add_endpoint("gpt-4o", api_key="openai_key")
    lb.add_endpoint("claude-sonnet-4-6", api_key="anthropic_key")

    gpt = lb.get_endpoint("gpt-4o")
    claude = lb.get_endpoint("claude-sonnet-4-6")
    assert gpt.api_key == "openai_key"
    assert claude.api_key == "anthropic_key"


def test_get_stats():
    lb = ModelLoadBalancer()
    lb.add_endpoint("gpt-4o", api_key="key1")
    lb.add_endpoint("gpt-4o", api_key="key2")
    lb.add_endpoint("claude-sonnet-4-6", api_key="key3")

    stats = lb.get_stats()
    assert stats["gpt-4o"]["total_endpoints"] == 2
    assert stats["claude-sonnet-4-6"]["total_endpoints"] == 1


def test_all_unhealthy_falls_back():
    """When all endpoints are unhealthy, least-recently-failed is tried."""
    lb = ModelLoadBalancer()
    lb.add_endpoint("gpt-4o", api_key="key1")
    lb.add_endpoint("gpt-4o", api_key="key2")

    ep1 = lb.get_endpoint("gpt-4o")
    lb.report_failure(ep1)
    ep2 = lb.get_endpoint("gpt-4o")
    lb.report_failure(ep2)

    # All unhealthy — should still return something
    ep3 = lb.get_endpoint("gpt-4o")
    assert ep3 is not None
