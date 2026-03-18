"""Tests for UsageLimits."""

import pytest
from agentfuse.core.usage_limits import UsageLimits, UsageLimitExceeded


def test_no_limits_passes():
    limits = UsageLimits()
    limits.check("user:alice")  # no limits set, should pass


def test_cost_limit_enforcement():
    limits = UsageLimits()
    limits.set_limit("user:bob", max_cost_per_day=1.0)
    limits.record("user:bob", cost=0.90)
    limits.check("user:bob")  # still under
    limits.record("user:bob", cost=0.20)
    with pytest.raises(UsageLimitExceeded, match="cost_per_day"):
        limits.check("user:bob")


def test_request_limit_enforcement():
    limits = UsageLimits()
    limits.set_limit("user:carol", max_requests_per_hour=5)
    for _ in range(5):
        limits.record("user:carol")
    with pytest.raises(UsageLimitExceeded, match="requests_per_hour"):
        limits.check("user:carol")


def test_token_limit_enforcement():
    limits = UsageLimits()
    limits.set_limit("team:eng", max_tokens_per_day=1000)
    limits.record("team:eng", tokens=900)
    limits.check("team:eng")
    limits.record("team:eng", tokens=200)
    with pytest.raises(UsageLimitExceeded, match="tokens_per_day"):
        limits.check("team:eng")


def test_get_usage():
    limits = UsageLimits()
    limits.record("user:dave", cost=0.50, tokens=100)
    usage = limits.get_usage("user:dave")
    assert usage["cost_today"] == 0.50
    assert usage["tokens_today"] == 100
    assert usage["requests_this_hour"] == 1


def test_multiple_entities_independent():
    limits = UsageLimits()
    limits.set_limit("user:a", max_cost_per_day=1.0)
    limits.set_limit("user:b", max_cost_per_day=1.0)
    limits.record("user:a", cost=0.90)
    limits.check("user:b")  # b is independent, should pass


def test_exception_fields():
    limits = UsageLimits()
    limits.set_limit("user:x", max_cost_per_day=0.50)
    limits.record("user:x", cost=0.60)
    try:
        limits.check("user:x")
    except UsageLimitExceeded as e:
        assert e.entity_id == "user:x"
        assert e.limit_type == "cost_per_day"
        assert e.current >= 0.60
        assert e.limit == 0.50
