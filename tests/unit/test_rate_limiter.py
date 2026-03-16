"""
Tests for TokenBucketRateLimiter and CostAlertManager.
"""

import pytest
import time
from agentfuse.core.rate_limiter import TokenBucketRateLimiter, RateLimitExceeded
from agentfuse.core.cost_alert import CostAlertManager, CostAlert
from agentfuse.core.budget import BudgetEngine


# --- Rate Limiter Tests ---

def test_rate_limiter_allows_within_burst():
    """Calls within burst limit must succeed immediately."""
    limiter = TokenBucketRateLimiter(rate=10, burst=5)
    for _ in range(5):
        assert limiter.acquire("tenant_1", block=False)


def test_rate_limiter_rejects_over_burst():
    """Calls exceeding burst must raise RateLimitExceeded."""
    limiter = TokenBucketRateLimiter(rate=10, burst=3)
    limiter.acquire("tenant_1", block=False)
    limiter.acquire("tenant_1", block=False)
    limiter.acquire("tenant_1", block=False)
    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.acquire("tenant_1", block=False)
    assert exc_info.value.tenant_id == "tenant_1"
    assert exc_info.value.retry_after > 0


def test_rate_limiter_refills_over_time():
    """Tokens must refill over time."""
    limiter = TokenBucketRateLimiter(rate=100, burst=2)
    limiter.acquire("tenant_1", block=False)
    limiter.acquire("tenant_1", block=False)
    time.sleep(0.05)  # 100 tokens/sec * 0.05s = 5 tokens refilled
    assert limiter.acquire("tenant_1", block=False)


def test_rate_limiter_per_tenant_isolation():
    """Different tenants must have independent buckets."""
    limiter = TokenBucketRateLimiter(rate=10, burst=2)
    limiter.acquire("tenant_a", block=False)
    limiter.acquire("tenant_a", block=False)
    # tenant_a exhausted, but tenant_b should be fine
    assert limiter.acquire("tenant_b", block=False)


def test_rate_limiter_get_remaining():
    """get_remaining must return current token count."""
    limiter = TokenBucketRateLimiter(rate=10, burst=5)
    assert limiter.get_remaining("tenant_1") == 5.0
    limiter.acquire("tenant_1", block=False)
    remaining = limiter.get_remaining("tenant_1")
    assert 3.9 < remaining < 4.1  # approximately 4


def test_rate_limiter_reset():
    """reset must clear tenant state."""
    limiter = TokenBucketRateLimiter(rate=10, burst=2)
    limiter.acquire("tenant_1", block=False)
    limiter.acquire("tenant_1", block=False)
    limiter.reset("tenant_1")
    assert limiter.acquire("tenant_1", block=False)


# --- Cost Alert Tests ---

def test_cost_alert_fires_at_threshold():
    """Alert must fire when threshold is crossed."""
    alerts = []
    manager = CostAlertManager(
        thresholds=[0.50, 0.75],
        callback=lambda a: alerts.append(a),
    )
    engine = BudgetEngine("alert_test", 10.0, "gpt-4o")
    engine.record_cost(5.5)  # 55% — crosses 50%

    result = manager.check(engine)
    assert result is not None
    assert result.threshold_pct == 0.50
    assert len(alerts) == 1


def test_cost_alert_fires_only_once():
    """Same threshold must not fire twice for same run."""
    alerts = []
    manager = CostAlertManager(
        thresholds=[0.50],
        callback=lambda a: alerts.append(a),
    )
    engine = BudgetEngine("once_test", 10.0, "gpt-4o")
    engine.record_cost(6.0)

    manager.check(engine)
    manager.check(engine)  # second check at same level
    assert len(alerts) == 1  # only fired once


def test_cost_alert_fires_multiple_thresholds():
    """Multiple thresholds crossed at once must fire the lowest unfired."""
    alerts = []
    manager = CostAlertManager(
        thresholds=[0.25, 0.50, 0.75],
        callback=lambda a: alerts.append(a),
    )
    engine = BudgetEngine("multi_test", 10.0, "gpt-4o")
    engine.record_cost(8.0)  # 80% — crosses 25%, 50%, 75%

    manager.check(engine)
    manager.check(engine)
    manager.check(engine)
    assert len(alerts) == 3


def test_cost_alert_json_serializable():
    """CostAlert must serialize to valid JSON."""
    import json
    alert = CostAlert(
        alert_type="threshold",
        tenant_id="org_123",
        run_id="run_456",
        threshold_pct=0.75,
        current_pct=0.80,
        spent_usd=8.0,
        budget_usd=10.0,
        model="gpt-4o",
    )
    data = json.loads(alert.to_json())
    assert data["tenant_id"] == "org_123"
    assert data["threshold_pct"] == 0.75


def test_cost_alert_callback_failure_doesnt_crash():
    """Failed callback must not crash the alert system."""
    def bad_callback(alert):
        raise RuntimeError("webhook down")

    manager = CostAlertManager(
        thresholds=[0.50],
        callback=bad_callback,
    )
    engine = BudgetEngine("safe_test", 10.0, "gpt-4o")
    engine.record_cost(6.0)

    # Must not raise
    result = manager.check(engine)
    assert result is not None


def test_cost_alert_reset():
    """Reset must allow alerts to fire again."""
    alerts = []
    manager = CostAlertManager(
        thresholds=[0.50],
        callback=lambda a: alerts.append(a),
    )
    engine = BudgetEngine("reset_test", 10.0, "gpt-4o")
    engine.record_cost(6.0)

    manager.check(engine)
    assert len(alerts) == 1

    manager.reset("reset_test")
    manager.check(engine)
    assert len(alerts) == 2
