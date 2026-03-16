"""
Phase 3 — Redis budget store integration tests.

Requires Redis on localhost:6379.
"""

import threading
import time
import pytest

try:
    import redis
    _r = redis.Redis(host="localhost", port=6379)
    _r.ping()
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available on localhost:6379")


def _make_store():
    from agentfuse.storage.redis_store import RedisBudgetStore
    return RedisBudgetStore("redis://localhost:6379")


def _unique_run_id():
    import uuid
    return f"test_{uuid.uuid4().hex[:8]}"


def test_atomic_check_deduct_correct():
    """Single run, deduct 3 times, correct remainder."""
    store = _make_store()
    run_id = _unique_run_id()
    store.create_run(run_id, 10.0, ttl=60)

    assert store.check_and_deduct(run_id, 3.0) == 1
    assert store.check_and_deduct(run_id, 3.0) == 1
    assert store.check_and_deduct(run_id, 3.0) == 1

    remaining = store.get_remaining(run_id)
    assert abs(remaining - 1.0) < 0.001


def test_concurrent_runs_no_bleed():
    """10 runs simultaneously, each with own budget, no crossover."""
    store = _make_store()
    run_ids = [_unique_run_id() for _ in range(10)]
    for rid in run_ids:
        store.create_run(rid, 1.0, ttl=60)

    errors = []

    def deduct_run(run_id):
        try:
            for _ in range(10):
                store.check_and_deduct(run_id, 0.05)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=deduct_run, args=(rid,)) for rid in run_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for rid in run_ids:
        remaining = store.get_remaining(rid)
        assert abs(remaining - 0.5) < 0.01


def test_reconcile_adds_back_overestimate():
    """Reconcile adjusts budget correctly."""
    store = _make_store()
    run_id = _unique_run_id()
    store.create_run(run_id, 10.0, ttl=60)
    store.check_and_deduct(run_id, 5.0)
    new_remaining = store.reconcile(run_id, 5.0, 3.0)
    assert abs(new_remaining - 7.0) < 0.001


def test_expired_run_returns_not_found():
    """Expired run must return None."""
    store = _make_store()
    run_id = _unique_run_id()
    store.create_run(run_id, 10.0, ttl=1)
    time.sleep(2)
    assert store.get_remaining(run_id) is None


def test_insufficient_budget_returns_zero():
    """Deducting more than budget must return 0 (insufficient)."""
    store = _make_store()
    run_id = _unique_run_id()
    store.create_run(run_id, 1.0, ttl=60)
    result = store.check_and_deduct(run_id, 2.0)
    assert result == 0
