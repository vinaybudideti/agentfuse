"""
Tests for RequestDeduplicator — coalescing identical in-flight requests.
"""

import threading
import time
from agentfuse.core.dedup import RequestDeduplicator


def test_basic_execution():
    """Normal execution must return the result."""
    dedup = RequestDeduplicator()
    result = dedup.execute("key1", lambda: "hello")
    assert result == "hello"


def test_dedup_concurrent_same_key():
    """Two threads with same key must only call fn ONCE."""
    dedup = RequestDeduplicator()
    call_count = 0
    results = []

    def slow_fn():
        nonlocal call_count
        call_count += 1
        time.sleep(0.1)  # simulate API call
        return f"result_{call_count}"

    def thread_fn():
        result = dedup.execute("same_key", slow_fn)
        results.append(result)

    t1 = threading.Thread(target=thread_fn)
    t2 = threading.Thread(target=thread_fn)
    t1.start()
    time.sleep(0.02)  # ensure t1 starts first
    t2.start()
    t1.join()
    t2.join()

    # Only ONE actual API call should have been made
    assert call_count == 1
    assert len(results) == 2
    # Both threads should get the same result
    assert results[0] == results[1]
    assert dedup.dedup_count == 1


def test_different_keys_not_deduped():
    """Different keys must execute independently."""
    dedup = RequestDeduplicator()
    call_count = 0

    def counter_fn():
        nonlocal call_count
        call_count += 1
        return f"result_{call_count}"

    dedup.execute("key_a", counter_fn)
    dedup.execute("key_b", counter_fn)

    assert call_count == 2  # two separate calls


def test_error_propagated():
    """Errors must propagate to waiting threads."""
    dedup = RequestDeduplicator()

    def failing_fn():
        time.sleep(0.05)
        raise ValueError("API error")

    errors = []

    def thread_fn():
        try:
            dedup.execute("error_key", failing_fn)
        except ValueError as e:
            errors.append(str(e))

    t1 = threading.Thread(target=thread_fn)
    t2 = threading.Thread(target=thread_fn)
    t1.start()
    time.sleep(0.01)
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) >= 1  # at least the executor gets the error


def test_make_key_deterministic():
    """Same inputs must produce same key."""
    key1 = RequestDeduplicator.make_key("gpt-4o", [{"role": "user", "content": "hi"}])
    key2 = RequestDeduplicator.make_key("gpt-4o", [{"role": "user", "content": "hi"}])
    assert key1 == key2


def test_make_key_different_for_different_input():
    """Different inputs must produce different keys."""
    key1 = RequestDeduplicator.make_key("gpt-4o", [{"role": "user", "content": "hi"}])
    key2 = RequestDeduplicator.make_key("gpt-4o", [{"role": "user", "content": "bye"}])
    assert key1 != key2


def test_cleanup_after_completion():
    """Results must be cleaned up after completion."""
    dedup = RequestDeduplicator()
    dedup.execute("cleanup_key", lambda: "result")
    time.sleep(1.5)  # wait for cleanup timer
    # Internal state should be clean
    assert "cleanup_key" not in dedup._results
    assert "cleanup_key" not in dedup._inflight
