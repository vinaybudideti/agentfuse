"""
Tests for CostTracker — unified cost aggregation.
"""

import threading
from agentfuse.core.cost_tracker import CostTracker


def test_record_call():
    tracker = CostTracker()
    tracker.record_call("gpt-4o", "openai", "run_1", input_tokens=100, output_tokens=50, cost_usd=0.0075)
    snap = tracker.get_snapshot()
    assert snap.total_usd == 0.0075
    assert snap.total_calls == 1
    assert snap.total_input_tokens == 100


def test_cache_hit_rate():
    tracker = CostTracker()
    tracker.record_call("gpt-4o", "openai", "run_1", cost_usd=0.01)  # miss
    tracker.record_cache_hit("gpt-4o", "run_1", tokens_saved=100)
    tracker.record_cache_hit("gpt-4o", "run_1", tokens_saved=100)
    # 2 hits, 1 miss = 66.7% hit rate
    assert abs(tracker.cache_hit_rate() - 0.667) < 0.01


def test_by_model_aggregation():
    tracker = CostTracker()
    tracker.record_call("gpt-4o", "openai", "run_1", cost_usd=0.01)
    tracker.record_call("gpt-4o", "openai", "run_2", cost_usd=0.02)
    tracker.record_call("claude-sonnet-4-6", "anthropic", "run_3", cost_usd=0.05)

    snap = tracker.get_snapshot()
    assert snap.by_model["gpt-4o"]["cost"] == 0.03
    assert snap.by_model["claude-sonnet-4-6"]["cost"] == 0.05


def test_top_models():
    tracker = CostTracker()
    tracker.record_call("gpt-4o", "openai", "run_1", cost_usd=0.10)
    tracker.record_call("claude-sonnet-4-6", "anthropic", "run_2", cost_usd=0.50)
    tracker.record_call("gpt-4o-mini", "openai", "run_3", cost_usd=0.01)

    top = tracker.get_top_models(2)
    assert top[0][0] == "claude-sonnet-4-6"
    assert top[0][1] == 0.50


def test_concurrent_tracking():
    tracker = CostTracker()
    errors = []

    def stress(thread_id):
        try:
            for i in range(100):
                tracker.record_call(f"model_{thread_id}", "openai", f"run_{thread_id}",
                                    input_tokens=10, output_tokens=5, cost_usd=0.001)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=stress, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    snap = tracker.get_snapshot()
    assert snap.total_calls == 1000  # 10 threads * 100 calls
    assert abs(snap.total_usd - 1.0) < 0.001  # 1000 * 0.001


def test_cost_per_call():
    tracker = CostTracker()
    tracker.record_call("gpt-4o", "openai", "run_1", cost_usd=0.10)
    tracker.record_call("gpt-4o", "openai", "run_2", cost_usd=0.20)
    assert abs(tracker.cost_per_call() - 0.15) < 0.001


def test_reset():
    tracker = CostTracker()
    tracker.record_call("gpt-4o", "openai", "run_1", cost_usd=0.10)
    tracker.reset()
    snap = tracker.get_snapshot()
    assert snap.total_usd == 0.0
    assert snap.total_calls == 0
