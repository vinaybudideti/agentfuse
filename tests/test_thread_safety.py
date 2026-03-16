"""Thread safety tests — verify no crashes or data corruption under concurrency."""

import threading
import pytest
from agentfuse.core.budget import BudgetEngine, BudgetExhaustedGracefully
from agentfuse.storage.memory import InMemoryStore


def test_budget_engine_concurrent_record_cost():
    """10 threads recording cost simultaneously must not lose updates."""
    engine = BudgetEngine("run_concurrent", budget_usd=100.0, model="gpt-4o")
    errors = []

    def record_many():
        try:
            for _ in range(100):
                engine.record_cost(0.01)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=record_many) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threads raised errors: {errors}"
    # 10 threads * 100 iterations * 0.01 = 10.0
    assert abs(engine.spent - 10.0) < 0.001, f"Expected 10.0, got {engine.spent}"


def test_budget_engine_concurrent_check_and_act():
    """Concurrent check_and_act must not crash or corrupt state."""
    engine = BudgetEngine("run_check", budget_usd=10.0, model="gpt-4o")
    errors = []

    def check_many():
        try:
            for _ in range(50):
                messages = [{"role": "user", "content": "hello"}]
                engine.check_and_act(0.01, messages)
        except BudgetExhaustedGracefully:
            pass  # Expected when budget runs out
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=check_many) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threads raised unexpected errors: {errors}"
    # State must be a valid BudgetState
    from agentfuse.core.budget import BudgetState
    assert engine.state in list(BudgetState)


def test_memory_store_concurrent_access():
    """10 threads reading and writing to InMemoryStore must not crash."""
    store = InMemoryStore()
    errors = []

    def write_many(thread_id):
        try:
            for i in range(100):
                store.set(f"run_{thread_id}", f"key_{i}", f"value_{i}")
        except Exception as e:
            errors.append(e)

    def read_many(thread_id):
        try:
            for i in range(100):
                store.get(f"run_{thread_id}", f"key_{i}")
                store.get_all(f"run_{thread_id}")
                store.list_runs()
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(5):
        threads.append(threading.Thread(target=write_many, args=(i,)))
        threads.append(threading.Thread(target=read_many, args=(i,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threads raised errors: {errors}"
    # Verify data integrity
    for i in range(5):
        data = store.get_all(f"run_{i}")
        assert len(data) == 100, f"Expected 100 keys for run_{i}, got {len(data)}"
