"""
Phase 3 — InMemoryBudgetStore tests.
"""

import asyncio
import threading

from agentfuse.storage.memory import InMemoryBudgetStore, AsyncInMemoryBudgetStore


def test_create_and_deduct():
    """Basic create → deduct → remaining flow."""
    store = InMemoryBudgetStore()
    assert store.create_run("run1", 10.0)
    assert store.check_and_deduct("run1", 3.0)
    assert store.get_remaining("run1") == 7.0


def test_insufficient_budget_returns_false():
    """Deducting more than remaining must return False."""
    store = InMemoryBudgetStore()
    store.create_run("run1", 1.0)
    assert not store.check_and_deduct("run1", 2.0)
    assert store.get_remaining("run1") == 1.0  # unchanged


def test_reconcile_adds_back_overestimate():
    """Reconcile: estimated=1.0, actual=0.6 → 0.4 added back."""
    store = InMemoryBudgetStore()
    store.create_run("run1", 10.0)
    store.check_and_deduct("run1", 1.0)
    remaining = store.reconcile("run1", 1.0, 0.6)
    assert abs(remaining - 9.4) < 0.001


def test_duplicate_create_returns_false():
    """Creating a run that already exists must return False."""
    store = InMemoryBudgetStore()
    assert store.create_run("run1", 10.0)
    assert not store.create_run("run1", 20.0)


def test_missing_run_returns_none():
    """Getting remaining for nonexistent run must return None."""
    store = InMemoryBudgetStore()
    assert store.get_remaining("nonexistent") is None


def test_concurrent_deductions_no_bleed():
    """10 threads deducting must not corrupt budget."""
    store = InMemoryBudgetStore()
    store.create_run("run1", 100.0)
    errors = []

    def deduct_many():
        try:
            for _ in range(100):
                store.check_and_deduct("run1", 0.01)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=deduct_many) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    remaining = store.get_remaining("run1")
    # 10 * 100 * 0.01 = 10.0 deducted from 100.0
    assert abs(remaining - 90.0) < 0.01


def test_async_store_basic():
    """Async budget store basic flow."""
    async def run():
        store = AsyncInMemoryBudgetStore()
        assert await store.create_run("run1", 10.0)
        assert await store.check_and_deduct("run1", 3.0)
        remaining = await store.get_remaining("run1")
        assert remaining == 7.0

    asyncio.run(run())


def test_async_store_reconcile():
    """Async reconcile must adjust budget correctly."""
    async def run():
        store = AsyncInMemoryBudgetStore()
        await store.create_run("run1", 10.0)
        await store.check_and_deduct("run1", 5.0)
        remaining = await store.reconcile("run1", 5.0, 3.0)
        assert abs(remaining - 7.0) < 0.001

    asyncio.run(run())


def test_budget_summary_returns_none_for_missing():
    """get_budget_summary for missing run must return None."""
    store = InMemoryBudgetStore()
    assert store.get_budget_summary("nonexistent") is None


def test_original_inmemorystore_backward_compat():
    """Original InMemoryStore must still work."""
    from agentfuse.storage.memory import InMemoryStore
    store = InMemoryStore()
    store.set("run1", "key1", "value1")
    assert store.get("run1", "key1") == "value1"
    assert store.get("run1", "key2", "default") == "default"
    assert "run1" in store.list_runs()
    store.delete("run1")
    assert "run1" not in store.list_runs()


def test_inmemorystore_get_all():
    """get_all must return all key-value pairs for a run."""
    from agentfuse.storage.memory import InMemoryStore
    store = InMemoryStore()
    store.set("run1", "k1", "v1")
    store.set("run1", "k2", "v2")
    all_data = store.get_all("run1")
    assert all_data == {"k1": "v1", "k2": "v2"}


def test_budget_store_multiple_runs():
    """Budget store must isolate multiple concurrent runs."""
    store = InMemoryBudgetStore()
    store.create_run("run1", 10.0)
    store.create_run("run2", 20.0)

    store.check_and_deduct("run1", 3.0)
    store.check_and_deduct("run2", 5.0)

    assert store.get_remaining("run1") == 7.0
    assert store.get_remaining("run2") == 15.0
