"""
InMemoryBudgetStore — thread-safe budget store with TTL and LRU eviction.
Also provides AsyncInMemoryBudgetStore for async codepaths.

The original InMemoryStore (key-value run state) is preserved for backward compat.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from cachetools import TTLCache


# --- Budget Store ---

@dataclass
class BudgetEntry:
    run_id: str
    initial_budget: float
    remaining: float
    created_at: float = field(default_factory=time.monotonic)


class InMemoryBudgetStore:
    """Thread-safe budget store using TTLCache for automatic expiry."""

    def __init__(self, max_entries: int = 10_000, ttl_seconds: float = 3600.0):
        self._cache = TTLCache(maxsize=max_entries, ttl=ttl_seconds)
        self._lock = threading.RLock()

    def create_run(self, run_id: str, budget_usd: float) -> bool:
        with self._lock:
            if run_id in self._cache:
                return False
            self._cache[run_id] = BudgetEntry(
                run_id=run_id,
                initial_budget=budget_usd,
                remaining=budget_usd,
            )
            return True

    def check_and_deduct(self, run_id: str, estimated_cost: float) -> bool:
        with self._lock:
            entry = self._cache.get(run_id)
            if entry is None:
                return False
            if entry.remaining < estimated_cost:
                return False
            entry.remaining -= estimated_cost
            return True

    def reconcile(self, run_id: str, estimated: float, actual: float) -> float:
        with self._lock:
            entry = self._cache.get(run_id)
            if entry is None:
                return 0.0
            entry.remaining += (estimated - actual)
            return entry.remaining

    def get_remaining(self, run_id: str) -> Optional[float]:
        with self._lock:
            entry = self._cache.get(run_id)
            return entry.remaining if entry else None


class AsyncInMemoryBudgetStore:
    """Async-safe budget store using asyncio.Lock."""

    def __init__(self, max_entries: int = 10_000, ttl_seconds: float = 3600.0):
        self._cache = TTLCache(maxsize=max_entries, ttl=ttl_seconds)
        self._lock = asyncio.Lock()

    async def create_run(self, run_id: str, budget_usd: float) -> bool:
        async with self._lock:
            if run_id in self._cache:
                return False
            self._cache[run_id] = BudgetEntry(
                run_id=run_id,
                initial_budget=budget_usd,
                remaining=budget_usd,
            )
            return True

    async def check_and_deduct(self, run_id: str, estimated_cost: float) -> bool:
        async with self._lock:
            entry = self._cache.get(run_id)
            if entry is None:
                return False
            if entry.remaining < estimated_cost:
                return False
            entry.remaining -= estimated_cost
            return True

    async def reconcile(self, run_id: str, estimated: float, actual: float) -> float:
        async with self._lock:
            entry = self._cache.get(run_id)
            if entry is None:
                return 0.0
            entry.remaining += (estimated - actual)
            return entry.remaining

    async def get_remaining(self, run_id: str) -> Optional[float]:
        async with self._lock:
            entry = self._cache.get(run_id)
            return entry.remaining if entry else None


# --- Original InMemoryStore (backward compat) ---

class InMemoryStore:
    """Original key-value run state storage (zero-dependency)."""

    def __init__(self):
        self._store = {}
        self._lock = threading.RLock()

    def set(self, run_id, key, value):
        with self._lock:
            if run_id not in self._store:
                self._store[run_id] = {}
            self._store[run_id][key] = value

    def get(self, run_id, key, default=None):
        with self._lock:
            return self._store.get(run_id, {}).get(key, default)

    def delete(self, run_id):
        with self._lock:
            self._store.pop(run_id, None)

    def get_all(self, run_id):
        with self._lock:
            return dict(self._store.get(run_id, {}))

    def list_runs(self):
        with self._lock:
            return list(self._store.keys())
