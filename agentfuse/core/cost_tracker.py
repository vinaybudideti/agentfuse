"""
CostTracker — unified cost aggregation across runs, models, and providers.

Aggregates cost data in real-time for dashboard display, billing reports,
and budget governance. Thread-safe for concurrent access.

This is the single source of truth for "how much has this organization
spent on LLM calls?" — answerable in O(1) at any time.
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CostSnapshot:
    """Point-in-time cost summary."""
    total_usd: float = 0.0
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    by_model: dict = field(default_factory=dict)
    by_provider: dict = field(default_factory=dict)
    by_run: dict = field(default_factory=dict)
    period_start: float = field(default_factory=time.time)


class CostTracker:
    """
    Real-time cost aggregation across all AgentFuse operations.

    Thread-safe. Designed for high-frequency updates from concurrent runs.

    Usage:
        tracker = CostTracker()
        tracker.record_call("gpt-4o", "openai", "run_123",
                           input_tokens=100, output_tokens=50, cost_usd=0.0075)
        tracker.record_cache_hit("gpt-4o", "run_123", tokens_saved=100)

        snapshot = tracker.get_snapshot()
        print(f"Total spent: ${snapshot.total_usd:.2f}")
        print(f"Cache hit rate: {tracker.cache_hit_rate():.1%}")
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._total_usd = 0.0
        self._total_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cached_tokens = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._by_model = defaultdict(lambda: {"cost": 0.0, "calls": 0})
        self._by_provider = defaultdict(lambda: {"cost": 0.0, "calls": 0})
        self._by_run = defaultdict(lambda: {"cost": 0.0, "calls": 0})
        self._start_time = time.time()

    def record_call(
        self,
        model: str,
        provider: str,
        run_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ):
        """Record an LLM API call."""
        with self._lock:
            self._total_usd += cost_usd
            self._total_calls += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._cache_misses += 1

            self._by_model[model]["cost"] += cost_usd
            self._by_model[model]["calls"] += 1
            self._by_provider[provider]["cost"] += cost_usd
            self._by_provider[provider]["calls"] += 1
            self._by_run[run_id]["cost"] += cost_usd
            self._by_run[run_id]["calls"] += 1

    def record_cache_hit(self, model: str, run_id: str, tokens_saved: int = 0):
        """Record a cache hit (no API call made)."""
        with self._lock:
            self._cache_hits += 1
            self._total_cached_tokens += tokens_saved

    def cache_hit_rate(self) -> float:
        """Current cache hit rate (0.0 to 1.0)."""
        with self._lock:
            total = self._cache_hits + self._cache_misses
            return self._cache_hits / total if total > 0 else 0.0

    def cost_per_call(self) -> float:
        """Average cost per API call."""
        with self._lock:
            return self._total_usd / self._total_calls if self._total_calls > 0 else 0.0

    def get_snapshot(self) -> CostSnapshot:
        """Get a point-in-time cost summary."""
        with self._lock:
            return CostSnapshot(
                total_usd=self._total_usd,
                total_calls=self._total_calls,
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                total_cached_tokens=self._total_cached_tokens,
                cache_hit_count=self._cache_hits,
                cache_miss_count=self._cache_misses,
                by_model=dict(self._by_model),
                by_provider=dict(self._by_provider),
                by_run=dict(self._by_run),
                period_start=self._start_time,
            )

    def get_top_models(self, n: int = 5) -> list[tuple[str, float]]:
        """Get top N most expensive models."""
        with self._lock:
            sorted_models = sorted(
                self._by_model.items(),
                key=lambda x: x[1]["cost"],
                reverse=True,
            )
            return [(m, d["cost"]) for m, d in sorted_models[:n]]

    def get_top_runs(self, n: int = 5) -> list[tuple[str, float]]:
        """Get top N most expensive runs."""
        with self._lock:
            sorted_runs = sorted(
                self._by_run.items(),
                key=lambda x: x[1]["cost"],
                reverse=True,
            )
            return [(r, d["cost"]) for r, d in sorted_runs[:n]]

    def reset(self):
        """Reset all tracking state."""
        with self._lock:
            self._total_usd = 0.0
            self._total_calls = 0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_cached_tokens = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._by_model.clear()
            self._by_provider.clear()
            self._by_run.clear()
            self._start_time = time.time()
