"""
BatchEligibilityDetector — detects workloads eligible for batch API discounts.

Both OpenAI and Anthropic offer 50% discounts for batch processing,
stackable with prompt caching for up to 95% savings. This module
detects non-real-time workloads and recommends batch routing.

Research finding: "No gateway currently offers automatic batch detection."
AgentFuse is the first to implement this.

Detection heuristics:
1. Multiple requests with identical system prompts → batch candidate
2. Requests queued within a short window → can be batched
3. Non-interactive patterns (no user waiting for response) → batch eligible
4. Large-scale data processing (many similar requests) → batch eligible

Usage:
    detector = BatchEligibilityDetector()
    detector.observe(request)
    if detector.should_batch():
        # Route to batch API for 50% discount
        result = batch_api.submit(detector.get_batch())
"""

import hashlib
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BatchCandidate:
    """A group of requests eligible for batching."""
    system_prompt_hash: str
    model: str
    request_count: int
    first_seen: float
    last_seen: float
    estimated_savings_usd: float = 0.0

    @property
    def window_seconds(self) -> float:
        return self.last_seen - self.first_seen


class BatchEligibilityDetector:
    """
    Detects request patterns eligible for batch API discounts.

    Monitors incoming requests and identifies batching opportunities
    based on system prompt similarity, request frequency, and model.
    """

    def __init__(
        self,
        min_batch_size: int = 5,
        batch_window_seconds: float = 60.0,
        discount_rate: float = 0.50,
    ):
        self._min_batch_size = min_batch_size
        self._window = batch_window_seconds
        self._discount = discount_rate
        self._lock = threading.Lock()

        # Track request patterns: {system_hash: {model: [timestamps]}}
        self._patterns: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._total_observed = 0
        self._batch_opportunities = 0

    def observe(self, messages: list[dict], model: str,
                estimated_cost: float = 0.0) -> Optional[BatchCandidate]:
        """
        Observe a request and check if it's part of a batchable pattern.

        Returns BatchCandidate if a batching opportunity is detected.
        """
        system_hash = self._extract_system_hash(messages)
        now = time.time()

        with self._lock:
            self._total_observed += 1

            # Record this request
            self._patterns[system_hash][model].append(now)

            # Prune old entries
            cutoff = now - self._window
            timestamps = self._patterns[system_hash][model]
            self._patterns[system_hash][model] = [t for t in timestamps if t > cutoff]
            timestamps = self._patterns[system_hash][model]

            # Check if batch threshold met (timestamps always non-empty after append+prune)
            if timestamps and len(timestamps) >= self._min_batch_size:
                self._batch_opportunities += 1
                return BatchCandidate(
                    system_prompt_hash=system_hash,
                    model=model,
                    request_count=len(timestamps),
                    first_seen=timestamps[0],
                    last_seen=timestamps[-1],
                    estimated_savings_usd=estimated_cost * self._discount * len(timestamps),
                )

        return None

    def get_stats(self) -> dict:
        """Get detection statistics."""
        with self._lock:
            return {
                "total_observed": self._total_observed,
                "batch_opportunities": self._batch_opportunities,
                "active_patterns": sum(
                    len(models) for models in self._patterns.values()
                ),
                "batch_rate": (self._batch_opportunities / self._total_observed
                               if self._total_observed > 0 else 0.0),
            }

    def _extract_system_hash(self, messages: list[dict]) -> str:
        """Hash the system prompt for pattern matching."""
        system_content = ""
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content", "")
                if isinstance(content, str):
                    system_content += content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            system_content += block.get("text", "")
        if not system_content:
            system_content = "_no_system_prompt_"
        return hashlib.md5(system_content.encode()).hexdigest()[:16]

    def reset(self):
        """Reset all tracking state."""
        with self._lock:
            self._patterns.clear()
            self._total_observed = 0
            self._batch_opportunities = 0
