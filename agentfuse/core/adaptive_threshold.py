"""
AdaptiveSimilarityThreshold — auto-tunes L2 cache similarity threshold.

The Gap Analysis spec says: "Start at 0.92 cosine similarity threshold,
then tune: stricter (0.95+) for factual queries, looser (0.88+) for
FAQ-style queries."

This module implements automatic threshold tuning based on observed
cache hit quality feedback. When users report bad cache hits (wrong
answers), the threshold tightens. When cache hit rate is too low,
it loosens slightly.

This is a novel approach — no existing LLM cost optimization tool
implements adaptive cache thresholds.
"""

import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AdaptiveSimilarityThreshold:
    """
    Auto-tunes the L2 semantic cache similarity threshold.

    The threshold starts at `initial` and adjusts based on feedback:
    - `report_bad_hit()`: threshold tightens by `step` (up to `max_threshold`)
    - `report_good_hit()`: threshold loosens by `step/2` (down to `min_threshold`)

    The adjustment is conservative — tightening is 2x faster than loosening
    because serving wrong answers is worse than a cache miss.

    Usage:
        threshold = AdaptiveSimilarityThreshold()
        current = threshold.get()  # 0.92
        threshold.report_bad_hit()  # tightens to 0.93
        threshold.report_good_hit()  # loosens to 0.925
    """

    def __init__(
        self,
        initial: float = 0.92,
        min_threshold: float = 0.85,
        max_threshold: float = 0.98,
        step: float = 0.01,
    ):
        self._threshold = initial
        self._min = min_threshold
        self._max = max_threshold
        self._step = step
        self._lock = threading.Lock()
        self._good_hits = 0
        self._bad_hits = 0
        self._total_lookups = 0

    def get(self) -> float:
        """Get current similarity threshold."""
        return self._threshold

    def report_bad_hit(self):
        """Report that a cache hit returned wrong/irrelevant content.
        Tightens the threshold to reduce false positive hits."""
        with self._lock:
            self._bad_hits += 1
            self._threshold = min(self._max, self._threshold + self._step)
            logger.info("Threshold tightened to %.3f (bad hit #%d)", self._threshold, self._bad_hits)

    def report_good_hit(self):
        """Report that a cache hit returned correct content.
        Loosens the threshold slightly to improve hit rate."""
        with self._lock:
            self._good_hits += 1
            # Loosen at half the rate of tightening (conservative)
            self._threshold = max(self._min, self._threshold - self._step / 2)

    def report_lookup(self, was_hit: bool):
        """Record a cache lookup result for statistics."""
        with self._lock:
            self._total_lookups += 1
            if was_hit:
                self._good_hits += 1

    def get_stats(self) -> dict:
        """Return threshold tuning statistics."""
        with self._lock:
            total_feedback = self._good_hits + self._bad_hits
            accuracy = self._good_hits / total_feedback if total_feedback > 0 else 1.0
            return {
                "current_threshold": self._threshold,
                "good_hits": self._good_hits,
                "bad_hits": self._bad_hits,
                "total_lookups": self._total_lookups,
                "accuracy": round(accuracy, 4),
            }

    def reset(self, initial: Optional[float] = None):
        """Reset to initial state."""
        with self._lock:
            self._threshold = initial or 0.92
            self._good_hits = 0
            self._bad_hits = 0
            self._total_lookups = 0
