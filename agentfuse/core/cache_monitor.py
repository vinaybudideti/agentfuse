"""
CacheMonitor — tracks cache hit/miss rates and alerts on degradation.

SAFE-CACHE defense: monitors cache miss rate to detect:
1. Embedding drift (model update broke cache matches)
2. Cache poisoning attacks (adversarial prompts causing misses)
3. Workload shift (users asking different types of questions)

Usage:
    monitor = CacheMonitor(alert_threshold=0.5)
    monitor.record_hit()
    monitor.record_miss()

    if monitor.is_degraded():
        print(f"Cache degraded! Hit rate: {monitor.hit_rate:.1%}")
"""

import logging
import threading
import time
from typing import Optional, Callable
from collections import deque

logger = logging.getLogger(__name__)


class CacheMonitor:
    """
    Monitors cache hit/miss rates with sliding window and alerting.
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_threshold: float = 0.5,
        alert_callback: Optional[Callable] = None,
    ):
        self._window: deque[bool] = deque(maxlen=window_size)
        self._total_hits = 0
        self._total_misses = 0
        self._alert_threshold = alert_threshold
        self._alert_callback = alert_callback
        self._alert_fired = False
        self._lock = threading.Lock()

    def record_hit(self):
        """Record a cache hit."""
        with self._lock:
            self._window.append(True)
            self._total_hits += 1
            self._alert_fired = False  # reset alert on hit

    def record_miss(self, reason: str = ""):
        """Record a cache miss."""
        with self._lock:
            self._window.append(False)
            self._total_misses += 1
            self._check_alert(reason)

    @property
    def hit_rate(self) -> float:
        """Current hit rate in the sliding window."""
        with self._lock:
            if not self._window:
                return 0.0
            return sum(self._window) / len(self._window)

    @property
    def overall_hit_rate(self) -> float:
        """Overall hit rate since creation."""
        total = self._total_hits + self._total_misses
        return self._total_hits / total if total > 0 else 0.0

    def is_degraded(self) -> bool:
        """Check if cache performance has degraded below threshold."""
        return len(self._window) >= 10 and self.hit_rate < self._alert_threshold

    def _check_alert(self, reason: str):
        """Fire alert if hit rate drops below threshold."""
        if self._alert_fired or len(self._window) < 10:
            return
        if self.hit_rate < self._alert_threshold:
            self._alert_fired = True
            msg = (f"Cache degraded: hit rate {self.hit_rate:.1%} "
                   f"below threshold {self._alert_threshold:.1%}")
            if reason:
                msg += f" (last miss: {reason})"
            logger.warning(msg)
            if self._alert_callback:
                try:
                    self._alert_callback(self.hit_rate, reason)
                except Exception:
                    pass

    def get_stats(self) -> dict:
        """Get cache monitoring statistics."""
        with self._lock:
            return {
                "window_hit_rate": round(self.hit_rate, 4),
                "overall_hit_rate": round(self.overall_hit_rate, 4),
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "total_lookups": self._total_hits + self._total_misses,
                "window_size": len(self._window),
                "is_degraded": self.is_degraded(),
            }

    def reset(self):
        """Reset all monitoring state."""
        with self._lock:
            self._window.clear()
            self._total_hits = 0
            self._total_misses = 0
            self._alert_fired = False
