"""
GCRA Rate Limiter — Generic Cell Rate Algorithm.

Based on Helicone's architecture: GCRA provides smoother traffic shaping
than token-bucket approaches. Instead of allowing bursts up to bucket
capacity, GCRA enforces a steady rate with configurable tolerance.

GCRA is used by Helicone at global, per-router, and per-API-key levels.
It's the standard algorithm used by telecom networks and CDNs.

How it works:
- Each request has a "theoretical arrival time" (TAT)
- If a request arrives before its TAT, it's rejected (rate exceeded)
- If it arrives after, the TAT is updated for the next request
- The "tolerance" parameter allows small bursts without rejection

Usage:
    limiter = GCRARateLimiter(rate=10, burst_tolerance=5)
    allowed = limiter.check("tenant_123")
    if not allowed:
        raise RateLimitExceeded(...)
"""

import threading
import time
from typing import Optional


class GCRARateLimiter:
    """
    Generic Cell Rate Algorithm rate limiter.

    Provides smoother traffic shaping than token-bucket:
    - No sudden burst-then-empty pattern
    - Constant-rate enforcement with configurable tolerance
    - O(1) memory per tenant
    - Thread-safe
    """

    def __init__(self, rate: float = 10.0, burst_tolerance: int = 5):
        """
        Args:
            rate: Allowed requests per second per tenant
            burst_tolerance: Number of requests that can exceed steady rate
        """
        self._emission_interval = 1.0 / rate if rate > 0 else float('inf')
        self._delay_tolerance = burst_tolerance * self._emission_interval
        self._tat: dict[str, float] = {}  # tenant -> theoretical arrival time
        self._lock = threading.Lock()

    def check(self, tenant_id: str) -> bool:
        """
        Check if a request should be allowed.
        Returns True if allowed, False if rate exceeded.
        """
        now = time.monotonic()

        with self._lock:
            tat = self._tat.get(tenant_id, now)

            # New TAT = max(TAT, now) + emission_interval
            new_tat = max(tat, now) + self._emission_interval

            # Allow if new TAT is within tolerance
            if new_tat - now <= self._delay_tolerance + self._emission_interval:
                self._tat[tenant_id] = new_tat
                return True
            else:
                return False

    def check_and_wait(self, tenant_id: str, max_wait: float = 5.0) -> bool:
        """
        Check rate limit. If exceeded, wait up to max_wait seconds.
        Returns True if request can proceed, False if max_wait exceeded.
        """
        now = time.monotonic()
        deadline = now + max_wait

        while time.monotonic() < deadline:
            if self.check(tenant_id):
                return True
            time.sleep(self._emission_interval * 0.5)

        return False

    def get_wait_time(self, tenant_id: str) -> float:
        """Get time in seconds until next request would be allowed."""
        now = time.monotonic()
        with self._lock:
            tat = self._tat.get(tenant_id, now)
            wait = max(0.0, tat - now)
            return wait

    def reset(self, tenant_id: Optional[str] = None):
        """Reset rate limiter state."""
        with self._lock:
            if tenant_id:
                self._tat.pop(tenant_id, None)
            else:
                self._tat.clear()
