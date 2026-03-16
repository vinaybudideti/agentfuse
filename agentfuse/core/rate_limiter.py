"""
TokenBucketRateLimiter — per-tenant rate limiting for LLM API calls.

Uses token bucket algorithm: each tenant gets N tokens per second,
each API call consumes 1 token. When bucket is empty, calls are
delayed or rejected.

This prevents a single tenant from exhausting shared API rate limits
and ensures fair resource distribution across tenants.
"""

import threading
import time
from typing import Optional


class RateLimitExceeded(Exception):
    """Raised when a tenant exceeds their rate limit."""
    def __init__(self, tenant_id: str, retry_after: float):
        self.tenant_id = tenant_id
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for tenant '{tenant_id}'. "
            f"Retry after {retry_after:.1f}s"
        )


class TokenBucketRateLimiter:
    """
    Per-tenant rate limiter using token bucket algorithm.

    Each tenant gets `rate` tokens per second, up to `burst` tokens max.
    Each API call consumes 1 token. If the bucket is empty:
    - block=True: wait until a token is available (up to max_wait)
    - block=False: raise RateLimitExceeded immediately

    Usage:
        limiter = TokenBucketRateLimiter(rate=10, burst=20)
        limiter.acquire("tenant_123")  # blocks if rate exceeded
    """

    def __init__(self, rate: float = 10.0, burst: int = 20, max_wait: float = 30.0):
        """
        Args:
            rate: Tokens replenished per second per tenant
            burst: Maximum tokens in the bucket (peak capacity)
            max_wait: Maximum seconds to wait for a token (0 = no waiting)
        """
        self._rate = rate
        self._burst = burst
        self._max_wait = max_wait
        self._buckets: dict[str, dict] = {}  # tenant_id -> {tokens, last_refill}
        self._lock = threading.Lock()

    def acquire(self, tenant_id: str, block: bool = True) -> bool:
        """
        Acquire a token for the given tenant.

        Returns True if token acquired, raises RateLimitExceeded if not.
        """
        with self._lock:
            bucket = self._get_or_create_bucket(tenant_id)
            self._refill(bucket)

            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True

            if not block or self._max_wait <= 0:
                wait_time = (1.0 - bucket["tokens"]) / self._rate
                raise RateLimitExceeded(tenant_id, retry_after=wait_time)

        # Block until token available (release lock while waiting)
        deadline = time.monotonic() + self._max_wait
        while time.monotonic() < deadline:
            time.sleep(min(0.1, 1.0 / self._rate))
            with self._lock:
                bucket = self._buckets[tenant_id]
                self._refill(bucket)
                if bucket["tokens"] >= 1.0:
                    bucket["tokens"] -= 1.0
                    return True

        raise RateLimitExceeded(tenant_id, retry_after=1.0 / self._rate)

    def _get_or_create_bucket(self, tenant_id: str) -> dict:
        if tenant_id not in self._buckets:
            self._buckets[tenant_id] = {
                "tokens": float(self._burst),
                "last_refill": time.monotonic(),
            }
        return self._buckets[tenant_id]

    def _refill(self, bucket: dict):
        now = time.monotonic()
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(
            self._burst,
            bucket["tokens"] + elapsed * self._rate,
        )
        bucket["last_refill"] = now

    def get_remaining(self, tenant_id: str) -> float:
        """Get remaining tokens for a tenant."""
        with self._lock:
            bucket = self._get_or_create_bucket(tenant_id)
            self._refill(bucket)
            return bucket["tokens"]

    def reset(self, tenant_id: Optional[str] = None):
        """Reset rate limiter state for a tenant or all tenants."""
        with self._lock:
            if tenant_id:
                self._buckets.pop(tenant_id, None)
            else:
                self._buckets.clear()
