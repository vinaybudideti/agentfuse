"""
UsageLimits — configurable per-user/per-team usage quotas.

Production companies need usage limits beyond just dollar budgets:
- Max requests per hour per user
- Max tokens per day per team
- Max cost per month per project

This module provides hierarchical usage limits with automatic enforcement.

Usage:
    limits = UsageLimits()
    limits.set_limit("user:alice", max_requests_per_hour=100, max_cost_per_day=10.0)
    limits.check("user:alice")  # raises UsageLimitExceeded if over
    limits.record("user:alice", cost=0.05)
"""

import logging
import threading
import time
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class UsageLimitExceeded(Exception):
    """Raised when a usage limit is exceeded."""
    def __init__(self, entity_id: str, limit_type: str, current: float, limit: float):
        self.entity_id = entity_id
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        super().__init__(
            f"Usage limit exceeded for '{entity_id}': "
            f"{limit_type} = {current:.2f} (limit: {limit:.2f})"
        )


@dataclass
class LimitConfig:
    """Configuration for an entity's usage limits."""
    max_requests_per_hour: Optional[int] = None
    max_cost_per_day: Optional[float] = None
    max_tokens_per_day: Optional[int] = None


class UsageLimits:
    """
    Per-entity usage limits with automatic enforcement.

    Supports any entity: user:alice, team:engineering, project:chat-bot
    """

    def __init__(self):
        self._limits: dict[str, LimitConfig] = {}
        self._usage: dict[str, dict] = {}
        self._lock = threading.Lock()

    def set_limit(
        self,
        entity_id: str,
        max_requests_per_hour: Optional[int] = None,
        max_cost_per_day: Optional[float] = None,
        max_tokens_per_day: Optional[int] = None,
    ):
        """Set usage limits for an entity."""
        self._limits[entity_id] = LimitConfig(
            max_requests_per_hour=max_requests_per_hour,
            max_cost_per_day=max_cost_per_day,
            max_tokens_per_day=max_tokens_per_day,
        )
        if entity_id not in self._usage:
            self._usage[entity_id] = self._empty_usage()

    def record(self, entity_id: str, cost: float = 0.0, tokens: int = 0):
        """Record usage for an entity."""
        with self._lock:
            if entity_id not in self._usage:
                self._usage[entity_id] = self._empty_usage()

            usage = self._usage[entity_id]
            now = time.time()

            # Clean old hourly data
            usage["hourly_requests"] = [
                t for t in usage["hourly_requests"] if now - t < 3600
            ]
            usage["hourly_requests"].append(now)
            usage["daily_cost"] += cost
            usage["daily_tokens"] += tokens

            # Reset daily counters if new day
            if now - usage["day_start"] > 86400:
                usage["daily_cost"] = cost
                usage["daily_tokens"] = tokens
                usage["day_start"] = now

    def check(self, entity_id: str):
        """Check if entity is within limits. Raises UsageLimitExceeded if not."""
        config = self._limits.get(entity_id)
        if config is None:
            return  # no limits configured

        with self._lock:
            usage = self._usage.get(entity_id, self._empty_usage())
            now = time.time()

            # Check requests per hour
            if config.max_requests_per_hour is not None:
                recent = [t for t in usage["hourly_requests"] if now - t < 3600]
                if len(recent) >= config.max_requests_per_hour:
                    raise UsageLimitExceeded(
                        entity_id, "requests_per_hour",
                        float(len(recent)), float(config.max_requests_per_hour)
                    )

            # Check cost per day
            if config.max_cost_per_day is not None:
                if usage["daily_cost"] >= config.max_cost_per_day:
                    raise UsageLimitExceeded(
                        entity_id, "cost_per_day",
                        usage["daily_cost"], config.max_cost_per_day
                    )

            # Check tokens per day
            if config.max_tokens_per_day is not None:
                if usage["daily_tokens"] >= config.max_tokens_per_day:
                    raise UsageLimitExceeded(
                        entity_id, "tokens_per_day",
                        float(usage["daily_tokens"]), float(config.max_tokens_per_day)
                    )

    def get_usage(self, entity_id: str) -> dict:
        """Get current usage for an entity."""
        with self._lock:
            usage = self._usage.get(entity_id, self._empty_usage())
            now = time.time()
            recent_requests = [t for t in usage["hourly_requests"] if now - t < 3600]
            return {
                "requests_this_hour": len(recent_requests),
                "cost_today": round(usage["daily_cost"], 6),
                "tokens_today": usage["daily_tokens"],
            }

    @staticmethod
    def _empty_usage() -> dict:
        return {
            "hourly_requests": [],
            "daily_cost": 0.0,
            "daily_tokens": 0,
            "day_start": time.time(),
        }
