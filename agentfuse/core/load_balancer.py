"""
ModelLoadBalancer — distributes requests across multiple API keys/endpoints.

Production systems like LiteLLM and Portkey let you configure multiple
API keys for the same model. When one key hits rate limits, requests
automatically route to another key.

Strategies:
- round_robin: rotate through keys evenly
- least_cost: prefer the cheapest available endpoint
- least_latency: prefer the fastest responding endpoint
- random: random selection (simplest, good for stateless deployments)

Usage:
    balancer = ModelLoadBalancer()
    balancer.add_endpoint("gpt-4o", api_key="sk-key1")
    balancer.add_endpoint("gpt-4o", api_key="sk-key2")
    balancer.add_endpoint("gpt-4o", api_key="sk-key3", base_url="https://custom.endpoint.com")

    endpoint = balancer.get_endpoint("gpt-4o")
    # Returns the next available endpoint based on strategy
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Endpoint:
    """A single API endpoint configuration."""
    model: str
    api_key: str
    base_url: Optional[str] = None
    weight: float = 1.0
    healthy: bool = True
    last_failure: float = 0.0
    total_requests: int = 0
    total_failures: int = 0
    avg_latency_ms: float = 0.0
    _latency_ema: float = 0.0


class ModelLoadBalancer:
    """
    Distributes LLM requests across multiple API keys/endpoints.

    Automatically marks endpoints as unhealthy after failures and
    recovers them after a cooldown period.
    """

    HEALTH_COOLDOWN = 60.0  # seconds before retrying unhealthy endpoint

    def __init__(self, strategy: str = "round_robin"):
        """
        Args:
            strategy: "round_robin", "least_latency", "random"
        """
        self._strategy = strategy
        self._endpoints: dict[str, list[Endpoint]] = {}  # model -> [endpoints]
        self._rr_index: dict[str, int] = {}  # round-robin counters
        self._lock = threading.Lock()

    def add_endpoint(self, model: str, api_key: str,
                     base_url: Optional[str] = None, weight: float = 1.0):
        """Register an API endpoint for a model."""
        with self._lock:
            if model not in self._endpoints:
                self._endpoints[model] = []
                self._rr_index[model] = 0

            self._endpoints[model].append(Endpoint(
                model=model, api_key=api_key,
                base_url=base_url, weight=weight,
            ))

    def get_endpoint(self, model: str) -> Optional[Endpoint]:
        """Get the next available endpoint for a model based on strategy."""
        with self._lock:
            endpoints = self._endpoints.get(model, [])
            if not endpoints:
                return None

            # Filter to healthy endpoints (or recover stale unhealthy ones)
            now = time.time()
            available = []
            for ep in endpoints:
                if ep.healthy:
                    available.append(ep)
                elif now - ep.last_failure > self.HEALTH_COOLDOWN:
                    ep.healthy = True  # recovery
                    available.append(ep)

            if not available:
                # All unhealthy — try the least recently failed
                available = sorted(endpoints, key=lambda e: e.last_failure)[:1]
                if available:
                    available[0].healthy = True

            if not available:
                return None

            if self._strategy == "round_robin":
                return self._round_robin(model, available)
            elif self._strategy == "least_latency":
                return min(available, key=lambda e: e.avg_latency_ms)
            elif self._strategy == "random":
                return random.choice(available)
            else:
                return available[0]

    def report_success(self, endpoint: Endpoint, latency_ms: float):
        """Report a successful request to update endpoint stats."""
        with self._lock:
            endpoint.total_requests += 1
            endpoint.healthy = True
            # Exponential moving average of latency
            alpha = 0.1
            endpoint._latency_ema = alpha * latency_ms + (1 - alpha) * endpoint._latency_ema
            endpoint.avg_latency_ms = endpoint._latency_ema

    def report_failure(self, endpoint: Endpoint):
        """Report a failed request to mark endpoint unhealthy."""
        with self._lock:
            endpoint.total_failures += 1
            endpoint.healthy = False
            endpoint.last_failure = time.time()
            logger.warning("Endpoint marked unhealthy: %s (key=%s...)",
                           endpoint.model, endpoint.api_key[:8])

    def get_stats(self) -> dict:
        """Get load balancer statistics."""
        with self._lock:
            stats = {}
            for model, endpoints in self._endpoints.items():
                stats[model] = {
                    "total_endpoints": len(endpoints),
                    "healthy": sum(1 for e in endpoints if e.healthy),
                    "total_requests": sum(e.total_requests for e in endpoints),
                    "total_failures": sum(e.total_failures for e in endpoints),
                }
            return stats

    def _round_robin(self, model: str, available: list[Endpoint]) -> Endpoint:
        idx = self._rr_index.get(model, 0) % len(available)
        self._rr_index[model] = idx + 1
        return available[idx]
