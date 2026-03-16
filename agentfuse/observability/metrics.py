"""
Prometheus metrics for AgentFuse.

OTel-aligned GenAI metrics + AgentFuse-specific cost optimization metrics.
All metric operations are wrapped in try/except — never propagate to user code.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge

    # OTel-aligned GenAI metrics
    TOKEN_USAGE = Histogram(
        "gen_ai_client_token_usage",
        "Token usage per operation",
        labelnames=["model", "token_type"],
        buckets=[1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576],
    )

    OPERATION_DURATION = Histogram(
        "gen_ai_client_operation_duration_seconds",
        "Operation duration in seconds",
        labelnames=["model", "operation"],
        buckets=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92],
    )

    # AgentFuse-specific metrics
    CACHE_HITS = Counter(
        "agentfuse_cache_hits_total",
        "Cache hit count",
        labelnames=["model", "tier"],
    )

    CACHE_LOOKUPS = Counter(
        "agentfuse_cache_lookups_total",
        "Cache lookup count",
        labelnames=["model"],
    )

    COST_TOTAL = Counter(
        "agentfuse_cost_usd_total",
        "Cumulative cost in USD",
        labelnames=["model", "provider"],
    )

    COST_PER_REQ = Histogram(
        "agentfuse_cost_per_request_usd",
        "Cost per request in USD",
        labelnames=["model"],
        buckets=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    )

    BUDGET_REMAIN = Gauge(
        "agentfuse_budget_remaining_usd",
        "Remaining budget in USD",
        labelnames=["budget_id"],
    )

    TOKENS_SAVED = Counter(
        "agentfuse_tokens_saved_total",
        "Tokens avoided via caching",
        labelnames=["model"],
    )

    ERRORS = Counter(
        "agentfuse_errors_total",
        "Error count by type and provider",
        labelnames=["error_type", "provider"],
    )

    FALLBACKS = Counter(
        "agentfuse_model_fallbacks_total",
        "Model fallback count",
        labelnames=["original_model", "fallback_model"],
    )

    METRICS_AVAILABLE = True

except ImportError:
    METRICS_AVAILABLE = False
    logger.info("prometheus_client not installed, metrics disabled")


def record_cache_lookup(model: str, hit: bool, tier: int = 0):
    """Record a cache lookup. Never raises."""
    if not METRICS_AVAILABLE:
        return
    try:
        CACHE_LOOKUPS.labels(model=model).inc()
        if hit:
            CACHE_HITS.labels(model=model, tier=str(tier)).inc()
    except Exception:
        pass


def record_cost(model: str, provider: str, cost_usd: float):
    """Record cost metrics. Never raises."""
    if not METRICS_AVAILABLE:
        return
    try:
        COST_TOTAL.labels(model=model, provider=provider).inc(cost_usd)
        COST_PER_REQ.labels(model=model).observe(cost_usd)
    except Exception:
        pass


def record_error(error_type: str, provider: str):
    """Record an error. Never raises."""
    if not METRICS_AVAILABLE:
        return
    try:
        ERRORS.labels(error_type=error_type, provider=provider).inc()
    except Exception:
        pass


def record_tokens(model: str, input_tokens: int = 0, output_tokens: int = 0):
    """Record token usage. Never raises."""
    if not METRICS_AVAILABLE:
        return
    try:
        if input_tokens:
            TOKEN_USAGE.labels(model=model, token_type="input").observe(input_tokens)
        if output_tokens:
            TOKEN_USAGE.labels(model=model, token_type="output").observe(output_tokens)
    except Exception:
        pass


def record_budget_remaining(budget_id: str, remaining_usd: float):
    """Update budget remaining gauge. Never raises."""
    if not METRICS_AVAILABLE:
        return
    try:
        BUDGET_REMAIN.labels(budget_id=budget_id).set(remaining_usd)
    except Exception:
        pass


def record_model_fallback(original_model: str, fallback_model: str):
    """Record a model fallback/downgrade. Never raises."""
    if not METRICS_AVAILABLE:
        return
    try:
        FALLBACKS.labels(original_model=original_model, fallback_model=fallback_model).inc()
    except Exception:
        pass


def record_tokens_saved(model: str, tokens: int):
    """Record tokens saved via caching. Never raises."""
    if not METRICS_AVAILABLE:
        return
    try:
        TOKENS_SAVED.labels(model=model).inc(tokens)
    except Exception:
        pass
