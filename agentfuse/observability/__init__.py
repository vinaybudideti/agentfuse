# AgentFuse observability — OTel spans, structlog logging, Prometheus metrics
from agentfuse.observability.otel import agentfuse_span, record_usage_on_span
from agentfuse.observability.metrics import record_cache_lookup, record_cost, record_error, record_tokens
