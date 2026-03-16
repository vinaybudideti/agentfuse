"""
OTel GenAI semantic convention spans (semconv v1.40).

Span name format: "{operation} {model}" (e.g. "chat gpt-4o")
Span kind: CLIENT
"""

import contextlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, StatusCode
    _tracer = trace.get_tracer("agentfuse", "0.2.0")
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    _tracer = None


@contextlib.contextmanager
def agentfuse_span(operation: str, model: str, provider: str = "unknown"):
    """
    Context manager that creates an OTel span with GenAI semconv attributes.

    Usage:
        with agentfuse_span("chat", "gpt-4o", "openai") as span:
            response = client.chat(...)
            record_usage_on_span(span, usage, cost)
    """
    if not OTEL_AVAILABLE or _tracer is None:
        yield None
        return

    span_name = f"{operation} {model}"
    with _tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
        try:
            span.set_attribute("gen_ai.operation.name", operation)
            span.set_attribute("gen_ai.provider.name", provider)
            span.set_attribute("gen_ai.request.model", model)
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


def record_usage_on_span(
    span,
    usage=None,
    cost_usd: float = 0.0,
    cache_hit: bool = False,
    cache_tier: Optional[int] = None,
):
    """Record token usage and cost on an existing span."""
    if span is None:
        return

    try:
        if usage:
            span.set_attribute("gen_ai.usage.input_tokens", getattr(usage, "total_input_tokens", 0))
            span.set_attribute("gen_ai.usage.output_tokens", getattr(usage, "total_output_tokens", 0))
            if hasattr(usage, "cached_input_tokens") and usage.cached_input_tokens:
                span.set_attribute("gen_ai.usage.cache_read.input_tokens", usage.cached_input_tokens)

        span.set_attribute("agentfuse.cost_usd", cost_usd)
        span.set_attribute("agentfuse.cache_hit", cache_hit)
        if cache_tier is not None:
            span.set_attribute("agentfuse.cache_tier", cache_tier)
    except Exception as e:
        logger.debug("Failed to record usage on span: %s", e)
