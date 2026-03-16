"""
Structured JSON logging with OTel trace context injection.

Injects trace_id, span_id from OTel for correlation.
Also injects dd.trace_id, dd.span_id for Datadog (lower 64 bits as decimal).
"""

import logging

import structlog


def _add_trace_context(logger, method_name, event_dict):
    """Inject OTel trace/span IDs into log events."""
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            ctx = span.get_span_context()
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
            # Datadog format (lower 64 bits as decimal)
            event_dict["dd.trace_id"] = str(ctx.trace_id & 0xFFFFFFFFFFFFFFFF)
            event_dict["dd.span_id"] = str(ctx.span_id)
    except Exception:
        pass
    return event_dict


# Configure structlog with JSON output
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_trace_context,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("agentfuse")
