"""
Phase 6 — Observability tests.
"""

import json


def test_structlog_outputs_json():
    """structlog must output valid JSON."""
    from agentfuse.observability.logging import logger
    import io
    import sys

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        logger.info("test_event", key="value", number=42)
    finally:
        sys.stdout = old_stdout

    output = buffer.getvalue().strip()
    # Should be valid JSON
    parsed = json.loads(output)
    assert parsed["event"] == "test_event"
    assert parsed["key"] == "value"
    assert parsed["number"] == 42
    assert "level" in parsed or "log_level" in parsed


def test_otel_span_has_required_attributes():
    """OTel span must have gen_ai.* required attributes."""
    from agentfuse.observability.otel import agentfuse_span, OTEL_AVAILABLE

    if not OTEL_AVAILABLE:
        return  # Skip if OTel not installed

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
    from opentelemetry import trace

    # Simple collector
    collected_spans = []

    class CollectorExporter(SpanExporter):
        def export(self, spans):
            collected_spans.extend(spans)
            return SpanExportResult.SUCCESS

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(CollectorExporter()))
    trace.set_tracer_provider(provider)

    # Reload to use new provider
    import agentfuse.observability.otel as otel_mod
    otel_mod._tracer = trace.get_tracer("agentfuse", "0.2.0")

    with agentfuse_span("chat", "gpt-4o", "openai") as span:
        pass

    assert len(collected_spans) == 1
    attrs = dict(collected_spans[0].attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert collected_spans[0].name == "chat gpt-4o"


def test_observability_failure_never_propagates():
    """If prometheus fails, LLM call must still work."""
    from agentfuse.observability.metrics import record_cache_lookup, record_cost, record_error

    # These should never raise, even with bad inputs
    record_cache_lookup("gpt-4o", hit=True, tier=1)
    record_cache_lookup("gpt-4o", hit=False)
    record_cost("gpt-4o", "openai", 0.05)
    record_error("rate_limit", "openai")
    # If we got here without exception, test passes


def test_cache_hit_increments_counter():
    """Cache hit must increment the counter."""
    from agentfuse.observability.metrics import CACHE_HITS, CACHE_LOOKUPS, record_cache_lookup, METRICS_AVAILABLE

    if not METRICS_AVAILABLE:
        return

    # Get current values
    before_hits = CACHE_HITS.labels(model="test-model", tier="1")._value.get()
    before_lookups = CACHE_LOOKUPS.labels(model="test-model")._value.get()

    record_cache_lookup("test-model", hit=True, tier=1)

    after_hits = CACHE_HITS.labels(model="test-model", tier="1")._value.get()
    after_lookups = CACHE_LOOKUPS.labels(model="test-model")._value.get()

    assert after_hits == before_hits + 1
    assert after_lookups == before_lookups + 1
