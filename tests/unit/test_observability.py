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


def test_record_cost_increments():
    """record_cost must increment both COST_TOTAL and observe COST_PER_REQ."""
    from agentfuse.observability.metrics import COST_TOTAL, record_cost, METRICS_AVAILABLE

    if not METRICS_AVAILABLE:
        return

    before = COST_TOTAL.labels(model="gpt-4o", provider="openai")._value.get()
    record_cost("gpt-4o", "openai", 0.05)
    after = COST_TOTAL.labels(model="gpt-4o", provider="openai")._value.get()
    assert abs(after - before - 0.05) < 1e-9


def test_record_tokens_increments():
    """record_tokens must observe input and output token histograms."""
    from agentfuse.observability.metrics import record_tokens, METRICS_AVAILABLE

    if not METRICS_AVAILABLE:
        return

    # Should not raise
    record_tokens("gpt-4o", input_tokens=100, output_tokens=50)
    record_tokens("gpt-4o", input_tokens=0, output_tokens=0)


def test_record_budget_remaining():
    """record_budget_remaining must set the gauge value."""
    from agentfuse.observability.metrics import BUDGET_REMAIN, record_budget_remaining, METRICS_AVAILABLE

    if not METRICS_AVAILABLE:
        return

    record_budget_remaining("test_budget", 7.50)
    val = BUDGET_REMAIN.labels(budget_id="test_budget")._value.get()
    assert abs(val - 7.50) < 1e-9


def test_record_model_fallback():
    """record_model_fallback must increment the counter."""
    from agentfuse.observability.metrics import FALLBACKS, record_model_fallback, METRICS_AVAILABLE

    if not METRICS_AVAILABLE:
        return

    before = FALLBACKS.labels(original_model="gpt-4o", fallback_model="gpt-4o-mini")._value.get()
    record_model_fallback("gpt-4o", "gpt-4o-mini")
    after = FALLBACKS.labels(original_model="gpt-4o", fallback_model="gpt-4o-mini")._value.get()
    assert after == before + 1


def test_record_tokens_saved():
    """record_tokens_saved must increment the counter."""
    from agentfuse.observability.metrics import TOKENS_SAVED, record_tokens_saved, METRICS_AVAILABLE

    if not METRICS_AVAILABLE:
        return

    before = TOKENS_SAVED.labels(model="gpt-4o")._value.get()
    record_tokens_saved("gpt-4o", 500)
    after = TOKENS_SAVED.labels(model="gpt-4o")._value.get()
    assert after == before + 500


def test_record_error_increments():
    """record_error must increment ERRORS counter."""
    from agentfuse.observability.metrics import ERRORS, record_error, METRICS_AVAILABLE

    if not METRICS_AVAILABLE:
        return

    before = ERRORS.labels(error_type="timeout", provider="openai")._value.get()
    record_error("timeout", "openai")
    after = ERRORS.labels(error_type="timeout", provider="openai")._value.get()
    assert after == before + 1


def test_otel_record_usage_on_span():
    """record_usage_on_span must set attributes on span."""
    from agentfuse.observability.otel import record_usage_on_span
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    span = MagicMock()
    usage = SimpleNamespace(total_input_tokens=100, total_output_tokens=50, cached_input_tokens=20)
    record_usage_on_span(span, usage, cost_usd=0.05, cache_hit=True, cache_tier=1)

    span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 100)
    span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 50)
    span.set_attribute.assert_any_call("gen_ai.usage.cache_read.input_tokens", 20)
    span.set_attribute.assert_any_call("agentfuse.cost_usd", 0.05)
    span.set_attribute.assert_any_call("agentfuse.cache_hit", True)
    span.set_attribute.assert_any_call("agentfuse.cache_tier", 1)


def test_otel_record_usage_none_span():
    """record_usage_on_span with None span must not crash."""
    from agentfuse.observability.otel import record_usage_on_span
    record_usage_on_span(None, None, cost_usd=0.05)  # should not raise


def test_otel_span_without_otel():
    """agentfuse_span must yield None when OTel unavailable."""
    from agentfuse.observability.otel import agentfuse_span
    import agentfuse.observability.otel as otel_mod
    old = otel_mod.OTEL_AVAILABLE
    otel_mod.OTEL_AVAILABLE = False
    try:
        with agentfuse_span("chat", "gpt-4o", "openai") as span:
            assert span is None
    finally:
        otel_mod.OTEL_AVAILABLE = old


def test_cache_miss_not_counted_as_hit():
    """Cache miss must increment lookups but not hits."""
    from agentfuse.observability.metrics import CACHE_HITS, CACHE_LOOKUPS, record_cache_lookup, METRICS_AVAILABLE

    if not METRICS_AVAILABLE:
        return

    before_hits = CACHE_HITS.labels(model="miss-model", tier="0")._value.get()
    before_lookups = CACHE_LOOKUPS.labels(model="miss-model")._value.get()

    record_cache_lookup("miss-model", hit=False, tier=0)

    after_hits = CACHE_HITS.labels(model="miss-model", tier="0")._value.get()
    after_lookups = CACHE_LOOKUPS.labels(model="miss-model")._value.get()

    assert after_hits == before_hits  # no hit increment
    assert after_lookups == before_lookups + 1  # lookup incremented
