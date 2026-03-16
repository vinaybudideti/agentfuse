"""
Tests for the middleware pipeline — composable request/response processing.
"""

from agentfuse.core.middleware import (
    MiddlewarePipeline, MiddlewareStage, LLMRequest, LLMResponse,
    LoggingStage, TimingStage,
)


class CounterStage(MiddlewareStage):
    def __init__(self):
        self.before_count = 0
        self.after_count = 0

    def before_call(self, request):
        self.before_count += 1
        return None

    def after_call(self, request, response):
        self.after_count += 1
        return response


class ShortCircuitStage(MiddlewareStage):
    def before_call(self, request):
        return LLMResponse(text="short-circuited", cached=True)


class ErrorRecoveryStage(MiddlewareStage):
    def on_error(self, request, error):
        return LLMResponse(text="recovered", metadata={"error": str(error)})


def test_pipeline_executes_in_order():
    """Middleware stages must execute before_call in order."""
    pipeline = MiddlewarePipeline()
    s1 = CounterStage()
    s2 = CounterStage()
    pipeline.add(s1).add(s2)

    request = LLMRequest(model="gpt-4o", messages=[])
    pipeline.execute(request, lambda r: "raw_result")

    assert s1.before_count == 1
    assert s2.before_count == 1
    assert s1.after_count == 1
    assert s2.after_count == 1


def test_pipeline_after_call_reverse_order():
    """after_call must execute in REVERSE order."""
    order = []

    class OrderStage(MiddlewareStage):
        def __init__(self, name):
            self.name = name

        def after_call(self, request, response):
            order.append(self.name)
            return response

    pipeline = MiddlewarePipeline()
    pipeline.add(OrderStage("first")).add(OrderStage("second")).add(OrderStage("third"))

    request = LLMRequest(model="gpt-4o", messages=[])
    pipeline.execute(request, lambda r: "result")

    assert order == ["third", "second", "first"]


def test_pipeline_short_circuit():
    """Short-circuit in before_call must skip LLM call."""
    call_made = False

    def fake_call(r):
        nonlocal call_made
        call_made = True
        return "should not reach"

    pipeline = MiddlewarePipeline()
    pipeline.add(ShortCircuitStage())

    request = LLMRequest(model="gpt-4o", messages=[])
    response = pipeline.execute(request, fake_call)

    assert response.text == "short-circuited"
    assert response.cached is True
    assert not call_made  # LLM call was skipped


def test_pipeline_error_recovery():
    """on_error must allow recovery from LLM call failure."""
    def failing_call(r):
        raise RuntimeError("API down")

    pipeline = MiddlewarePipeline()
    pipeline.add(ErrorRecoveryStage())

    request = LLMRequest(model="gpt-4o", messages=[])
    response = pipeline.execute(request, failing_call)

    assert response.text == "recovered"
    assert response.metadata["error"] == "API down"


def test_pipeline_error_no_recovery_raises():
    """Without error recovery, pipeline must re-raise."""
    import pytest

    def failing_call(r):
        raise ValueError("bad request")

    pipeline = MiddlewarePipeline()
    pipeline.add(CounterStage())

    request = LLMRequest(model="gpt-4o", messages=[])
    with pytest.raises(ValueError, match="bad request"):
        pipeline.execute(request, failing_call)


def test_timing_stage():
    """TimingStage must add latency_ms to response."""
    import time

    pipeline = MiddlewarePipeline()
    pipeline.add(TimingStage())

    request = LLMRequest(model="gpt-4o", messages=[])

    def slow_call(r):
        time.sleep(0.01)
        return "result"

    response = pipeline.execute(request, slow_call)
    assert response.latency_ms >= 10  # at least 10ms


def test_logging_stage_no_crash():
    """LoggingStage must not crash."""
    pipeline = MiddlewarePipeline()
    pipeline.add(LoggingStage())

    request = LLMRequest(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    response = pipeline.execute(request, lambda r: "result")
    assert response is not None


def test_llm_request_defaults():
    """LLMRequest must have sensible defaults."""
    req = LLMRequest(model="gpt-4o", messages=[])
    assert req.temperature == 0.0
    assert req.stream is False
    assert req.tools is None
    assert req.metadata == {}


def test_llm_response_defaults():
    """LLMResponse must have sensible defaults."""
    resp = LLMResponse()
    assert resp.text == ""
    assert resp.cost_usd == 0.0
    assert resp.cached is False
    assert resp.latency_ms == 0.0
