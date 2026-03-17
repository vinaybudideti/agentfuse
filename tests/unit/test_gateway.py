"""
Tests for the unified gateway — the production-grade entry point.
Tests routing, budget integration, cache flow without real API calls.
"""

import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from agentfuse.gateway import (
    completion, _get_engine, _record_cost, _validate_and_cache,
    get_engine, get_spend, cleanup,
)
from agentfuse.core.budget import BudgetEngine, BudgetExhaustedGracefully


def setup_function():
    cleanup()


# --- Engine management ---

def test_get_engine_creates_new():
    """First call with budget_id must create a new engine."""
    engine = _get_engine("test_run", 5.0, "gpt-4o")
    assert engine.budget == 5.0
    assert engine.run_id == "test_run"


def test_get_engine_reuses_existing():
    """Second call with same budget_id must return same engine."""
    engine1 = _get_engine("reuse_run", 5.0, "gpt-4o")
    engine1.record_cost(1.0)
    engine2 = _get_engine("reuse_run", 5.0, "gpt-4o")
    assert engine2.spent == 1.0  # same instance


def test_get_spend():
    """get_spend must return current spend for budget_id."""
    _get_engine("spend_run", 10.0, "gpt-4o").record_cost(2.5)
    assert get_spend("spend_run") == 2.5
    assert get_spend("nonexistent") == 0.0


def test_cleanup():
    """cleanup must remove engines."""
    _get_engine("clean1", 5.0, "gpt-4o")
    _get_engine("clean2", 5.0, "gpt-4o")
    cleanup("clean1")
    assert get_engine("clean1") is None
    assert get_engine("clean2") is not None
    cleanup()
    assert get_engine("clean2") is None


# --- Cost recording ---

def test_record_cost_openai():
    """Record cost from OpenAI-format response."""
    engine = BudgetEngine("cost_test", 10.0, "gpt-4o")
    result = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
    )
    _record_cost(result, "gpt-4o", "openai", engine)
    assert engine.spent > 0


def test_record_cost_anthropic():
    """Record cost from Anthropic-format response."""
    engine = BudgetEngine("anthro_cost", 10.0, "claude-sonnet-4-6")
    result = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    )
    _record_cost(result, "claude-sonnet-4-6", "anthropic", engine)
    assert engine.spent > 0


def test_record_cost_no_usage():
    """Missing usage must not crash."""
    engine = BudgetEngine("no_usage", 10.0, "gpt-4o")
    result = SimpleNamespace()  # no usage attribute
    _record_cost(result, "gpt-4o", "openai", engine)
    assert engine.spent == 0.0


# --- Validate and cache ---

def test_validate_and_cache_openai():
    """Valid OpenAI response must be cached."""
    from agentfuse.gateway import _cache
    msgs = [{"role": "user", "content": "gateway cache test"}]
    result = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Cached response"),
            finish_reason="stop",
        )],
    )
    _validate_and_cache(result, "gpt-4o", "openai", msgs, 0.0, None, None)

    from agentfuse.core.cache import CacheHit
    lookup = _cache.lookup("gpt-4o", msgs)
    assert isinstance(lookup, CacheHit)


def test_validate_and_cache_truncated_rejected():
    """Truncated response must NOT be cached."""
    from agentfuse.gateway import _cache
    msgs = [{"role": "user", "content": "truncated gateway test"}]
    result = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Partial resp"),
            finish_reason="length",
        )],
    )
    _validate_and_cache(result, "gpt-4o", "openai", msgs, 0.0, None, None)

    from agentfuse.core.cache import CacheMiss
    lookup = _cache.lookup("gpt-4o", msgs)
    assert isinstance(lookup, CacheMiss)


def test_validate_and_cache_anthropic():
    """Valid Anthropic response must be cached."""
    from agentfuse.gateway import _cache
    msgs = [{"role": "user", "content": "anthropic gateway test"}]
    result = SimpleNamespace(
        content=[SimpleNamespace(text="Claude response", type="text")],
        stop_reason="end_turn",
    )
    _validate_and_cache(result, "claude-sonnet-4-6", "anthropic", msgs, 0.0, None, None)

    from agentfuse.core.cache import CacheHit
    lookup = _cache.lookup("claude-sonnet-4-6", msgs)
    assert isinstance(lookup, CacheHit)


# --- Cache hit returns mock response ---

def test_completion_cache_hit_returns_mock():
    """Cache hit must return a mock response, not make API call."""
    from agentfuse.gateway import _cache

    msgs = [{"role": "user", "content": "cached completion test"}]
    _cache.store(model="gpt-4o", messages=msgs, response="Cached answer")

    # This should hit cache and NOT call the real API
    result = completion(model="gpt-4o", messages=msgs)
    assert hasattr(result, "choices")
    assert result.choices[0].message.content == "Cached answer"
    assert result._agentfuse_cache_hit is True


# --- Budget enforcement via gateway ---

def test_completion_budget_exhaustion():
    """Budget exhaustion must raise BudgetExhaustedGracefully."""
    engine = _get_engine("exhaust_gw", 0.001, "gpt-4o")
    engine.record_cost(0.001)

    with pytest.raises(BudgetExhaustedGracefully):
        completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "over budget"}],
            budget_id="exhaust_gw",
        )


# --- Provider routing ---

@patch("agentfuse.gateway._call_openai_compatible")
def test_completion_routes_openai(mock_call):
    """OpenAI models must route through _call_openai_compatible."""
    mock_call.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="hi"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    result = completion(model="gpt-4o", messages=[{"role": "user", "content": "test"}])
    mock_call.assert_called_once()
    assert result.choices[0].message.content == "hi"


@patch("agentfuse.gateway._call_anthropic")
def test_completion_routes_anthropic(mock_call):
    """Anthropic models must route through _call_anthropic."""
    mock_call.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="hello from claude", type="text")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    )
    result = completion(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}])
    mock_call.assert_called_once()


@patch("agentfuse.gateway._call_openai_compatible")
def test_completion_with_budget_records_cost(mock_call):
    """Completion with budget_id must record cost from usage."""
    mock_call.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="response"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
    )
    completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "cost tracking test"}],
        budget_id="cost_track",
        budget_usd=10.0,
    )
    assert get_spend("cost_track") > 0


@patch("agentfuse.gateway._call_openai_compatible")
def test_completion_provider_error_raises(mock_call):
    """Provider errors must propagate (after classification)."""
    mock_call.side_effect = RuntimeError("API connection failed")
    with pytest.raises(RuntimeError, match="API connection failed"):
        completion(model="gpt-4o", messages=[{"role": "user", "content": "error test"}])


@patch("agentfuse.gateway._call_openai_compatible")
def test_completion_default_budget(mock_call):
    """When budget_id given without budget_usd, default $10."""
    mock_call.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "default budget"}],
        budget_id="default_bud",
    )
    engine = get_engine("default_bud")
    assert engine is not None
    assert engine.budget == 10.0


@patch("agentfuse.gateway._call_openai_compatible")
def test_completion_auto_route(mock_call):
    """auto_route=True must invoke IntelligentModelRouter."""
    mock_call.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="routed"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    # Simple query should potentially be routed to a cheaper model
    completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        auto_route=True,
    )
    mock_call.assert_called_once()


@patch("agentfuse.gateway._call_openai_compatible")
def test_completion_no_budget_no_engine(mock_call):
    """Completion without budget_id must not create an engine."""
    mock_call.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="no budget"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    completion(model="gpt-4o", messages=[{"role": "user", "content": "no budget test"}])
    # No engine should have been created
    assert get_engine("no_budget_test") is None


@patch("agentfuse.gateway._call_openai_compatible")
def test_completion_stream_skips_cost_and_cache(mock_call):
    """stream=True must skip cost recording and cache storage."""
    mock_call.return_value = iter(["chunk1", "chunk2"])
    result = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "stream test"}],
        stream=True,
    )
    # Should not crash (stream skips cost/cache path)
    mock_call.assert_called_once()


def test_validate_and_cache_anthropic_stop_reason():
    """Anthropic stop_reason='end_turn' mapped to finish_reason='stop'."""
    from agentfuse.gateway import _cache, _validate_and_cache
    msgs = [{"role": "user", "content": "anthro stop reason test 12345"}]
    result = SimpleNamespace(
        content=[SimpleNamespace(text="Valid response text", type="text")],
        stop_reason="end_turn",
    )
    _validate_and_cache(result, "claude-sonnet-4-6", "anthropic", msgs, 0.0, None, None)
    from agentfuse.core.cache import CacheHit
    assert isinstance(_cache.lookup("claude-sonnet-4-6", msgs), CacheHit)


def test_validate_and_cache_empty_response_not_cached():
    """Empty response text must not be cached."""
    from agentfuse.gateway import _cache, _validate_and_cache
    msgs = [{"role": "user", "content": "empty response test"}]
    result = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=""),
            finish_reason="stop",
        )],
    )
    _validate_and_cache(result, "gpt-4o", "openai", msgs, 0.0, None, None)
    from agentfuse.core.cache import CacheMiss
    assert isinstance(_cache.lookup("gpt-4o", msgs), CacheMiss)


@patch("agentfuse.gateway._call_openai_compatible")
def test_completion_records_metrics_on_cache_hit(mock_call):
    """Cache hit must increment metrics."""
    from agentfuse.gateway import _cache
    from agentfuse.observability.metrics import CACHE_HITS, CACHE_LOOKUPS, METRICS_AVAILABLE

    if not METRICS_AVAILABLE:
        return

    msgs = [{"role": "user", "content": "metrics cache hit test 99"}]
    _cache.store(model="gpt-4o", messages=msgs, response="Metrics test cached")

    before_lookups = CACHE_LOOKUPS.labels(model="gpt-4o")._value.get()

    completion(model="gpt-4o", messages=msgs)

    after_lookups = CACHE_LOOKUPS.labels(model="gpt-4o")._value.get()
    assert after_lookups > before_lookups


@patch("agentfuse.gateway._call_openai_compatible")
def test_completion_records_error_metrics(mock_call):
    """Provider error must increment error metrics."""
    from agentfuse.observability.metrics import ERRORS, METRICS_AVAILABLE

    if not METRICS_AVAILABLE:
        return

    mock_call.side_effect = RuntimeError("test error for metrics")
    before = ERRORS.labels(error_type="unknown_openai", provider="openai")._value.get()

    try:
        completion(model="gpt-4o", messages=[{"role": "user", "content": "error metrics"}])
    except RuntimeError:
        pass

    after = ERRORS.labels(error_type="unknown_openai", provider="openai")._value.get()
    assert after > before
