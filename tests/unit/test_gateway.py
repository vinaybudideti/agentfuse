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
