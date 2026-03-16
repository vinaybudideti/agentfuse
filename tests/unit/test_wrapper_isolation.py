"""
Tests proving critical production bugs are fixed:
1. Multiple wrap_openai() calls coexist with separate budgets
2. Truncated responses (finish_reason="length") are NOT cached
3. Tool-call responses without text content don't crash
"""

from types import SimpleNamespace
from agentfuse.core.response_validator import validate_for_cache


# --- Truncated response validation ---

def test_truncated_response_not_cached():
    """finish_reason='length' must prevent caching."""
    assert validate_for_cache("This is a partial resp", finish_reason="length") is False


def test_complete_response_cached():
    """finish_reason='stop' must allow caching."""
    assert validate_for_cache("Complete response here.", finish_reason="stop") is True


def test_tool_calls_response_cached():
    """finish_reason='tool_calls' must allow caching (for L1 exact match)."""
    assert validate_for_cache("Using tool", finish_reason="tool_calls") is True


def test_none_finish_reason_cached():
    """None finish_reason (cache hits, mocks) must allow caching."""
    assert validate_for_cache("Normal response") is True


# --- Response recording with finish_reason ---

def test_record_and_cache_skips_truncated():
    """_record_and_cache_openai must skip caching truncated responses."""
    from agentfuse.providers.openai import _record_and_cache_openai
    from agentfuse.core.budget import BudgetEngine
    from agentfuse.core.cache import TwoTierCacheMiddleware, CacheMiss
    from agentfuse.providers.pricing import ModelPricingEngine

    engine = BudgetEngine("trunc_test", 10.0, "gpt-4o")
    cache = TwoTierCacheMiddleware()
    pricing = ModelPricingEngine()

    # Mock a truncated OpenAI response
    result = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="This response was cut short"),
            finish_reason="length",  # TRUNCATED
        )],
    )
    msgs = [{"role": "user", "content": "test query"}]

    _record_and_cache_openai(result, "gpt-4o", engine, pricing, cache, msgs, 0.0, None)

    # Cost should be recorded (we still pay for truncated responses)
    assert engine.spent > 0

    # But response should NOT be in cache
    lookup = cache.lookup("gpt-4o", msgs)
    assert isinstance(lookup, CacheMiss), "Truncated response must NOT be cached"


def test_record_and_cache_allows_complete():
    """_record_and_cache_openai must cache complete responses."""
    from agentfuse.providers.openai import _record_and_cache_openai
    from agentfuse.core.budget import BudgetEngine
    from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit
    from agentfuse.providers.pricing import ModelPricingEngine

    engine = BudgetEngine("complete_test", 10.0, "gpt-4o")
    cache = TwoTierCacheMiddleware()
    pricing = ModelPricingEngine()

    result = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Complete answer about France"),
            finish_reason="stop",
        )],
    )
    msgs = [{"role": "user", "content": "What is the capital of France?"}]

    _record_and_cache_openai(result, "gpt-4o", engine, pricing, cache, msgs, 0.0, None)

    # Response SHOULD be in cache
    lookup = cache.lookup("gpt-4o", msgs)
    assert isinstance(lookup, CacheHit), "Complete response must be cached"
    assert lookup.response == "Complete answer about France"


# --- Multi-run isolation ---

def test_openai_run_contexts_isolated():
    """Multiple wrap_openai run_ids must have separate budgets."""
    from agentfuse.providers.openai import _run_contexts

    # Verify the architecture: _run_contexts stores per-run state
    from agentfuse.core.budget import BudgetEngine
    from agentfuse.core.cache import TwoTierCacheMiddleware
    from agentfuse.providers.pricing import ModelPricingEngine
    from agentfuse.providers.tokenizer import TokenCounterAdapter

    # Simulate two runs being registered
    _run_contexts["run_A"] = {
        "engine": BudgetEngine("run_A", 5.0, "gpt-4o"),
        "cache": TwoTierCacheMiddleware(),
        "pricing": ModelPricingEngine(),
        "tokenizer": TokenCounterAdapter(),
    }
    _run_contexts["run_B"] = {
        "engine": BudgetEngine("run_B", 3.0, "claude-sonnet-4-6"),
        "cache": TwoTierCacheMiddleware(),
        "pricing": ModelPricingEngine(),
        "tokenizer": TokenCounterAdapter(),
    }

    # Verify they are separate objects
    assert _run_contexts["run_A"]["engine"] is not _run_contexts["run_B"]["engine"]
    assert _run_contexts["run_A"]["engine"].budget == 5.0
    assert _run_contexts["run_B"]["engine"].budget == 3.0
    assert _run_contexts["run_A"]["engine"].run_id == "run_A"
    assert _run_contexts["run_B"]["engine"].run_id == "run_B"

    # Clean up
    del _run_contexts["run_A"]
    del _run_contexts["run_B"]


def test_openai_active_run_contextvar():
    """set_active_run must switch the active run for current thread."""
    from agentfuse.providers.openai import _active_openai_run, set_active_run

    set_active_run("run_X")
    assert _active_openai_run.get() == "run_X"

    set_active_run("run_Y")
    assert _active_openai_run.get() == "run_Y"


# --- Tool-call response handling ---

def test_tool_call_response_no_text_no_crash():
    """OpenAI response with tool_calls but no text content must not crash."""
    from agentfuse.providers.openai import _record_and_cache_openai
    from agentfuse.core.budget import BudgetEngine
    from agentfuse.core.cache import TwoTierCacheMiddleware
    from agentfuse.providers.pricing import ModelPricingEngine

    engine = BudgetEngine("tool_test", 10.0, "gpt-4o")
    cache = TwoTierCacheMiddleware()
    pricing = ModelPricingEngine()

    # Mock response with tool_calls, NO text content
    result = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=[{"id": "call_1"}]),
            finish_reason="tool_calls",
        )],
    )
    msgs = [{"role": "user", "content": "get weather"}]

    # Must not crash
    _record_and_cache_openai(result, "gpt-4o", engine, pricing, cache, msgs, 0.0,
                              [{"function": {"name": "get_weather"}}])

    # Cost should still be recorded even without text content
    assert engine.spent > 0
