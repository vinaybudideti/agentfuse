"""
Regression tests — prevent previously fixed bugs from returning.
"""

from types import SimpleNamespace
from agentfuse.providers.response import extract_usage
from agentfuse.core.budget import BudgetEngine, BudgetExhaustedGracefully
from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit, CacheMiss
from agentfuse.core.keys import extract_semantic_content, build_l1_cache_key
from agentfuse.core.response_validator import validate_for_cache
from agentfuse.core.error_classifier import classify_error
import pytest


# --- Anthropic billing regression ---

def test_anthropic_input_tokens_excludes_cached():
    """REGRESSION: Anthropic input_tokens EXCLUDES cached — must add cache fields."""
    usage = SimpleNamespace(
        input_tokens=100,  # only uncached
        output_tokens=50,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=50,
    )
    result = extract_usage("anthropic", usage)
    # Total must include all: 100 + 200 + 50 = 350
    assert result.total_input_tokens == 350


def test_openai_reasoning_not_double_counted():
    """REGRESSION: OpenAI reasoning_tokens already included in completion_tokens."""
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=200,  # INCLUDES reasoning
        completion_tokens_details=SimpleNamespace(reasoning_tokens=50),
    )
    result = extract_usage("openai", usage)
    # completion_tokens is 200, NOT 200+50
    assert result.total_output_tokens == 200


# --- Cross-model contamination regression ---

def test_cross_model_cache_key_different():
    """REGRESSION: Different models must produce different L1 cache keys."""
    msgs = [{"role": "user", "content": "same question"}]
    key_gpt = build_l1_cache_key("gpt-4o", msgs)
    key_claude = build_l1_cache_key("claude-sonnet-4-6", msgs)
    assert key_gpt != key_claude


def test_cross_tenant_cache_key_different():
    """REGRESSION: Different tenants must produce different L1 cache keys."""
    msgs = [{"role": "user", "content": "same question"}]
    key_a = build_l1_cache_key("gpt-4o", msgs, tenant_id="tenant_a")
    key_b = build_l1_cache_key("gpt-4o", msgs, tenant_id="tenant_b")
    assert key_a != key_b


# --- Budget bypass regression ---

def test_budget_engine_terminates_at_100pct():
    """REGRESSION: Budget at 100% must raise, not silently pass."""
    engine = BudgetEngine("regression_budget", 1.0, "gpt-4o")
    engine.record_cost(0.99)
    with pytest.raises(BudgetExhaustedGracefully):
        engine.check_and_act(0.10, [{"role": "user", "content": "hi"}])


def test_budget_safety_margin():
    """REGRESSION: Budget check uses 1.5x safety margin."""
    engine = BudgetEngine("margin_test", 1.0, "gpt-4o")
    engine.spent = 0.55
    # 0.55 + (0.10 * 1.5) = 0.70 — should trigger 60% alert but not downgrade
    msgs, model = engine.check_and_act(0.10, [{"role": "user", "content": "hi"}])
    # At 70%, should have triggered alert but not downgrade (80% threshold)


# --- Response validation regression ---

def test_truncated_never_cached():
    """REGRESSION: finish_reason='length' must never be cached."""
    assert validate_for_cache("Valid text", finish_reason="length") is False


def test_content_filter_never_cached():
    """REGRESSION: finish_reason='content_filter' must never be cached."""
    assert validate_for_cache("Valid text", finish_reason="content_filter") is False


def test_refusal_never_cached():
    """REGRESSION: Anthropic refusal must never be cached."""
    assert validate_for_cache("Valid text", finish_reason="refusal") is False


# --- Error classifier regression ---

def test_insufficient_quota_not_retryable():
    """REGRESSION: OpenAI insufficient_quota 429 is NOT retryable."""
    class MockError(Exception):
        status_code = 429
        __module__ = "test"
    MockError.__name__ = "RateLimitError"
    exc = MockError("insufficient_quota error")
    result = classify_error(exc, "openai")
    assert result.retryable is False


def test_anthropic_overloaded_always_retryable():
    """REGRESSION: Anthropic 529 OverloadedError is ALWAYS retryable."""
    class MockError(Exception):
        status_code = 529
        __module__ = "test"
    MockError.__name__ = "OverloadedError"
    result = classify_error(MockError("overloaded"), "anthropic")
    assert result.retryable is True


# --- Negation detection regression ---

def test_negation_prefix_in_semantic_content():
    """REGRESSION: Negated queries must get NOT: prefix."""
    msgs = [{"role": "user", "content": "I don't want to cancel"}]
    content = extract_semantic_content(msgs)
    assert content.startswith("NOT:")


def test_non_negated_no_prefix():
    """REGRESSION: Non-negated queries must not get NOT: prefix."""
    msgs = [{"role": "user", "content": "I want to subscribe"}]
    content = extract_semantic_content(msgs)
    assert not content.startswith("NOT:")


# --- Cache side effects regression ---

def test_side_effect_tool_skip_cache():
    """REGRESSION: Side-effect tools must skip cache entirely."""
    cache = TwoTierCacheMiddleware()
    msgs = [{"role": "user", "content": "send email to boss"}]
    tools = [{"function": {"name": "send_email", "parameters": {}}}]
    cache.store(model="gpt-4o", messages=msgs, response="Email sent", tools=tools)
    result = cache.lookup(model="gpt-4o", messages=msgs, tools=tools)
    assert isinstance(result, CacheMiss)


def test_high_temp_skip_cache():
    """REGRESSION: temperature > 0.5 must skip cache."""
    cache = TwoTierCacheMiddleware()
    msgs = [{"role": "user", "content": "creative writing prompt"}]
    cache.store(model="gpt-4o", messages=msgs, response="Creative text", temperature=0.8)
    result = cache.lookup(model="gpt-4o", messages=msgs, temperature=0.8)
    assert isinstance(result, CacheMiss)
