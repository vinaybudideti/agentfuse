"""
Production runtime correctness tests.

These tests verify the exact scenarios that would fail in production:
- Anthropic cost calculation with cache tokens
- Budget not allowing overspend
- Empty response caching
- Streaming cost accuracy
"""

from types import SimpleNamespace
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.providers.response import NormalizedUsage, extract_usage
from agentfuse.core.budget import BudgetEngine, BudgetExhaustedGracefully
import pytest


# --- Anthropic billing accuracy ---

def test_anthropic_cached_response_cost_accuracy():
    """Anthropic cache_read tokens must be billed at cached_input rate (0.1x),
    not the full input rate. This was the #1 production bug."""
    p = ModelPricingEngine()

    usage = NormalizedUsage(
        total_input_tokens=800,   # 100 uncached + 500 cache_read + 200 cache_write
        total_output_tokens=50,
        cached_input_tokens=500,
        cache_write_tokens=200,
        provider="anthropic",
    )

    cost = p.total_cost_normalized("claude-sonnet-4-6", usage)

    # Correct billing:
    # uncached: 100 tokens * $3.00/M = $0.0003
    # cache_read: 500 tokens * $0.30/M = $0.00015
    # cache_write: 200 tokens * $3.75/M (1.25x) = $0.00075
    # output: 50 tokens * $15.00/M = $0.00075
    # Total = $0.00195
    expected = 0.0003 + 0.00015 + 0.00075 + 0.00075
    assert abs(cost - expected) < 0.00001, f"Expected ${expected:.6f}, got ${cost:.6f}"

    # Compare with WRONG old method (uniform rate on all input)
    wrong_cost = p.total_cost("claude-sonnet-4-6", 800, 50)
    # Wrong: 800 * $3.00/M + 50 * $15.00/M = $0.0024 + $0.00075 = $0.00315
    # The old method over-charges by ~61%
    assert wrong_cost > cost, "Old method should be more expensive than correct method"


def test_anthropic_no_cache_falls_back_to_total_cost():
    """When there are no cache tokens, total_cost_normalized should match total_cost."""
    p = ModelPricingEngine()
    usage = NormalizedUsage(
        total_input_tokens=1000,
        total_output_tokens=500,
        provider="anthropic",
    )
    cost_norm = p.total_cost_normalized("claude-sonnet-4-6", usage)
    cost_old = p.total_cost("claude-sonnet-4-6", 1000, 500)
    assert abs(cost_norm - cost_old) < 0.00001


def test_openai_total_cost_normalized_matches_total_cost():
    """For OpenAI (no cache tokens), both methods should agree."""
    p = ModelPricingEngine()
    usage = NormalizedUsage(
        total_input_tokens=1000,
        total_output_tokens=500,
        provider="openai",
    )
    cost_norm = p.total_cost_normalized("gpt-4o", usage)
    cost_old = p.total_cost("gpt-4o", 1000, 500)
    assert abs(cost_norm - cost_old) < 0.00001


def test_extract_usage_then_price_end_to_end():
    """End-to-end: extract Anthropic usage then price it correctly."""
    usage_obj = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=500,
        cache_creation_input_tokens=200,
    )
    normalized = extract_usage("anthropic", usage_obj)
    assert normalized.total_input_tokens == 800
    assert normalized.cached_input_tokens == 500
    assert normalized.cache_write_tokens == 200

    p = ModelPricingEngine()
    cost = p.total_cost_normalized("claude-sonnet-4-6", normalized)
    assert cost > 0
    assert cost < p.total_cost("claude-sonnet-4-6", 800, 50)  # cheaper than uniform


# --- Cache correctness ---

def test_cache_rejects_empty_response():
    """Empty responses must never be cached."""
    from agentfuse.core.cache import TwoTierCacheMiddleware, CacheMiss
    cache = TwoTierCacheMiddleware()

    # Store empty response — should be silently rejected
    cache.store("gpt-4o", [{"role": "user", "content": "test"}], "")
    cache.store("gpt-4o", [{"role": "user", "content": "test"}], "   ")

    # Lookup should miss
    result = cache.lookup("gpt-4o", [{"role": "user", "content": "test"}])
    assert isinstance(result, CacheMiss)


def test_cache_rejects_empty_via_compat_api():
    """Empty responses rejected via backward-compat store API too."""
    from agentfuse.core.cache import TwoTierCacheMiddleware, CacheMiss
    cache = TwoTierCacheMiddleware()

    cache.store("some_key", "", "gpt-4o")  # old API, empty response
    result = cache.check("some_key", "gpt-4o")
    assert isinstance(result, CacheMiss)


# --- Pricing edge cases ---

def test_negative_tokens_return_zero_cost():
    """Negative token counts must return 0 cost, not negative cost."""
    p = ModelPricingEngine()
    assert p.input_cost("gpt-4o", -100) == 0.0
    assert p.output_cost("gpt-4o", -50) == 0.0
    assert p.cached_input_cost("gpt-4o", -100) == 0.0


def test_cache_write_cost():
    """cache_write_cost must be 1.25x the input rate."""
    p = ModelPricingEngine()
    input_rate = 3.00  # claude-sonnet-4-6
    write_cost = p.cache_write_cost("claude-sonnet-4-6", 1_000_000)
    assert abs(write_cost - (input_rate * 1.25)) < 0.01


def test_total_cost_normalized_with_none():
    """None usage must return 0 cost."""
    p = ModelPricingEngine()
    assert p.total_cost_normalized("gpt-4o", None) == 0.0


# --- Budget alert safety ---

def test_alert_callback_exception_doesnt_crash_budget():
    """A failing alert callback must not crash budget enforcement."""
    def bad_callback(pct, event):
        raise RuntimeError("callback crashed!")

    engine = BudgetEngine("safe_alert", 1.00, "gpt-4o", alert_cb=bad_callback)
    engine.spent = 0.55
    # This should NOT raise — alert callback failure is swallowed
    msgs, model = engine.check_and_act(0.06, [{"role": "user", "content": "hi"}])
    assert model == "gpt-4o"


# --- Prompt cache correctness ---

def test_prompt_cache_handles_list_content():
    """Prompt cache must handle content that is already a list of blocks."""
    from agentfuse.core.prompt_cache import PromptCachingMiddleware
    m = PromptCachingMiddleware()

    # Need enough text to exceed 1024 tokens with 1.20x Anthropic margin
    long_text = "System instructions for the agent to follow carefully. " * 500
    msgs = [
        {"role": "system", "content": [
            {"type": "text", "text": long_text},
        ]},
    ]
    result = m.inject(msgs, "claude-sonnet-4-6")
    # Should add cache_control to the last text block, not overwrite the list
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][0].get("cache_control") == {"type": "ephemeral"}


def test_prompt_cache_detects_uuid_as_dynamic():
    """UUIDs in content must be detected as dynamic (not cached)."""
    from agentfuse.core.prompt_cache import PromptCachingMiddleware
    m = PromptCachingMiddleware()

    text_with_uuid = "Process request 550e8400-e29b-41d4-a716-446655440000 now. " * 200
    assert not m._is_static(text_with_uuid)
