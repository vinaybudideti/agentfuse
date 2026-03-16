"""
Phase 3 — Usage normalization tests.

Verifies that token usage is correctly extracted across providers,
especially Anthropic's unique input_tokens behavior.
"""

from types import SimpleNamespace
from agentfuse.providers.response import extract_usage, NormalizedUsage


def test_anthropic_sums_all_input_fields():
    """Anthropic: input=100, cache_read=50, cache_write=20 → total=170."""
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=200,
        cache_read_input_tokens=50,
        cache_creation_input_tokens=20,
    )
    result = extract_usage("anthropic", usage)
    assert result.total_input_tokens == 170  # 100 + 50 + 20
    assert result.total_output_tokens == 200
    assert result.cached_input_tokens == 50
    assert result.cache_write_tokens == 20


def test_openai_reasoning_not_double_counted():
    """OpenAI: completion_tokens already includes reasoning_tokens."""
    usage = SimpleNamespace(
        prompt_tokens=500,
        completion_tokens=300,
        prompt_tokens_details=SimpleNamespace(cached_tokens=100),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=150),
    )
    result = extract_usage("openai", usage)
    # completion_tokens is 300 (already includes 150 reasoning)
    assert result.total_output_tokens == 300
    assert result.reasoning_tokens == 150
    assert result.total_input_tokens == 500
    assert result.cached_input_tokens == 100


def test_gemini_thoughts_counted_as_output():
    """Gemini: thoughts_token_count is billed as output, add to candidates."""
    usage = SimpleNamespace(
        prompt_token_count=400,
        candidates_token_count=200,
        thoughts_token_count=100,
        cached_content_token_count=50,
    )
    result = extract_usage("gemini", usage)
    assert result.total_output_tokens == 300  # 200 + 100 thoughts
    assert result.total_input_tokens == 400
    assert result.reasoning_tokens == 100
    assert result.cached_input_tokens == 50


def test_unknown_provider_fallback():
    """Unknown provider must extract what it can, not crash."""
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
    )
    result = extract_usage("custom-provider", usage)
    assert result.total_input_tokens == 100
    assert result.total_output_tokens == 50
    assert result.provider == "custom-provider"


def test_none_usage_returns_empty():
    """None usage object must return zero-valued NormalizedUsage."""
    result = extract_usage("openai", None)
    assert result.total_input_tokens == 0
    assert result.total_output_tokens == 0
    assert result.total_tokens == 0


def test_total_tokens_property():
    """total_tokens must be sum of input + output."""
    result = NormalizedUsage(total_input_tokens=100, total_output_tokens=200)
    assert result.total_tokens == 300
