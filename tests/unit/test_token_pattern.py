"""
Tests for TokenPatternAdapter — auto-discovers LLM usage field patterns.
"""

from types import SimpleNamespace
from agentfuse.providers.token_pattern import (
    discover_usage_pattern, extract_with_pattern, reset_patterns,
)


def setup_function():
    reset_patterns()


def test_discover_openai_pattern():
    """OpenAI usage fields must be discovered correctly."""
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        prompt_tokens_details=SimpleNamespace(cached_tokens=20),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
    )
    pattern = discover_usage_pattern(usage, "openai")
    assert pattern["input"] == "prompt_tokens"
    assert pattern["output"] == "completion_tokens"
    assert "cached_input_nested" in pattern
    assert "reasoning_nested" in pattern


def test_discover_anthropic_pattern():
    """Anthropic usage fields must be discovered correctly."""
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=20,
    )
    pattern = discover_usage_pattern(usage, "anthropic")
    assert pattern["input"] == "input_tokens"
    assert pattern["output"] == "output_tokens"
    assert pattern["cached_input"] == "cache_read_input_tokens"
    assert pattern["cache_write"] == "cache_creation_input_tokens"


def test_discover_gemini_pattern():
    """Gemini camelCase fields must be discovered."""
    usage = SimpleNamespace(
        promptTokenCount=100,
        candidatesTokenCount=50,
        thoughtsTokenCount=20,
    )
    pattern = discover_usage_pattern(usage, "gemini")
    assert pattern["input"] == "promptTokenCount"
    assert pattern["output"] == "candidatesTokenCount"
    assert pattern["reasoning"] == "thoughtsTokenCount"


def test_discover_deepseek_pattern():
    """DeepSeek cache fields must be discovered."""
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        prompt_cache_hit_tokens=30,
        prompt_cache_miss_tokens=20,
    )
    pattern = discover_usage_pattern(usage, "deepseek")
    assert pattern["input"] == "prompt_tokens"
    assert pattern["cached_input"] == "prompt_cache_hit_tokens"
    assert pattern["cache_write"] == "prompt_cache_miss_tokens"


def test_extract_openai():
    """OpenAI extraction must produce correct NormalizedUsage."""
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
    )
    result = extract_with_pattern(usage, "openai")
    assert result.total_input_tokens == 100
    assert result.total_output_tokens == 50
    assert result.provider == "openai"


def test_extract_anthropic_adds_cache():
    """Anthropic extraction must add cache tokens to input total."""
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=20,
    )
    result = extract_with_pattern(usage, "anthropic")
    assert result.total_input_tokens == 150  # 100 + 30 + 20
    assert result.cached_input_tokens == 30
    assert result.cache_write_tokens == 20


def test_extract_gemini_adds_thoughts():
    """Gemini extraction must add thoughts to output."""
    usage = SimpleNamespace(
        promptTokenCount=100,
        candidatesTokenCount=50,
        thoughtsTokenCount=20,
    )
    result = extract_with_pattern(usage, "gemini")
    assert result.total_output_tokens == 70  # 50 + 20 thoughts


def test_extract_unknown_provider():
    """Unknown provider must extract what it can."""
    usage = SimpleNamespace(
        prompt_tokens=200,
        completion_tokens=100,
    )
    result = extract_with_pattern(usage, "custom-llm")
    assert result.total_input_tokens == 200
    assert result.total_output_tokens == 100


def test_extract_none_usage():
    """None usage must return empty NormalizedUsage."""
    result = extract_with_pattern(None, "openai")
    assert result.total_input_tokens == 0
    assert result.total_output_tokens == 0


def test_pattern_cached_per_provider():
    """Pattern discovery must be cached per provider."""
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    extract_with_pattern(usage, "test_provider")
    extract_with_pattern(usage, "test_provider")

    from agentfuse.providers.token_pattern import _discovered_patterns
    assert "test_provider" in _discovered_patterns


def test_empty_usage_object():
    """Usage object with no known fields must return zeros."""
    usage = SimpleNamespace(custom_field=42)
    result = extract_with_pattern(usage, "weird_provider")
    assert result.total_input_tokens == 0
    assert result.total_output_tokens == 0
