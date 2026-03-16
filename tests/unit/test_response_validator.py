"""
Tests for ResponseValidator — catches bad responses before caching.
"""

from agentfuse.core.response_validator import validate_response, validate_for_cache


def test_empty_response_invalid():
    result = validate_response("")
    assert not result.valid
    assert not result.should_cache
    assert result.should_retry


def test_whitespace_response_invalid():
    result = validate_response("   \n\t  ")
    assert not result.valid
    assert not result.should_cache


def test_normal_response_valid():
    result = validate_response("The capital of France is Paris.")
    assert result.valid
    assert result.should_cache


def test_truncated_response_not_cached():
    result = validate_response("This is a partial resp", finish_reason="length")
    assert result.valid  # still deliverable
    assert not result.should_cache  # but don't cache incomplete


def test_llm_refusal_not_cached():
    result = validate_response(
        "I'm sorry, I cannot help with that request. As an AI language model, "
        "I am not able to provide that information."
    )
    assert result.valid  # still deliverable
    assert not result.should_cache  # but don't cache refusals


def test_single_error_pattern_still_cached():
    """Single error-like phrase in otherwise good response should cache."""
    result = validate_response(
        "Here's how to handle the error: first check the logs, then "
        "verify the configuration is correct. The error usually means "
        "the database connection timed out."
    )
    assert result.should_cache  # only 0-1 error patterns, below threshold


def test_validate_for_cache_simple():
    assert validate_for_cache("Good response") is True
    assert validate_for_cache("") is False
    assert validate_for_cache("truncated", finish_reason="length") is False


def test_tool_calls_finish_reason_valid():
    """tool_calls finish reason should be cacheable (L1 exact match only)."""
    result = validate_response("Using search tool", finish_reason="tool_calls")
    assert result.valid
    assert result.should_cache


def test_very_short_response_not_cached():
    """Single character responses shouldn't be cached."""
    result = validate_response(".", min_length=5)
    assert result.valid
    assert not result.should_cache
