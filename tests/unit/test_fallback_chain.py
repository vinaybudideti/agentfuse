"""
Tests for FallbackModelChain.
"""

import asyncio
import pytest
from agentfuse.core.fallback_chain import FallbackModelChain


class MockRateLimitError(Exception):
    pass
MockRateLimitError.__name__ = "RateLimitError"


class MockAuthError(Exception):
    pass
MockAuthError.__name__ = "AuthenticationError"


def test_primary_model_success():
    """When primary succeeds, no fallback needed."""
    chain = FallbackModelChain("gpt-4o")
    result = chain.call(lambda msgs, model: f"response from {model}", [])
    assert result == "response from gpt-4o"
    assert chain.model_used == "gpt-4o"
    assert chain.fallback_count == 0


def test_fallback_on_rate_limit():
    """Rate limit on primary must trigger fallback."""
    call_count = 0

    def flaky(msgs, model):
        nonlocal call_count
        call_count += 1
        if model == "gpt-4o":
            raise MockRateLimitError("Rate limited")
        return f"response from {model}"

    chain = FallbackModelChain("gpt-4o", provider="openai")
    result = chain.call(flaky, [])
    assert "gpt-4o-mini" in result
    assert chain.model_used == "gpt-4o-mini"
    assert chain.fallback_count == 1


def test_non_retryable_error_no_fallback():
    """Auth errors must NOT trigger fallback — re-raised immediately."""
    def auth_fail(msgs, model):
        raise MockAuthError("Invalid key")

    chain = FallbackModelChain("gpt-4o", provider="openai")
    with pytest.raises(MockAuthError):
        chain.call(auth_fail, [])


def test_all_models_fail():
    """When all models fail, last error is raised."""
    def always_fail(msgs, model):
        raise MockRateLimitError(f"{model} rate limited")

    chain = FallbackModelChain("gpt-4o", fallback_models=["gpt-4o-mini"], provider="openai")
    with pytest.raises(MockRateLimitError):
        chain.call(always_fail, [])


def test_custom_fallback_chain():
    """Custom fallback chain must be used instead of defaults."""
    chain = FallbackModelChain("custom-model", fallback_models=["fallback-1", "fallback-2"])

    def fail_custom(msgs, model):
        if model == "custom-model":
            raise MockRateLimitError("fail")
        return f"ok from {model}"

    chain.provider = "openai"
    result = chain.call(fail_custom, [])
    assert result == "ok from fallback-1"


def test_async_fallback():
    """Async fallback must work."""
    async def async_flaky(msgs, model):
        if model == "claude-opus-4-6":
            raise MockRateLimitError("overloaded")
        return f"async response from {model}"

    chain = FallbackModelChain("claude-opus-4-6", provider="anthropic")
    result = asyncio.run(chain.call_async(async_flaky, []))
    assert "claude-sonnet-4-6" in result


def test_get_status():
    chain = FallbackModelChain("gpt-4o")
    status = chain.get_status()
    assert status["primary_model"] == "gpt-4o"
    assert status["fallback_count"] == 0
    assert len(status["available_fallbacks"]) > 0
