"""
Tests for async retry and streaming wrapper features.
"""

import asyncio
import pytest
from agentfuse.core.retry import CostAwareRetry, RetryBudgetExhausted
from agentfuse.core.budget import BudgetEngine


class MockAsyncRateLimitError(Exception):
    pass

MockAsyncRateLimitError.__name__ = "RateLimitError"


def test_async_retry_success():
    """wrap_async must work for successful calls."""
    budget = BudgetEngine("async_ok", 10.0, "gpt-4o")
    retry = CostAwareRetry(budget, max_attempts=3, provider="openai")

    async def success(messages, model):
        return "async success"

    result = asyncio.run(retry.wrap_async(success, [{"role": "user", "content": "hi"}], "gpt-4o"))
    assert result == "async success"


def test_async_retry_retries():
    """wrap_async must retry on retryable errors."""
    budget = BudgetEngine("async_retry", 10.0, "gpt-4o")
    retry = CostAwareRetry(budget, max_retry_cost_usd=10.0, max_attempts=3, provider="openai")
    call_count = 0

    async def flaky(messages, model):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise MockAsyncRateLimitError("Rate limited")
        return "recovered"

    result = asyncio.run(retry.wrap_async(flaky, [{"role": "user", "content": "hi"}], "gpt-4o"))
    assert result == "recovered"
    assert call_count == 3


def test_langchain_wrapper_records_cost():
    """AgentFuseChatModel must record cost after inner model call."""
    from agentfuse.integrations.langchain import AgentFuseChatModel
    from types import SimpleNamespace

    # Create a mock inner model
    class MockInner:
        def invoke(self, messages, **kwargs):
            return SimpleNamespace(content="Hello from mock")

    model = AgentFuseChatModel(inner=MockInner(), budget=5.00)
    model.invoke([{"role": "user", "content": "test"}])

    # Cost should be recorded (non-zero)
    assert model.engine.spent > 0, f"Expected cost > 0, got {model.engine.spent}"


def test_langchain_wrapper_cache_hit_no_cost():
    """Cache hit must not record any cost."""
    from agentfuse.integrations.langchain import AgentFuseChatModel
    from types import SimpleNamespace

    class MockInner:
        def invoke(self, messages, **kwargs):
            return SimpleNamespace(content="First response")

    model = AgentFuseChatModel(inner=MockInner(), budget=5.00)

    # First call — records cost and caches
    model.invoke([{"role": "user", "content": "cached query"}])
    first_cost = model.engine.spent

    # Second call — should hit cache, no additional cost
    result = model.invoke([{"role": "user", "content": "cached query"}])
    assert model.engine.spent == first_cost, "Cache hit should not add cost"
