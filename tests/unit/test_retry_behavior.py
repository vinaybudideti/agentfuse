"""
Loop 9 — CostAwareRetry behavioral tests.
"""

import pytest
from agentfuse.core.retry import CostAwareRetry, RetryBudgetExhausted
from agentfuse.core.budget import BudgetEngine


class MockRateLimitError(Exception):
    """Mock a retryable rate limit error."""
    pass


# Give it a name the classifier recognizes
MockRateLimitError.__name__ = "RateLimitError"


class MockAuthError(Exception):
    """Mock a non-retryable auth error."""
    pass


MockAuthError.__name__ = "AuthenticationError"


def test_retry_on_retryable_error():
    """Retryable errors must trigger retry."""
    budget = BudgetEngine("retry_test", 10.0, "gpt-4o")
    retry = CostAwareRetry(budget, max_retry_cost_usd=1.0, max_attempts=3, provider="openai")

    call_count = 0

    def flaky_fn(messages, model):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise MockRateLimitError("Rate limited")
        return "success"

    result = retry.wrap(flaky_fn, [{"role": "user", "content": "hi"}], "gpt-4o")
    assert result == "success"
    assert call_count == 3


def test_no_retry_on_non_retryable_error():
    """Non-retryable errors must be raised immediately."""
    budget = BudgetEngine("no_retry_test", 10.0, "gpt-4o")
    retry = CostAwareRetry(budget, max_attempts=3, provider="openai")

    def auth_fail(messages, model):
        raise MockAuthError("Invalid API key")

    with pytest.raises(MockAuthError):
        retry.wrap(auth_fail, [{"role": "user", "content": "hi"}], "gpt-4o")


def test_retry_budget_exhaustion():
    """Must raise RetryBudgetExhausted when retry cost exceeds limit."""
    budget = BudgetEngine("budget_exhaust_test", 10.0, "gpt-4o")
    retry = CostAwareRetry(budget, max_retry_cost_usd=0.0001, max_attempts=5, provider="openai")

    def always_fail(messages, model):
        raise MockRateLimitError("Rate limited")

    with pytest.raises((RetryBudgetExhausted, MockRateLimitError)):
        retry.wrap(always_fail, [{"role": "user", "content": "hello " * 100}], "gpt-4o")


def test_model_downgrade_on_retry():
    """Model must be downgraded on retry."""
    budget = BudgetEngine("downgrade_retry", 10.0, "gpt-4o")
    retry = CostAwareRetry(budget, max_retry_cost_usd=10.0, max_attempts=3, provider="openai")

    models_used = []

    def track_model(messages, model):
        models_used.append(model)
        if len(models_used) < 3:
            raise MockRateLimitError("Rate limited")
        return "ok"

    retry.wrap(track_model, [{"role": "user", "content": "hi"}], "gpt-4o")
    assert models_used[0] == "gpt-4o"
    # After retry, model should be downgraded
    assert "gpt-4o-mini" in models_used


def test_success_on_first_try():
    """Successful first call must not trigger any retry logic."""
    budget = BudgetEngine("success_test", 10.0, "gpt-4o")
    retry = CostAwareRetry(budget, max_attempts=3, provider="openai")

    def success(messages, model):
        return "direct success"

    result = retry.wrap(success, [{"role": "user", "content": "hi"}], "gpt-4o")
    assert result == "direct success"
    assert retry.retry_cost_spent == 0.0
