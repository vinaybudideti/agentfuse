import pytest
from unittest.mock import MagicMock
from agentfuse.core.retry import CostAwareRetry, RetryBudgetExhausted
from agentfuse.core.budget import BudgetEngine


def test_retry_succeeds_on_second_attempt():
    engine = BudgetEngine("run_1", 10.00, "gpt-4o")
    retry = CostAwareRetry(engine, max_retry_cost_usd=1.00, max_attempts=3)
    call_count = [0]

    def flaky_fn(messages, model):
        call_count[0] += 1
        if call_count[0] < 2:
            raise Exception("ratelimit error")
        return "success"

    result = retry.wrap(flaky_fn, [{"role": "user", "content": "hi"}], "gpt-4o")
    assert result == "success"
    assert call_count[0] == 2


def test_retry_downgrades_model():
    engine = BudgetEngine("run_1", 10.00, "gpt-4o")
    retry = CostAwareRetry(engine, max_retry_cost_usd=1.00, max_attempts=3)
    models_used = []

    def track_model(messages, model):
        models_used.append(model)
        if len(models_used) < 2:
            raise Exception("ratelimit")
        return "ok"

    retry.wrap(track_model, [{"role": "user", "content": "hi"}], "gpt-4o")
    assert models_used[0] == "gpt-4o"
    assert models_used[1] == "gpt-4o-mini"  # downgraded on retry


def test_retry_budget_exhausted():
    engine = BudgetEngine("run_1", 10.00, "gpt-4o")
    retry = CostAwareRetry(engine, max_retry_cost_usd=0.00, max_attempts=3)

    def always_fails(messages, model):
        raise Exception("ratelimit")

    with pytest.raises((RetryBudgetExhausted, Exception)):
        retry.wrap(always_fails, [{"role": "user", "content": "hi"}], "gpt-4o")
