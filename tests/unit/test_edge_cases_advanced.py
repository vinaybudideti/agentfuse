"""
Advanced edge case tests across multiple modules.
Tests rare but important production scenarios.
"""

import pytest
from types import SimpleNamespace


def test_budget_engine_repr():
    """BudgetEngine must have useful __repr__."""
    from agentfuse.core.budget import BudgetEngine
    engine = BudgetEngine("test_repr", 10.0, "gpt-4o")
    r = repr(engine)
    assert "test_repr" in r
    assert "10.0" in r


def test_normalized_usage_repr():
    """NormalizedUsage must have useful __repr__."""
    from agentfuse.providers.response import NormalizedUsage
    usage = NormalizedUsage(total_input_tokens=100, total_output_tokens=50, provider="openai")
    r = repr(usage)
    assert "100" in r
    assert "openai" in r


def test_cache_hit_repr():
    """CacheHit must have useful __repr__."""
    from agentfuse.core.cache import CacheHit
    hit = CacheHit(tier=1, response="hello world", similarity=0.95)
    r = repr(hit)
    assert "tier=1" in r
    assert "0.950" in r


def test_cache_miss_repr():
    """CacheMiss must have useful __repr__."""
    from agentfuse.core.cache import CacheMiss
    miss = CacheMiss(reason="L2 error")
    r = repr(miss)
    assert "L2 error" in r


def test_classify_error_unknown_provider():
    """Unknown provider must default to retryable."""
    from agentfuse.core.error_classifier import classify_error
    result = classify_error(RuntimeError("test"), "some_unknown_provider")
    assert result.retryable is True


def test_model_registry_list_models():
    """list_models must return a sorted list."""
    from agentfuse.providers.registry import ModelRegistry
    reg = ModelRegistry(refresh_hours=0)
    models = reg.list_models()
    assert len(models) > 20
    assert models == sorted(models)


def test_model_registry_get_provider():
    """get_provider must return the correct provider."""
    from agentfuse.providers.registry import ModelRegistry
    reg = ModelRegistry(refresh_hours=0)
    assert reg.get_provider("gpt-4o") == "openai"
    assert reg.get_provider("claude-sonnet-4-6") == "anthropic"
    assert reg.get_provider("gemini-2.5-pro") == "gemini"


def test_token_counter_empty_messages():
    """count_messages with empty list must return 3 (priming tokens)."""
    from agentfuse.providers.tokenizer import TokenCounterAdapter
    tc = TokenCounterAdapter()
    assert tc.count_messages([], "gpt-4o") == 3


def test_request_optimizer_empty_messages():
    """Optimizing empty messages must not crash."""
    from agentfuse.core.request_optimizer import RequestOptimizer
    from agentfuse.providers.pricing import ModelPricingEngine
    from agentfuse.providers.tokenizer import TokenCounterAdapter
    opt = RequestOptimizer(ModelPricingEngine(), TokenCounterAdapter())
    msgs, report = opt.optimize([], "gpt-4o")
    assert msgs == []


def test_estimate_cost_zero_tokens():
    """Estimating cost with very short message must return tiny amount."""
    from agentfuse.gateway import estimate_cost
    est = estimate_cost("gpt-4o", [{"role": "user", "content": "hi"}], max_output_tokens=1)
    assert est["estimated_total_cost_usd"] > 0
    assert est["estimated_total_cost_usd"] < 0.01


def test_spend_report_has_correct_keys():
    """get_spend_report must always return all expected keys."""
    from agentfuse.gateway import get_spend_report
    report = get_spend_report()
    assert "total_usd" in report
    assert "by_model" in report
    assert "by_provider" in report
    assert "by_run" in report


def test_all_exports_accessible():
    """All __all__ exports must be importable."""
    import agentfuse
    for name in agentfuse.__all__:
        obj = getattr(agentfuse, name)
        assert obj is not None, f"{name} is None"


def test_version_is_string():
    """Version must be a valid string."""
    import agentfuse
    assert isinstance(agentfuse.__version__, str)
    parts = agentfuse.__version__.split(".")
    assert len(parts) == 3


def test_security_module_accessible():
    """Security functions must be importable from top level."""
    from agentfuse import mask_api_key, check_prompt_injection, validate_response_safety
    assert callable(mask_api_key)
    assert callable(check_prompt_injection)
    assert callable(validate_response_safety)


def test_session_module_accessible():
    """AgentSession must be importable from top level."""
    from agentfuse import AgentSession
    session = AgentSession("test", budget_usd=5.0)
    assert session.name == "test"


def test_kill_switch_accessible():
    """kill_switch singleton must be importable from top level."""
    from agentfuse import kill_switch
    assert kill_switch is not None
    assert hasattr(kill_switch, "kill")
    assert hasattr(kill_switch, "revive")
