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


def test_resolve_provider_all_known():
    """All major providers must be resolved correctly."""
    from agentfuse.providers.router import resolve_provider
    assert resolve_provider("gpt-5")[0] == "openai"
    assert resolve_provider("claude-sonnet-4-6")[0] == "anthropic"
    assert resolve_provider("gemini-2.5-pro")[0] == "gemini"
    assert resolve_provider("deepseek/deepseek-chat")[0] == "deepseek"
    assert resolve_provider("mistral-large-latest")[0] == "mistral"
    assert resolve_provider("grok-4.1-fast")[0] == "xai"


def test_pricing_gpt5_correct():
    """GPT-5 pricing must match registry."""
    from agentfuse.providers.pricing import ModelPricingEngine
    engine = ModelPricingEngine()
    cost = engine.input_cost("gpt-5", 1_000_000)
    assert abs(cost - 1.25) < 0.01  # $1.25 per 1M input tokens


def test_tokenizer_cjk_detection():
    """CJK text must use different char-per-token ratio."""
    from agentfuse.providers.tokenizer import TokenCounterAdapter
    tc = TokenCounterAdapter()
    # CJK chars are ~1.5 chars per token vs ~3.5 for English
    english = tc._count_fallback("hello world this is a test", "unknown")
    cjk = tc._count_fallback("你好世界这是一个测试", "unknown")
    # CJK should produce MORE tokens per character
    assert cjk / len("你好世界这是一个测试") > english / len("hello world this is a test")


def test_budget_exhausted_has_context():
    """BudgetExhaustedGracefully must include run context."""
    from agentfuse.core.budget import BudgetExhaustedGracefully
    exc = BudgetExhaustedGracefully(
        partial_results=["partial"],
        receipt=None,
        run_id="test_run",
        spent=4.50,
        budget=5.00,
    )
    assert exc.run_id == "test_run"
    assert exc.spent == 4.50
    assert "test_run" in str(exc)


def test_request_dedup_key_deterministic():
    """Same input must produce same dedup key."""
    from agentfuse.core.dedup import RequestDeduplicator
    k1 = RequestDeduplicator.make_key("gpt-4o", [{"role": "user", "content": "test"}])
    k2 = RequestDeduplicator.make_key("gpt-4o", [{"role": "user", "content": "test"}])
    assert k1 == k2


def test_request_dedup_key_different():
    """Different inputs must produce different dedup keys."""
    from agentfuse.core.dedup import RequestDeduplicator
    k1 = RequestDeduplicator.make_key("gpt-4o", [{"role": "user", "content": "a"}])
    k2 = RequestDeduplicator.make_key("gpt-4o", [{"role": "user", "content": "b"}])
    assert k1 != k2


def test_fallback_chain_known_models():
    """All models in fallback chains must exist."""
    from agentfuse.core.fallback_chain import DEFAULT_CHAINS
    from agentfuse.providers.registry import BUILTIN_MODELS
    for model, fallbacks in DEFAULT_CHAINS.items():
        for fb in fallbacks:
            assert fb in BUILTIN_MODELS or "/" in fb, f"Fallback {fb} not in registry"


def test_cost_tracker_thread_safety():
    """CostTracker must be thread-safe."""
    import threading
    from agentfuse.core.cost_tracker import CostTracker
    tracker = CostTracker()
    errors = []

    def worker():
        try:
            for _ in range(50):
                tracker.record_call("gpt-4o", "openai", "run_1", input_tokens=10, output_tokens=5, cost_usd=0.001)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(errors) == 0


def test_adaptive_threshold_bounds():
    """Adaptive threshold must stay within configured bounds."""
    from agentfuse.core.adaptive_threshold import AdaptiveSimilarityThreshold
    at = AdaptiveSimilarityThreshold(initial=0.90, min_threshold=0.80, max_threshold=0.98)
    # Many bad hits should tighten (raise) threshold but not above max
    for _ in range(100):
        at.report_bad_hit()
    assert at.get() <= 0.98


def test_anomaly_detector_warmup():
    """Anomaly detector must not fire during warmup period."""
    from agentfuse.core.anomaly import CostAnomalyDetector
    detector = CostAnomalyDetector(min_samples=10)
    for _ in range(5):
        result = detector.record("gpt-4o", 0.01)
        assert result is None  # still warming up


def test_batch_eligibility_savings_calculation():
    """Batch detector must calculate correct savings."""
    from agentfuse.core.batch_detector import BatchEligibilityDetector
    detector = BatchEligibilityDetector(min_batch_size=2)
    msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "hi"}]
    detector.observe(msgs, "gpt-4o", estimated_cost=0.10)
    result = detector.observe(msgs, "gpt-4o", estimated_cost=0.10)
    assert result is not None
    assert result.estimated_savings_usd > 0


def test_prompt_cache_non_claude_unchanged():
    """Non-Claude models must get messages unchanged from prompt cache."""
    from agentfuse.core.prompt_cache import PromptCachingMiddleware
    pc = PromptCachingMiddleware()
    msgs = [{"role": "system", "content": "x" * 5000}]
    result = pc.inject(msgs, "gpt-4o")
    assert result == msgs  # unchanged


def test_load_balancer_empty():
    """Load balancer with no endpoints must return None."""
    from agentfuse.core.load_balancer import ModelLoadBalancer
    lb = ModelLoadBalancer()
    assert lb.get_endpoint("gpt-4o") is None


def test_acompletion_importable():
    """acompletion must be importable from top level."""
    from agentfuse import acompletion
    assert callable(acompletion)


def test_estimate_cost_importable():
    """estimate_cost must be importable and callable."""
    from agentfuse import estimate_cost
    result = estimate_cost("gpt-4o", [{"role": "user", "content": "hi"}])
    assert result["model"] == "gpt-4o"


def test_add_api_key_importable():
    """add_api_key must be importable and callable."""
    from agentfuse import add_api_key
    assert callable(add_api_key)
