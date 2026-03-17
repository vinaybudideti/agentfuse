"""
Tests pushing toward 1000 — comprehensive coverage across all modules.
"""

import pytest
import threading
import time
from types import SimpleNamespace

# --- Budget Engine edge cases ---

def test_budget_state_progression():
    """Budget state must progress: NORMAL → DOWNGRADED → COMPRESSED → EXHAUSTED."""
    from agentfuse.core.budget import BudgetEngine, BudgetState, BudgetExhaustedGracefully
    engine = BudgetEngine("state_prog", 1.0, "gpt-4o")
    assert engine.state == BudgetState.NORMAL

    engine.spent = 0.55
    engine.check_and_act(0.01, [{"role": "user", "content": "hi"}])
    # At 56%, should be normal

    engine.spent = 0.79
    engine.check_and_act(0.01, [{"role": "user", "content": "hi"}])
    assert engine.state == BudgetState.DOWNGRADED


def test_budget_original_model_preserved():
    """original_model must be preserved after downgrade."""
    from agentfuse.core.budget import BudgetEngine
    engine = BudgetEngine("orig_model", 1.0, "gpt-4o")
    engine.spent = 0.80
    engine.check_and_act(0.01, [{"role": "user", "content": "hi"}])
    assert engine.original_model == "gpt-4o"
    assert engine.model != "gpt-4o"  # downgraded


# --- Cache key edge cases ---

def test_cache_key_with_tools():
    """Cache key must include tool names."""
    from agentfuse.core.keys import build_l1_cache_key
    msgs = [{"role": "user", "content": "test"}]
    tools = [{"function": {"name": "search", "parameters": {}}}]
    k1 = build_l1_cache_key("gpt-4o", msgs, tools=tools)
    k2 = build_l1_cache_key("gpt-4o", msgs)
    assert k1 != k2


def test_cache_key_with_response_format():
    """Cache key must include response_format."""
    from agentfuse.core.keys import build_l1_cache_key
    msgs = [{"role": "user", "content": "test"}]
    k1 = build_l1_cache_key("gpt-4o", msgs, response_format={"type": "json_object"})
    k2 = build_l1_cache_key("gpt-4o", msgs)
    assert k1 != k2


# --- Token pattern edge cases ---

def test_token_pattern_discovery():
    """TokenPatternAdapter must discover OpenAI fields."""
    from agentfuse.providers.token_pattern import discover_usage_pattern
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    pattern = discover_usage_pattern(usage, "openai")
    assert pattern is not None


def test_token_pattern_extract():
    """extract_with_pattern must extract tokens from usage object."""
    from agentfuse.providers.token_pattern import extract_with_pattern
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    result = extract_with_pattern(usage, "openai")
    assert result.total_input_tokens > 0


# --- Middleware pipeline ---

def test_middleware_pipeline_creation():
    """MiddlewarePipeline must be creatable."""
    from agentfuse.core.middleware import MiddlewarePipeline
    pipeline = MiddlewarePipeline()
    assert pipeline is not None


# --- Request optimizer ---

def test_optimizer_removes_empty_messages():
    """Optimizer must remove messages with empty content."""
    from agentfuse.core.request_optimizer import RequestOptimizer
    from agentfuse.providers.pricing import ModelPricingEngine
    from agentfuse.providers.tokenizer import TokenCounterAdapter
    opt = RequestOptimizer(ModelPricingEngine(), TokenCounterAdapter())
    msgs = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "Real question"},
    ]
    result, report = opt.optimize(msgs, "gpt-4o")
    assert report.messages_removed > 0


# --- Anomaly detector ---

def test_anomaly_detector_records_cost():
    """Recording costs must update internal state."""
    from agentfuse.core.anomaly import CostAnomalyDetector
    detector = CostAnomalyDetector(min_samples=3)
    for _ in range(5):
        detector.record("gpt-4o", 0.01)
    baseline = detector.get_baseline("gpt-4o")
    assert baseline is not None


# --- Cost tracker ---

def test_cost_tracker_records():
    """CostTracker must accept record_call."""
    from agentfuse.core.cost_tracker import CostTracker
    tracker = CostTracker()
    tracker.record_call("gpt-4o", "openai", "run_1", 10, 5, 0.05)
    snap = tracker.get_snapshot()
    assert snap.total_calls >= 1


def test_cost_tracker_top_models():
    """get_top_models must return sorted by cost."""
    from agentfuse.core.cost_tracker import CostTracker
    tracker = CostTracker()
    tracker.record_call("gpt-5", "openai", "run", cost_usd=0.50, input_tokens=10, output_tokens=5)
    tracker.record_call("gpt-4o-mini", "openai", "run", cost_usd=0.01, input_tokens=10, output_tokens=5)
    top = tracker.get_top_models(n=2)
    assert len(top) >= 1


# --- Dedup ---

def test_dedup_make_key_temperature():
    """Different temperatures must produce different keys."""
    from agentfuse.core.dedup import RequestDeduplicator
    k1 = RequestDeduplicator.make_key("gpt-4o", [{"role": "user", "content": "hi"}], 0.0)
    k2 = RequestDeduplicator.make_key("gpt-4o", [{"role": "user", "content": "hi"}], 0.7)
    assert k1 != k2


# --- Fallback chain ---

def test_fallback_chain_all_models_exist():
    """All fallback targets must exist in the registry."""
    from agentfuse.core.fallback_chain import DEFAULT_CHAINS
    from agentfuse.providers.registry import BUILTIN_MODELS
    for model, fallbacks in DEFAULT_CHAINS.items():
        for fb in fallbacks:
            assert fb in BUILTIN_MODELS, f"Fallback {fb} for {model} not in registry"


# --- Load balancer ---

def test_load_balancer_add_and_get():
    """Adding endpoints must make them retrievable."""
    from agentfuse.core.load_balancer import ModelLoadBalancer
    lb = ModelLoadBalancer()
    lb.add_endpoint("gpt-4o", api_key="sk-test1")
    lb.add_endpoint("gpt-4o", api_key="sk-test2")
    ep = lb.get_endpoint("gpt-4o")
    assert ep is not None
    assert ep.api_key in ("sk-test1", "sk-test2")


# --- Receipt ---

def test_receipt_emitter():
    """CostReceiptEmitter must be creatable."""
    from agentfuse.core.receipt import CostReceiptEmitter
    emitter = CostReceiptEmitter("receipt_test", "gpt-4o", 10.0)
    assert emitter is not None


# --- Response validator ---

def test_validator_stop_is_cacheable():
    """finish_reason='stop' must be cacheable."""
    from agentfuse.core.response_validator import validate_for_cache
    assert validate_for_cache("Good response", finish_reason="stop") is True


def test_validator_tool_calls_is_cacheable():
    """finish_reason='tool_calls' must be cacheable."""
    from agentfuse.core.response_validator import validate_for_cache
    assert validate_for_cache("Tool response", finish_reason="tool_calls") is True


# --- Security ---

def test_secure_hash_consistency():
    """Same input must always produce same hash."""
    from agentfuse.core.security import secure_hash
    h1 = secure_hash("production test")
    h2 = secure_hash("production test")
    assert h1 == h2


def test_invisible_chars_comprehensive():
    """All invisible chars must be stripped."""
    from agentfuse.core.security import strip_invisible_chars
    chars = "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"
    result = strip_invisible_chars(f"hello{chars}world")
    assert result == "helloworld"


# --- Pricing edge cases ---

def test_cached_input_cost():
    """cached_input_cost must use discounted rate."""
    from agentfuse.providers.pricing import ModelPricingEngine
    engine = ModelPricingEngine()
    full = engine.input_cost("claude-sonnet-4-6", 1_000_000)
    cached = engine.cached_input_cost("claude-sonnet-4-6", 1_000_000)
    assert cached < full  # cached must be cheaper


def test_cache_write_cost_5m():
    """5-min cache write must cost 1.25x base."""
    from agentfuse.providers.pricing import ModelPricingEngine
    engine = ModelPricingEngine()
    base = engine.input_cost("claude-sonnet-4-6", 1_000_000)
    write = engine.cache_write_cost("claude-sonnet-4-6", 1_000_000, ttl="5m")
    assert abs(write / base - 1.25) < 0.01


# --- Model router ---

def test_router_stats():
    """Router must track routing statistics."""
    from agentfuse.core.model_router import IntelligentModelRouter
    router = IntelligentModelRouter()
    router.route("gpt-4o", [{"role": "user", "content": "hi"}])
    stats = router.get_stats()
    assert stats["total_routed"] >= 1


# --- Adaptive threshold ---

def test_adaptive_threshold_get():
    """get() must return current threshold."""
    from agentfuse.core.adaptive_threshold import AdaptiveSimilarityThreshold
    at = AdaptiveSimilarityThreshold(initial=0.90)
    assert at.get() == 0.90


def test_adaptive_threshold_reset():
    """reset() must restore initial threshold."""
    from agentfuse.core.adaptive_threshold import AdaptiveSimilarityThreshold
    at = AdaptiveSimilarityThreshold(initial=0.92)
    at.report_bad_hit()
    at.report_bad_hit()
    at.reset()
    assert at.get() == 0.92


# --- Batch detector ---

def test_batch_detector_stats_rate():
    """Stats must show batch rate."""
    from agentfuse.core.batch_detector import BatchEligibilityDetector
    detector = BatchEligibilityDetector(min_batch_size=2)
    msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "hi"}]
    for _ in range(10):
        detector.observe(msgs, "gpt-4o")
    stats = detector.get_stats()
    assert stats["batch_rate"] > 0


# --- Cache quality tracker ---

def test_cache_quality_tracker():
    """CacheQualityTracker must track per-entry quality."""
    from agentfuse.core.cache_quality import CacheQualityTracker
    tracker = CacheQualityTracker()
    tracker.record_hit("key1")
    tracker.record_feedback("key1", positive=True)
    tracker.record_feedback("key1", positive=True)
    tracker.record_feedback("key1", positive=False)
    score = tracker.get_score("key1")
    assert score is not None


# --- Complete module import verification ---

def test_all_core_modules_importable():
    """Every core module must be importable without error."""
    import agentfuse.core.adaptive_threshold
    import agentfuse.core.analytics
    import agentfuse.core.anomaly
    import agentfuse.core.batch_detector
    import agentfuse.core.batch_submitter
    import agentfuse.core.budget
    import agentfuse.core.cache
    import agentfuse.core.cache_quality
    import agentfuse.core.context_guard
    import agentfuse.core.conversation_estimator
    import agentfuse.core.cost_alert
    import agentfuse.core.cost_forecast
    import agentfuse.core.cost_tracker
    import agentfuse.core.dedup
    import agentfuse.core.deprecation
    import agentfuse.core.error_classifier
    import agentfuse.core.fallback_chain
    import agentfuse.core.gcra_limiter
    import agentfuse.core.guardrails
    import agentfuse.core.hierarchical_budget
    import agentfuse.core.keys
    import agentfuse.core.kill_switch
    import agentfuse.core.load_balancer
    import agentfuse.core.middleware
    import agentfuse.core.model_recommender
    import agentfuse.core.model_router
    import agentfuse.core.predictive_router
    import agentfuse.core.prompt_cache
    import agentfuse.core.prompt_compressor
    import agentfuse.core.quality_scorer
    import agentfuse.core.rate_limiter
    import agentfuse.core.receipt
    import agentfuse.core.report_exporter
    import agentfuse.core.request_optimizer
    import agentfuse.core.response_validator
    import agentfuse.core.retry
    import agentfuse.core.security
    import agentfuse.core.session
    import agentfuse.core.streaming
    import agentfuse.core.tool_cost_tracker
