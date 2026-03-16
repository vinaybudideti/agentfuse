"""
Loop 4 — Edge case tests across all modules.
"""

import pytest
from agentfuse.core.keys import build_l1_cache_key, extract_semantic_content
from agentfuse.core.budget import BudgetEngine, BudgetExhaustedGracefully
from agentfuse.providers.tokenizer import TokenCounterAdapter
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.providers.registry import ModelRegistry
from agentfuse.providers.router import resolve_provider


# --- Cache key edge cases ---

def test_empty_messages_produce_valid_key():
    """Empty message list must still produce a valid cache key."""
    key = build_l1_cache_key("gpt-4o", [])
    assert key.startswith("agentfuse:v2:cache:")
    assert len(key.split(":")[-1]) == 64


def test_unicode_content_in_key():
    """Unicode content must not crash key generation."""
    key = build_l1_cache_key("gpt-4o", [
        {"role": "user", "content": "こんにちは世界 🌍 Привет мир"}
    ])
    assert key.startswith("agentfuse:v2:cache:")


def test_very_long_message_in_key():
    """Very long messages must produce a valid key."""
    long_content = "x" * 100_000
    key = build_l1_cache_key("gpt-4o", [
        {"role": "user", "content": long_content}
    ])
    assert len(key.split(":")[-1]) == 64  # SHA-256 is always 64 hex chars


def test_extract_semantic_no_user_messages():
    """No user messages → empty semantic content."""
    msgs = [
        {"role": "system", "content": "You are helpful"},
        {"role": "assistant", "content": "OK"},
    ]
    assert extract_semantic_content(msgs) == ""


# --- Budget engine edge cases ---

def test_zero_cost_check_and_act():
    """Zero estimated cost must not trigger any policy."""
    engine = BudgetEngine("run_zero", 1.00, "gpt-4o")
    msgs = [{"role": "user", "content": "hi"}]
    result_msgs, model = engine.check_and_act(0.0, msgs)
    assert model == "gpt-4o"


def test_very_small_budget():
    """Very small budget ($0.001) must still work."""
    engine = BudgetEngine("run_tiny", 0.001, "gpt-4o")
    msgs = [{"role": "user", "content": "hi"}]
    # 80% of 0.001 = 0.0008, estimated cost 0.001 → 100% → terminate
    with pytest.raises(BudgetExhaustedGracefully):
        engine.check_and_act(0.001, msgs)


def test_record_cost_negative_rejected():
    """record_cost with negative should still work (could be a refund)."""
    engine = BudgetEngine("run_neg", 10.0, "gpt-4o")
    engine.record_cost(5.0)
    engine.record_cost(-2.0)  # refund
    assert abs(engine.spent - 3.0) < 0.001


# --- Token counter edge cases ---

def test_count_tokens_newlines_and_special_chars():
    """Token counting must handle newlines and special characters."""
    t = TokenCounterAdapter()
    text = "Line 1\nLine 2\n\tTabbed\n\n\n"
    count = t.count_tokens(text, "gpt-4o")
    assert count > 0


def test_count_messages_empty_content():
    """Messages with empty content must not crash."""
    t = TokenCounterAdapter()
    msgs = [{"role": "user", "content": ""}]
    count = t.count_messages(msgs, "gpt-4o")
    assert count == 7  # 0 tokens + 4 overhead + 3 priming


def test_count_messages_missing_content():
    """Messages without content key must not crash."""
    t = TokenCounterAdapter()
    msgs = [{"role": "user"}]
    count = t.count_messages(msgs, "gpt-4o")
    assert count == 7  # "" → 0 tokens + 4 overhead + 3 priming


# --- Pricing edge cases ---

def test_total_cost_zero_tokens():
    """Zero tokens must cost zero."""
    p = ModelPricingEngine()
    assert p.total_cost("gpt-4o", 0, 0) == 0.0


def test_overrides_in_pricing():
    """User overrides must work through ModelPricingEngine."""
    p = ModelPricingEngine(overrides={
        "custom-model": {"input": 5.0, "output": 10.0}
    })
    assert p.input_cost("custom-model", 1_000_000) == 5.0


# --- Router edge cases ---

def test_resolve_empty_string():
    """Empty string model must not crash."""
    provider, base_url = resolve_provider("")
    assert isinstance(provider, str)


def test_resolve_slashes_in_model():
    """Models with slashes must route to the prefix provider."""
    provider, _ = resolve_provider("together/llama-3.3-70b")
    assert provider == "together"


def test_registry_list_models():
    """list_models must return sorted model names."""
    from agentfuse.providers.registry import ModelRegistry
    r = ModelRegistry(refresh_hours=0)
    models = r.list_models()
    assert len(models) > 15
    assert models == sorted(models)
    assert "gpt-4o" in models
    assert "claude-sonnet-4-6" in models


def test_normalized_usage_repr():
    """NormalizedUsage repr must be readable."""
    from agentfuse.providers.response import NormalizedUsage
    u = NormalizedUsage(total_input_tokens=100, total_output_tokens=200, provider="openai")
    r = repr(u)
    assert "input=100" in r
    assert "output=200" in r


def test_budget_engine_repr():
    """BudgetEngine repr must include key state info."""
    engine = BudgetEngine("repr_test", 5.0, "gpt-4o")
    engine.record_cost(1.5)
    r = repr(engine)
    assert "repr_test" in r
    assert "$5.00" in r
    assert "$1.5" in r
    assert "gpt-4o" in r
    assert "normal" in r


def test_jump_to_90_pct_both_downgrades_and_compresses():
    """Going straight to 90%+ must both downgrade AND compress."""
    engine = BudgetEngine("run_jump90", 1.00, "gpt-4o")
    engine.spent = 0.85
    msgs = (
        [{"role": "system", "content": "sys"}]
        + [{"role": "user", "content": f"m{i}"} for i in range(10)]
    )
    compressed, model = engine.check_and_act(0.06, msgs)
    # Should downgrade (80%+) AND compress (90%+)
    assert model == "gpt-4o-mini"
    non_system = [m for m in compressed if m["role"] != "system"]
    assert len(non_system) == 6


def test_budget_summary():
    """get_budget_summary returns correct utilization."""
    from agentfuse.storage.memory import InMemoryBudgetStore
    store = InMemoryBudgetStore()
    store.create_run("run1", 10.0)
    store.check_and_deduct("run1", 3.0)
    summary = store.get_budget_summary("run1")
    assert summary["initial_budget"] == 10.0
    assert summary["remaining"] == 7.0
    assert summary["spent"] == 3.0
    assert abs(summary["utilization_pct"] - 30.0) < 0.01
