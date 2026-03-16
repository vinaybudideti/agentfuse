"""
Phase 7 — Pricing behavioral tests.
"""

from agentfuse.providers.pricing import ModelPricingEngine


def test_gpt4o_1M_input_tokens_costs_2_50():
    """1M input tokens on gpt-4o must cost exactly $2.50."""
    p = ModelPricingEngine()
    assert p.input_cost("gpt-4o", 1_000_000) == 2.50


def test_claude_sonnet_total_cost_correct():
    """1M input + 1M output on claude-sonnet-4-6 must cost $18.00."""
    p = ModelPricingEngine()
    cost = p.total_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
    assert cost == 18.00


def test_unknown_model_returns_zero_not_raises():
    """Unknown model must return $0.00, not raise."""
    p = ModelPricingEngine()
    cost = p.input_cost("completely-unknown-model", 1_000_000)
    assert cost == 0.0
    cost = p.output_cost("completely-unknown-model", 1_000_000)
    assert cost == 0.0


def test_o3_pricing():
    """o3 pricing at March 2026 rates."""
    p = ModelPricingEngine()
    assert p.input_cost("o3", 1_000_000) == 2.00
    assert p.output_cost("o3", 1_000_000) == 8.00


def test_gemini_flash_pricing():
    """Gemini 2.0 Flash is ultra-cheap."""
    p = ModelPricingEngine()
    assert p.input_cost("gemini-2.0-flash", 1_000_000) == 0.10
    assert p.output_cost("gemini-2.0-flash", 1_000_000) == 0.40


def test_is_supported():
    """Known models must be supported, unknown must not."""
    p = ModelPricingEngine()
    assert p.is_supported("gpt-4o")
    assert p.is_supported("claude-sonnet-4-6")
    assert not p.is_supported("totally-fake-model")


def test_estimate_cost_messages():
    """estimate_cost must calculate input cost from messages."""
    p = ModelPricingEngine()
    messages = [{"role": "user", "content": "Hello world"}]
    cost = p.estimate_cost("gpt-4o", messages)
    assert cost > 0  # Should be tiny but non-zero


def test_cached_input_cost_cheaper_than_regular():
    """Cached input cost must be <= regular input cost."""
    p = ModelPricingEngine()
    regular = p.input_cost("claude-sonnet-4-6", 1_000_000)
    cached = p.cached_input_cost("claude-sonnet-4-6", 1_000_000)
    assert cached < regular
    assert cached == 0.30  # Anthropic cached rate
    assert regular == 3.00


def test_cached_input_cost_falls_back_to_regular():
    """Models without cached_input pricing fall back to regular input."""
    p = ModelPricingEngine()
    regular = p.input_cost("mistral-large-latest", 1_000_000)
    cached = p.cached_input_cost("mistral-large-latest", 1_000_000)
    assert cached == regular  # No cached rate defined
