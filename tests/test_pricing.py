# Tests for ModelPricingEngine: verify model prices match official provider pricing

import pytest
from agentfuse.providers.pricing import ModelPricingEngine


def test_gpt4o_input_cost():
    p = ModelPricingEngine()
    cost = p.input_cost("gpt-4o", 1_000_000)
    assert cost == 2.50


def test_gpt4o_output_cost():
    p = ModelPricingEngine()
    cost = p.output_cost("gpt-4o", 1_000_000)
    assert cost == 10.00


def test_claude_sonnet_total():
    p = ModelPricingEngine()
    cost = p.total_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
    assert cost == 18.00


def test_unsupported_model_returns_zero():
    """Unknown models return zero cost (no crash) — backed by ModelRegistry."""
    p = ModelPricingEngine()
    cost = p.input_cost("fake-model-xyz", 1000)
    assert cost == 0.0
