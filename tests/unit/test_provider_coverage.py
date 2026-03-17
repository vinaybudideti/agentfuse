"""
Provider-specific tests for pricing, routing, and tokenizer edge cases.
"""

from agentfuse.providers.registry import ModelRegistry, BUILTIN_MODELS
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.providers.tokenizer import TokenCounterAdapter
from agentfuse.providers.router import resolve_provider, OPENAI_COMPATIBLE_PROVIDERS
from agentfuse.providers.response import extract_usage, NormalizedUsage
from types import SimpleNamespace


# --- Registry ---

def test_all_builtin_models_have_provider():
    """Every model in BUILTIN_MODELS must have a provider field."""
    for model, info in BUILTIN_MODELS.items():
        assert "provider" in info, f"{model} missing provider"


def test_all_builtin_models_have_input_output():
    """Every model must have input and output pricing."""
    for model, info in BUILTIN_MODELS.items():
        assert "input" in info, f"{model} missing input pricing"
        assert "output" in info, f"{model} missing output pricing"


def test_registry_has_25_plus_models():
    """Registry must have 25+ models."""
    assert len(BUILTIN_MODELS) >= 25


def test_registry_overrides():
    """User overrides must take priority."""
    reg = ModelRegistry(refresh_hours=0, overrides={"custom-model": {"input": 1.0, "output": 2.0}})
    p = reg.get_pricing("custom-model")
    assert p["input"] == 1.0


# --- Pricing edge cases ---

def test_pricing_zero_tokens():
    """Zero tokens must cost zero."""
    engine = ModelPricingEngine()
    assert engine.input_cost("gpt-4o", 0) == 0.0
    assert engine.output_cost("gpt-4o", 0) == 0.0


def test_pricing_negative_tokens():
    """Negative tokens must cost zero."""
    engine = ModelPricingEngine()
    assert engine.input_cost("gpt-4o", -100) == 0.0


def test_pricing_unknown_model_zero():
    """Unknown model must return zero cost."""
    engine = ModelPricingEngine()
    assert engine.input_cost("totally-unknown-model", 1000) == 0.0


def test_pricing_anthropic_overflow():
    """Anthropic >200K input must apply 2x multiplier."""
    engine = ModelPricingEngine()
    normal = engine.total_cost("claude-sonnet-4-6", 100_000, 1000)
    overflow = engine.total_cost("claude-sonnet-4-6", 300_000, 1000)
    # Overflow should cost more than 3x normal (2x multiplier on input + 1.5x on output)
    assert overflow > normal * 2


def test_pricing_gemini_overflow():
    """Gemini Pro >200K input must apply 2x multiplier."""
    engine = ModelPricingEngine()
    normal = engine.total_cost("gemini-2.5-pro", 100_000, 1000)
    overflow = engine.total_cost("gemini-2.5-pro", 300_000, 1000)
    assert overflow > normal * 2


# --- Tokenizer edge cases ---

def test_tokenizer_all_providers():
    """Tokenizer must handle all major providers without crash."""
    tc = TokenCounterAdapter()
    models = ["gpt-5", "claude-sonnet-4-6", "gemini-2.5-pro", "mistral-large-latest",
              "deepseek/deepseek-chat", "grok-4.1-fast", "gpt-oss-120b"]
    for model in models:
        tokens = tc.count_tokens("Hello, world!", model)
        assert tokens > 0, f"{model} returned 0 tokens"


def test_tokenizer_unicode():
    """Unicode text must be handled."""
    tc = TokenCounterAdapter()
    tokens = tc.count_tokens("こんにちは世界 🌍", "gpt-4o")
    assert tokens > 0


# --- Router edge cases ---

def test_router_all_providers_known():
    """All OPENAI_COMPATIBLE_PROVIDERS must resolve correctly."""
    for provider in OPENAI_COMPATIBLE_PROVIDERS:
        model = f"{provider}/test-model"
        name, url = resolve_provider(model)
        assert name == provider, f"{model} resolved to {name}, expected {provider}"
        assert url is not None


def test_router_fine_tuned():
    """Fine-tuned model must resolve to base model's provider."""
    name, url = resolve_provider("ft:gpt-4o:my-org:custom:id")
    assert name == "openai"


# --- Usage normalization edge cases ---

def test_usage_none_returns_empty():
    """None usage must return empty NormalizedUsage."""
    result = extract_usage("openai", None)
    assert result.total_input_tokens == 0
    assert result.total_output_tokens == 0


def test_usage_missing_fields_no_crash():
    """Usage with missing fields must not crash."""
    usage = SimpleNamespace()  # no fields at all
    result = extract_usage("openai", usage)
    assert result.total_input_tokens == 0


def test_normalized_usage_total_property():
    """total_tokens property must sum input + output."""
    usage = NormalizedUsage(total_input_tokens=100, total_output_tokens=50)
    assert usage.total_tokens == 150
