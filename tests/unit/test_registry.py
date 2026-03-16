"""
Phase 1 — ModelRegistry + ProviderRouter tests.
"""

from agentfuse.providers.registry import ModelRegistry
from agentfuse.providers.router import resolve_provider


def test_known_model_returns_pricing():
    """Known models must return non-zero pricing."""
    reg = ModelRegistry(refresh_hours=0)
    pricing = reg.get_pricing("gpt-4o")
    assert pricing["input"] == 2.50
    assert pricing["output"] == 10.00


def test_unknown_model_returns_zero_not_crash():
    """Unknown model must return zero pricing, not raise."""
    reg = ModelRegistry(refresh_hours=0)
    pricing = reg.get_pricing("totally-unknown-model-xyz")
    assert pricing["input"] == 0.0
    assert pricing["output"] == 0.0


def test_anthropic_model_detected_correctly():
    """Claude models must be detected as anthropic provider."""
    reg = ModelRegistry(refresh_hours=0)
    assert reg._detect_provider("claude-sonnet-4-6") == "anthropic"
    assert reg._detect_provider("claude-opus-4-6") == "anthropic"
    assert reg._detect_provider("claude-haiku-4-5-20251001") == "anthropic"


def test_fine_tuned_model_gets_2x_multiplier():
    """ft:gpt-4o:org:name must return 2x gpt-4o price."""
    reg = ModelRegistry(refresh_hours=0)
    pricing = reg.get_pricing("ft:gpt-4o:my-org:custom-model")
    base = reg.get_pricing("gpt-4o")
    assert pricing["input"] == base["input"] * 2
    assert pricing["output"] == base["output"] * 2


def test_provider_routing_claude_uses_native_sdk():
    """claude models must route to anthropic with no base_url."""
    provider, base_url = resolve_provider("claude-sonnet-4-6")
    assert provider == "anthropic"
    assert base_url is None


def test_provider_routing_deepseek_gets_base_url():
    """DeepSeek must route via OpenAI-compatible with base_url."""
    provider, base_url = resolve_provider("deepseek/deepseek-chat")
    assert provider == "deepseek"
    assert "deepseek" in base_url


def test_new_model_works_without_code_change():
    """A model with unknown provider prefix must not crash."""
    provider, base_url = resolve_provider("newprovider/new-model-v1")
    # Should not crash — returns unknown or attempts wildcard
    assert isinstance(provider, str)


def test_overrides_take_priority():
    """User overrides must override builtin pricing."""
    reg = ModelRegistry(refresh_hours=0, overrides={
        "gpt-4o": {"input": 99.0, "output": 99.0}
    })
    pricing = reg.get_pricing("gpt-4o")
    assert pricing["input"] == 99.0


def test_openai_models_route_correctly():
    """GPT and o-series models must route to openai."""
    assert resolve_provider("gpt-4o")[0] == "openai"
    assert resolve_provider("gpt-4.1")[0] == "openai"
    assert resolve_provider("o3")[0] == "openai"
    assert resolve_provider("o4-mini")[0] == "openai"


def test_gemini_routes_with_base_url():
    """Gemini models must get Google's OpenAI-compatible base_url."""
    provider, base_url = resolve_provider("gemini-2.5-pro")
    assert provider == "gemini"
    assert base_url is not None


def test_fine_tuned_model_routes_to_base_provider():
    """ft:gpt-4o:org:name must route to openai."""
    provider, base_url = resolve_provider("ft:gpt-4o:my-org:custom")
    assert provider == "openai"


def test_pricing_engine_uses_registry():
    """ModelPricingEngine must use ModelRegistry under the hood."""
    from agentfuse.providers.pricing import ModelPricingEngine
    p = ModelPricingEngine()
    # Known model — should return non-zero cost
    cost = p.input_cost("gpt-4o", 1_000_000)
    assert cost == 2.50
    # Unknown model — should return 0, not crash
    cost = p.input_cost("totally-unknown-xyz", 1000)
    assert cost == 0.0


def test_registry_env_var_refresh_hours():
    """AGENTFUSE_REGISTRY_REFRESH_HOURS env var must override constructor."""
    import os
    os.environ["AGENTFUSE_REGISTRY_REFRESH_HOURS"] = "48"
    try:
        reg = ModelRegistry(refresh_hours=24.0)
        assert reg._refresh_hours == 48.0
    finally:
        del os.environ["AGENTFUSE_REGISTRY_REFRESH_HOURS"]
