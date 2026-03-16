"""
Loop 39 — Advanced ProviderRouter tests.
"""

from agentfuse.providers.router import resolve_provider, OPENAI_COMPATIBLE_PROVIDERS


def test_all_compatible_providers_have_base_url():
    """Every OpenAI-compatible provider must have a valid base_url."""
    for provider, url in OPENAI_COMPATIBLE_PROVIDERS.items():
        assert url.startswith("http"), f"{provider} has invalid URL: {url}"


def test_ollama_routes_to_localhost():
    """Ollama must route to localhost."""
    provider, url = resolve_provider("ollama/llama-3")
    assert provider == "ollama"
    assert "localhost" in url


def test_openrouter_routes_correctly():
    provider, url = resolve_provider("openrouter/anthropic/claude")
    assert provider == "openrouter"
    assert "openrouter" in url


def test_fireworks_routes_correctly():
    provider, url = resolve_provider("fireworks/llama-v3")
    assert provider == "fireworks"
    assert "fireworks" in url


def test_vllm_routes_to_localhost():
    provider, url = resolve_provider("vllm/my-model")
    assert provider == "vllm"
    assert "localhost" in url


def test_ft_prefix_recursive():
    """Fine-tuned models must recurse to base provider."""
    provider, url = resolve_provider("ft:claude-sonnet-4-6:org:name")
    # claude- → anthropic
    assert provider == "anthropic"
    assert url is None


def test_multiple_slashes():
    """Provider/subpath/model must use the first segment as provider."""
    provider, url = resolve_provider("together/meta-llama/Llama-3")
    assert provider == "together"
