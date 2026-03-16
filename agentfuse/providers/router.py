"""
ProviderRouter — resolves model name to (provider, base_url).

Supports OpenAI-compatible providers via base_url swapping,
plus native SDK routing for Anthropic.
"""

import fnmatch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


OPENAI_COMPATIBLE_PROVIDERS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "mistral": "https://api.mistral.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "xai": "https://api.x.ai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}


def resolve_provider(model: str) -> tuple[str, Optional[str]]:
    """
    Resolve a model name to (provider_name, base_url_or_None).

    Returns:
        ("openai", None) — use default OpenAI client
        ("anthropic", None) — use native Anthropic SDK
        ("deepseek", "https://api.deepseek.com") — use OpenAI SDK with base_url
    """
    # Explicit prefix routing: "provider/model-name"
    if "/" in model:
        prefix = model.split("/")[0]
        if prefix in OPENAI_COMPATIBLE_PROVIDERS:
            return (prefix, OPENAI_COMPATIBLE_PROVIDERS[prefix])

    # Anthropic — requires native SDK
    if model.startswith("claude"):
        return ("anthropic", None)

    # OpenAI native models
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return ("openai", None)

    # Gemini
    if model.startswith("gemini"):
        return ("gemini", OPENAI_COMPATIBLE_PROVIDERS["gemini"])

    # Mistral
    if model.startswith("mistral"):
        return ("mistral", OPENAI_COMPATIBLE_PROVIDERS["mistral"])

    # Grok / xAI
    if model.startswith("grok"):
        return ("xai", OPENAI_COMPATIBLE_PROVIDERS["xai"])

    # Fine-tuned models (ft:base-model:org:name)
    if model.startswith("ft:"):
        parts = model.split(":")
        if len(parts) >= 2:
            base = parts[1]
            # Recurse on base model to get provider
            return resolve_provider(base)

    # Wildcard fallback — check all known patterns
    for provider, base_url in OPENAI_COMPATIBLE_PROVIDERS.items():
        if fnmatch.fnmatch(model.lower(), f"{provider}*"):
            return (provider, base_url)

    # Unknown
    logger.warning("Unknown provider for model '%s' — returning ('unknown', None)", model)
    return ("unknown", None)


def list_providers() -> dict[str, str]:
    """List all supported OpenAI-compatible providers and their base URLs."""
    return dict(OPENAI_COMPATIBLE_PROVIDERS)
