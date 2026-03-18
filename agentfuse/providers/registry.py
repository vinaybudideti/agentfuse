"""
ModelRegistry — hot-reloadable model pricing with LiteLLM remote refresh.

Pricing is per 1M tokens (USD). Supports 4-tier lookup:
1. User overrides (highest priority)
2. Exact model name match
3. Fine-tuned model (ft: prefix) → 2x base price
4. Unknown → warn + return zero pricing
"""

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


# March 2026 pricing — verified from research (per 1M tokens, USD)
# Cache discount multipliers:
#   OpenAI GPT-5.x: 90% discount (cached = 10% of base)
#   OpenAI GPT-4.1/o3/o4-mini: 75% discount (cached = 25% of base)
#   OpenAI GPT-4o: 50% discount (cached = 50% of base)
#   Anthropic: cache read = 0.1x base (90% discount); cache write = 1.25x base
#   Google: cache read = 0.1x base (90% discount)
BUILTIN_MODELS: dict[str, dict] = {
    # OpenAI (legacy — 50% cache discount)
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25, "context": 128_000, "max_output": 16_000, "provider": "openai"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075, "context": 128_000, "max_output": 16_000, "provider": "openai"},
    # OpenAI current (75% cache discount)
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.20, "context": 1_000_000, "max_output": 32_000, "provider": "openai"},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.04, "context": 1_000_000, "max_output": 32_000, "provider": "openai"},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.01, "context": 1_000_000, "max_output": 32_000, "provider": "openai"},
    "o3": {"input": 2.00, "output": 8.00, "cached_input": 0.50, "context": 200_000, "max_output": 100_000, "provider": "openai"},
    "o4-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.275, "context": 200_000, "max_output": 100_000, "provider": "openai"},
    # OpenAI GPT-5 family (90% cache discount)
    "gpt-5": {"input": 1.25, "output": 10.00, "cached_input": 0.125, "context": 128_000, "max_output": 32_000, "provider": "openai"},
    "gpt-5.3": {"input": 1.50, "output": 10.00, "cached_input": 0.15, "context": 1_000_000, "max_output": 32_000, "provider": "openai"},
    "gpt-5.4": {"input": 2.50, "output": 15.00, "cached_input": 0.25, "context": 1_050_000, "max_output": 32_000, "provider": "openai"},
    "gpt-5.4-pro": {"input": 30.00, "output": 180.00, "context": 1_050_000, "max_output": 32_000, "provider": "openai"},
    # Anthropic (cache read = 0.1x, cache write = 1.25x)
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cached_input": 0.30, "context": 200_000, "max_output": 64_000, "provider": "anthropic"},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00, "cached_input": 0.10, "context": 200_000, "max_output": 64_000, "provider": "anthropic"},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00, "cached_input": 0.50, "context": 200_000, "max_output": 64_000, "provider": "anthropic"},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00, "cached_input": 0.30, "context": 200_000, "max_output": 64_000, "provider": "anthropic"},
    # Google Gemini (cache read = 0.1x)
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40, "cached_input": 0.01, "context": 1_000_000, "max_output": 8_000, "provider": "gemini"},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached_input": 0.125, "context": 2_000_000, "max_output": 64_000, "provider": "gemini"},
    # DeepSeek (90% cache discount)
    "deepseek/deepseek-chat": {"input": 0.28, "output": 0.42, "cached_input": 0.028, "context": 128_000, "max_output": 64_000, "provider": "deepseek"},
    "deepseek/deepseek-reasoner": {"input": 0.55, "output": 2.19, "cached_input": 0.055, "context": 128_000, "max_output": 64_000, "provider": "deepseek"},
    # Mistral
    "mistral-large-latest": {"input": 0.50, "output": 1.50, "context": 128_000, "max_output": 128_000, "provider": "mistral"},
    "mistral-small-latest": {"input": 0.10, "output": 0.30, "context": 128_000, "max_output": 128_000, "provider": "mistral"},
    # xAI
    "grok-4.1-fast": {"input": 0.20, "output": 0.50, "cached_input": 0.05, "context": 2_000_000, "max_output": 131_000, "provider": "xai"},
    "grok-4.20": {"input": 2.00, "output": 6.00, "context": 256_000, "max_output": 131_000, "provider": "xai"},
    # Groq (Llama)
    "groq/llama-3.3-70b": {"input": 0.59, "output": 0.79, "context": 128_000, "max_output": 128_000, "provider": "groq"},
    # Together AI (Llama)
    "together/llama-3.3-70b": {"input": 0.88, "output": 0.88, "context": 128_000, "max_output": 128_000, "provider": "together"},
    # OpenAI o1
    "o1": {"input": 15.00, "output": 60.00, "context": 200_000, "max_output": 100_000, "provider": "openai"},
    # Mistral (updated March 2026)
    "mistral-medium-3": {"input": 0.40, "output": 2.00, "context": 128_000, "max_output": 128_000, "provider": "mistral"},
    # DeepSeek (updated)
    "deepseek/deepseek-v3.2": {"input": 0.28, "output": 0.42, "cached_input": 0.028, "context": 128_000, "max_output": 64_000, "provider": "deepseek"},
    # Meta Llama 4
    "together/llama-4-maverick": {"input": 0.27, "output": 0.85, "context": 1_000_000, "max_output": 64_000, "provider": "together"},
    # OpenAI gpt-oss (open-weight, Apache 2.0, self-hosted pricing varies)
    "gpt-oss-120b": {"input": 0.15, "output": 0.60, "context": 131_072, "max_output": 32_000, "provider": "openai"},
    "gpt-oss-20b": {"input": 0.03, "output": 0.14, "context": 131_072, "max_output": 32_000, "provider": "openai"},
    # Legacy compat
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "context": 128_000, "max_output": 4_096, "provider": "openai"},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30, "cached_input": 0.0075, "context": 1_000_000, "max_output": 8_000, "provider": "gemini"},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00, "cached_input": 0.125, "context": 1_000_000, "max_output": 64_000, "provider": "gemini"},
}

PROVIDER_PREFIXES: dict[str, str] = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude-": "anthropic",
    "gemini-": "gemini",
    "mistral-": "mistral",
    "deepseek/": "deepseek",
    "grok-": "xai",
    "llama": "meta",
    "command-": "cohere",
}


class ModelRegistry:
    """
    Hot-reloadable model registry with LiteLLM remote refresh.

    Lookup priority:
    1. User overrides (constructor param)
    2. Exact match in local registry
    3. ft: prefix → 2x base model price
    4. Unknown → log warning, return zero pricing
    """

    LITELLM_PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

    def __init__(self, refresh_hours: float = 24.0, overrides: Optional[dict] = None):
        self._models: dict[str, dict] = dict(BUILTIN_MODELS)
        self._overrides: dict[str, dict] = overrides or {}
        self._refresh_hours = float(os.environ.get("AGENTFUSE_REGISTRY_REFRESH_HOURS", str(refresh_hours)))
        self._last_refresh: float = 0.0
        self._refresh_attempted = False
        self._refresh_lock = threading.Lock()

    def get_pricing(self, model: str) -> dict:
        """
        4-tier lookup: overrides → exact → ft: pattern → warn+zero.
        Returns dict with 'input' and 'output' keys (per 1M tokens).
        """
        # Tier 1: user overrides
        if model in self._overrides:
            return self._overrides[model]

        # Maybe refresh from remote
        self._maybe_refresh()

        # Tier 2: exact match
        if model in self._models:
            return self._models[model]

        # Tier 3: fine-tuned model → 2x base price (including cached_input)
        if model.startswith("ft:"):
            base_model = self._extract_base_from_ft(model)
            if base_model and base_model in self._models:
                base = self._models[base_model]
                ft_pricing = {**base, "input": base["input"] * 2, "output": base["output"] * 2}
                if "cached_input" in base:
                    ft_pricing["cached_input"] = base["cached_input"] * 2
                return ft_pricing

        # Tier 4: unknown → warn + zero
        logger.warning("Unknown model '%s' — returning zero pricing. Add to overrides or registry.", model)
        return {"input": 0.0, "output": 0.0, "context": 0, "provider": "unknown"}

    def _extract_base_from_ft(self, model: str) -> Optional[str]:
        """Extract base model from ft: prefix. Format: ft:base-model:org:name."""
        parts = model.split(":")
        if len(parts) >= 2:
            return parts[1]
        return None

    def _detect_provider(self, model: str) -> str:
        """Detect provider from model name using prefix matching."""
        for prefix, provider in PROVIDER_PREFIXES.items():
            if model.startswith(prefix):
                return provider
        if "/" in model:
            return model.split("/")[0]
        return "unknown"

    def _maybe_refresh(self):
        """Check if refresh is needed based on time interval. Thread-safe."""
        if self._refresh_hours <= 0:
            return
        now = time.time()
        if now - self._last_refresh < self._refresh_hours * 3600:
            return
        # Prevent concurrent refresh attempts
        if self._refresh_lock.acquire(blocking=False):
            try:
                # Double-check after acquiring lock
                if time.time() - self._last_refresh >= self._refresh_hours * 3600:
                    self._refresh_from_remote()
            finally:
                self._refresh_lock.release()

    def _refresh_from_remote(self):
        """Fetch LiteLLM pricing JSON. Never crashes — logs warning on failure."""
        try:
            import httpx
            resp = httpx.get(self.LITELLM_PRICES_URL, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            updated = 0
            for model_name, info in data.items():
                if isinstance(info, dict) and "input_cost_per_token" in info:
                    input_per_m = info["input_cost_per_token"] * 1_000_000
                    output_per_m = info.get("output_cost_per_token", 0) * 1_000_000
                    if model_name not in self._models:
                        self._models[model_name] = {
                            "input": input_per_m,
                            "output": output_per_m,
                            "context": info.get("max_input_tokens", 0),
                            "provider": self._detect_provider(model_name),
                        }
                        updated += 1
            logger.info("Registry refreshed from LiteLLM: %d new models added", updated)
        except Exception as e:
            logger.warning("Registry remote refresh failed (using local prices): %s", e)
        finally:
            self._last_refresh = time.time()
            self._refresh_attempted = True

    def list_models(self) -> list[str]:
        """List all known model names."""
        return sorted(set(list(self._models.keys()) + list(self._overrides.keys())))

    def get_provider(self, model: str) -> str:
        """Get the provider name for a model."""
        pricing = self.get_pricing(model)
        if "provider" in pricing:
            return pricing["provider"]
        return self._detect_provider(model)
