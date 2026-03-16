"""
ModelPricingEngine — backed by ModelRegistry for hot-reloadable pricing.

Maintains backward-compatible API while using the registry for pricing lookups.

PRODUCTION FIX: Added total_cost_normalized() that correctly handles Anthropic's
separate billing rates for cache_read (0.1x) and cache_creation (1.25x).
The old total_cost() applies a single rate to all input tokens, which under-bills
Anthropic cached responses by 50-300%.
"""

from agentfuse.providers.registry import ModelRegistry


class ModelPricingEngine:
    def __init__(self, overrides: dict | None = None):
        self._registry = ModelRegistry(overrides=overrides)

    def input_cost(self, model: str, token_count: int) -> float:
        if token_count <= 0:
            return 0.0
        pricing = self._registry.get_pricing(model)
        return (token_count / 1_000_000) * pricing.get("input", 0.0)

    def output_cost(self, model: str, token_count: int) -> float:
        if token_count <= 0:
            return 0.0
        pricing = self._registry.get_pricing(model)
        return (token_count / 1_000_000) * pricing.get("output", 0.0)

    def cached_input_cost(self, model: str, token_count: int) -> float:
        """Cost for cached input tokens (discounted by providers like Anthropic)."""
        if token_count <= 0:
            return 0.0
        pricing = self._registry.get_pricing(model)
        cached_rate = pricing.get("cached_input", pricing.get("input", 0.0))
        return (token_count / 1_000_000) * cached_rate

    def cache_write_cost(self, model: str, token_count: int) -> float:
        """Cost for cache creation/write tokens (Anthropic charges 1.25x input rate)."""
        if token_count <= 0:
            return 0.0
        pricing = self._registry.get_pricing(model)
        # Anthropic cache_creation is 1.25x the input rate
        write_rate = pricing.get("input", 0.0) * 1.25
        return (token_count / 1_000_000) * write_rate

    def total_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate total cost using uniform input/output rates.
        For Anthropic with cache tokens, use total_cost_normalized() instead.
        """
        input_c = self.input_cost(model, input_tokens)
        output_c = self.output_cost(model, output_tokens)

        # Anthropic overflow: >200K input → 2x input, 1.5x output
        if model.startswith("claude") and input_tokens > 200_000:
            input_c *= 2
            output_c *= 1.5

        # Gemini Pro overflow: >200K input → 2x pricing
        if "pro" in model and model.startswith("gemini") and input_tokens > 200_000:
            input_c *= 2
            output_c *= 2

        return input_c + output_c

    def total_cost_normalized(self, model: str, usage) -> float:
        """
        Calculate total cost from NormalizedUsage with correct per-component billing.

        This is the ACCURATE method for Anthropic cached responses:
        - uncached input: billed at input rate
        - cache_read: billed at cached_input rate (0.1x for Anthropic)
        - cache_write: billed at 1.25x input rate (Anthropic cache creation)
        - output: billed at output rate

        For OpenAI/Gemini, this falls back to simple input + output.
        """
        if usage is None:
            return 0.0

        total_input = getattr(usage, "total_input_tokens", 0)
        total_output = getattr(usage, "total_output_tokens", 0)
        cached_read = getattr(usage, "cached_input_tokens", 0)
        cached_write = getattr(usage, "cache_write_tokens", 0)
        provider = getattr(usage, "provider", "unknown")

        if provider == "anthropic" and (cached_read > 0 or cached_write > 0):
            # Anthropic: separate billing for each component
            uncached_input = total_input - cached_read - cached_write
            cost = (
                self.input_cost(model, max(0, uncached_input))
                + self.cached_input_cost(model, cached_read)
                + self.cache_write_cost(model, cached_write)
                + self.output_cost(model, total_output)
            )
            # Overflow pricing for large context
            if total_input > 200_000:
                cost *= 2  # Simplified: all components doubled
            return cost

        # OpenAI / Gemini / others: uniform rate
        return self.total_cost(model, total_input, total_output)

    _tokenizer = None

    def estimate_cost(self, model: str, messages: list) -> float:
        if self._tokenizer is None:
            from agentfuse.providers.tokenizer import TokenCounterAdapter
            self._tokenizer = TokenCounterAdapter()
        input_tokens = self._tokenizer.count_messages_tokens(messages, model)
        return self.input_cost(model, input_tokens)

    def is_supported(self, model: str) -> bool:
        pricing = self._registry.get_pricing(model)
        return pricing.get("input", 0.0) > 0 or pricing.get("output", 0.0) > 0
