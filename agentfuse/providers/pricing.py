"""
ModelPricingEngine — backed by ModelRegistry for hot-reloadable pricing.

Maintains backward-compatible API while using the registry for pricing lookups.
"""

from agentfuse.providers.registry import ModelRegistry


class ModelPricingEngine:
    def __init__(self, overrides: dict | None = None):
        self._registry = ModelRegistry(overrides=overrides)

    def input_cost(self, model: str, token_count: int) -> float:
        pricing = self._registry.get_pricing(model)
        return (token_count / 1_000_000) * pricing.get("input", 0.0)

    def output_cost(self, model: str, token_count: int) -> float:
        pricing = self._registry.get_pricing(model)
        return (token_count / 1_000_000) * pricing.get("output", 0.0)

    def cached_input_cost(self, model: str, token_count: int) -> float:
        """Cost for cached input tokens (discounted by providers like Anthropic)."""
        pricing = self._registry.get_pricing(model)
        cached_rate = pricing.get("cached_input", pricing.get("input", 0.0))
        return (token_count / 1_000_000) * cached_rate

    def total_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate total cost. Applies overflow pricing for Anthropic models
        when input exceeds 200K tokens (2x input, 1.5x output).
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
