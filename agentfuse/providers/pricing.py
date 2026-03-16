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
        return self.input_cost(model, input_tokens) + self.output_cost(model, output_tokens)

    def estimate_cost(self, model: str, messages: list) -> float:
        from agentfuse.providers.tokenizer import TokenCounterAdapter
        counter = TokenCounterAdapter()
        input_tokens = counter.count_messages_tokens(messages, model)
        return self.input_cost(model, input_tokens)

    def is_supported(self, model: str) -> bool:
        pricing = self._registry.get_pricing(model)
        return pricing.get("input", 0.0) > 0 or pricing.get("output", 0.0) > 0
