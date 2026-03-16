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
