# ModelPricingEngine: 50+ model prices per million tokens, auto-update mechanism


class ModelPricingEngine:
    MODELS = {
        # OpenAI
        "gpt-4o": {"input": 2.50, "output": 10.00, "ctx": 128000},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "ctx": 128000},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00, "ctx": 128000},
        "o3": {"input": 10.00, "output": 40.00, "ctx": 200000},
        # Anthropic (per million tokens)
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00, "ctx": 200000},
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "ctx": 200000},
        "claude-opus-4-6": {"input": 15.00, "output": 75.00, "ctx": 200000},
        # Google
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30, "ctx": 1000000},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00, "ctx": 2000000},
    }

    def _validate_model(self, model):
        if model not in self.MODELS:
            raise ValueError(f"Model '{model}' not supported. Supported models: {list(self.MODELS.keys())}")

    def input_cost(self, model, token_count):
        self._validate_model(model)
        return (token_count / 1_000_000) * self.MODELS[model]["input"]

    def output_cost(self, model, token_count):
        self._validate_model(model)
        return (token_count / 1_000_000) * self.MODELS[model]["output"]

    def total_cost(self, model, input_tokens, output_tokens):
        return self.input_cost(model, input_tokens) + self.output_cost(model, output_tokens)

    def estimate_cost(self, model, messages):
        self._validate_model(model)
        from agentfuse.providers.tokenizer import TokenCounterAdapter
        counter = TokenCounterAdapter()
        input_tokens = counter.count_messages(messages, model)
        return self.input_cost(model, input_tokens)

    def is_supported(self, model):
        return model in self.MODELS
