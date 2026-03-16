"""
StreamingCostMiddleware — per-token cost accumulation during streaming responses.

Wraps a streaming LLM generator, tracks cost per output token in real time,
and aborts if max_stream_cost is exceeded.
"""


class StreamCostLimitReached(Exception):
    def __init__(self, cost, tokens):
        self.cost = cost
        self.tokens = tokens
        super().__init__(
            f"Stream cost limit reached: ${cost:.4f} after {tokens} tokens"
        )


class StreamingCostMiddleware:

    def __init__(self, model: str, pricing_engine, budget_engine,
                 max_stream_cost=None):
        self.model = model
        self.pricing = pricing_engine
        self.budget_engine = budget_engine
        self.max_stream_cost = max_stream_cost
        self.token_count = 0
        self.stream_cost = 0.0

    def wrap_stream(self, stream_generator, input_tokens: int):
        """
        Wraps a streaming LLM response generator.
        Yields (chunk, current_cost) tuples.
        Raises StreamCostLimitReached if max_stream_cost exceeded.
        """
        input_cost = self.pricing.input_cost(self.model, input_tokens)

        for chunk in stream_generator:
            has_content = False

            # OpenAI streaming format
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    has_content = True
            # Anthropic streaming format
            elif hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                if chunk.delta.text:
                    has_content = True

            if has_content:
                self.token_count += 1
                output_cost = self.pricing.output_cost(
                    self.model, self.token_count
                )
                self.stream_cost = input_cost + output_cost

                if (self.max_stream_cost and
                        self.stream_cost >= self.max_stream_cost):
                    raise StreamCostLimitReached(
                        cost=self.stream_cost,
                        tokens=self.token_count,
                    )

            yield chunk, self.stream_cost

    def get_final_cost(self) -> float:
        return self.stream_cost
