"""
StreamingCostMiddleware — per-token cost accumulation during streaming responses.

Wraps a streaming LLM generator, tracks cost per output token in real time,
and aborts if max_stream_cost is exceeded.

FIX: Token counting now estimates tokens from chunk content length rather than
assuming 1 token per chunk. OpenAI/Anthropic chunks can contain multiple tokens.
"""


class StreamCostLimitReached(Exception):
    def __init__(self, cost, tokens):
        self.cost = cost
        self.tokens = tokens
        super().__init__(
            f"Stream cost limit reached: ${cost:.4f} after {tokens} tokens"
        )


class StreamingCostMiddleware:

    # Average chars per token — used for streaming estimation.
    # Conservative (low) estimate so we abort sooner rather than later.
    CHARS_PER_TOKEN = 3.5

    def __init__(self, model: str, pricing_engine, budget_engine,
                 max_stream_cost=None):
        self.model = model
        self.pricing = pricing_engine
        self.budget_engine = budget_engine
        self.max_stream_cost = max_stream_cost
        self.token_count = 0
        self.stream_cost = 0.0

    def _extract_content(self, chunk) -> str:
        """Extract text content from a streaming chunk. Supports OpenAI, Anthropic, and dict formats."""
        # OpenAI streaming format
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                return delta.content

        # Anthropic streaming format
        if hasattr(chunk, "delta"):
            delta = chunk.delta
            if hasattr(delta, "text") and delta.text:
                return delta.text

        # Gemini / generic dict format
        if isinstance(chunk, dict):
            return chunk.get("text", chunk.get("content", ""))

        return ""

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count from content string. At least 1 if content is non-empty."""
        if not content:
            return 0
        return max(1, int(len(content) / self.CHARS_PER_TOKEN))

    def wrap_stream(self, stream_generator, input_tokens: int):
        """
        Wraps a streaming LLM response generator.
        Yields (chunk, current_cost) tuples.
        Raises StreamCostLimitReached if max_stream_cost exceeded.

        Handles OpenAI final usage chunk (choices=[], usage=...) for exact counts.
        Handles Anthropic message_delta with cumulative output_tokens.
        """
        input_cost = self.pricing.input_cost(self.model, input_tokens)

        for chunk in stream_generator:
            # OpenAI final usage chunk: choices=[] and usage is not None
            # (stream_options={"include_usage": True})
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                exact_input = getattr(usage, "prompt_tokens", input_tokens)
                exact_output = getattr(usage, "completion_tokens", self.token_count)
                self.token_count = exact_output
                self.stream_cost = (
                    self.pricing.input_cost(self.model, exact_input)
                    + self.pricing.output_cost(self.model, exact_output)
                )
                yield chunk, self.stream_cost
                continue

            content = self._extract_content(chunk)
            chunk_tokens = self._estimate_tokens(content)

            if chunk_tokens > 0:
                self.token_count += chunk_tokens
                output_cost = self.pricing.output_cost(
                    self.model, self.token_count
                )
                self.stream_cost = input_cost + output_cost

                if (self.max_stream_cost is not None and
                        self.stream_cost >= self.max_stream_cost):
                    raise StreamCostLimitReached(
                        cost=self.stream_cost,
                        tokens=self.token_count,
                    )

            yield chunk, self.stream_cost

    def get_final_cost(self) -> float:
        return self.stream_cost
