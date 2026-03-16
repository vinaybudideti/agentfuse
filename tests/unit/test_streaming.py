"""
Loop 6 — StreamingCostMiddleware tests.
"""

import pytest
from types import SimpleNamespace
from agentfuse.core.streaming import StreamingCostMiddleware, StreamCostLimitReached
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.core.budget import BudgetEngine


def _openai_chunk(content=None):
    """Create a mock OpenAI streaming chunk."""
    if content:
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
        )
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])


def test_wrap_stream_tracks_cost():
    """Cost must increase with each content chunk."""
    pricing = ModelPricingEngine()
    budget = BudgetEngine("stream_test", 10.0, "gpt-4o")
    middleware = StreamingCostMiddleware("gpt-4o", pricing, budget)

    chunks = [_openai_chunk("Hello"), _openai_chunk(" world"), _openai_chunk("!")]

    costs = []
    for chunk, cost in middleware.wrap_stream(iter(chunks), input_tokens=10):
        costs.append(cost)

    assert len(costs) == 3
    assert costs[-1] > 0
    assert middleware.token_count == 3


def test_stream_aborts_on_max_cost():
    """Stream must raise StreamCostLimitReached when max_stream_cost exceeded."""
    pricing = ModelPricingEngine()
    budget = BudgetEngine("stream_abort", 10.0, "gpt-4o")
    middleware = StreamingCostMiddleware("gpt-4o", pricing, budget, max_stream_cost=0.0001)

    # Generate many chunks to exceed the tiny max cost
    chunks = [_openai_chunk(f"token {i}") for i in range(1000)]

    with pytest.raises(StreamCostLimitReached) as exc_info:
        for _ in middleware.wrap_stream(iter(chunks), input_tokens=100):
            pass

    assert exc_info.value.cost > 0
    assert exc_info.value.tokens > 0


def test_empty_content_chunks_dont_count():
    """Chunks without content must not increment token count."""
    pricing = ModelPricingEngine()
    budget = BudgetEngine("stream_empty", 10.0, "gpt-4o")
    middleware = StreamingCostMiddleware("gpt-4o", pricing, budget)

    chunks = [_openai_chunk(None), _openai_chunk(None), _openai_chunk("Hello")]

    result_chunks = list(middleware.wrap_stream(iter(chunks), input_tokens=10))
    assert len(result_chunks) == 3
    assert middleware.token_count == 1


def test_get_final_cost():
    """get_final_cost must return the accumulated stream cost."""
    pricing = ModelPricingEngine()
    budget = BudgetEngine("stream_final", 10.0, "gpt-4o")
    middleware = StreamingCostMiddleware("gpt-4o", pricing, budget)

    chunks = [_openai_chunk("Hi")]
    list(middleware.wrap_stream(iter(chunks), input_tokens=10))
    assert middleware.get_final_cost() > 0


def test_anthropic_stream_format():
    """Anthropic streaming chunks with delta.text must be tracked."""
    pricing = ModelPricingEngine()
    budget = BudgetEngine("stream_anthropic", 10.0, "claude-sonnet-4-6")
    middleware = StreamingCostMiddleware("claude-sonnet-4-6", pricing, budget)

    # Anthropic format: chunk.delta.text
    chunks = [
        SimpleNamespace(delta=SimpleNamespace(text="Hello")),
        SimpleNamespace(delta=SimpleNamespace(text=" world")),
        SimpleNamespace(delta=SimpleNamespace(text=None)),  # end marker
    ]

    costs = []
    for _, cost in middleware.wrap_stream(iter(chunks), input_tokens=10):
        costs.append(cost)

    assert middleware.token_count == 2  # "Hello" and " world", not None


def test_no_max_stream_cost_allows_all():
    """Without max_stream_cost, all chunks must pass through."""
    pricing = ModelPricingEngine()
    budget = BudgetEngine("stream_no_max", 10.0, "gpt-4o")
    middleware = StreamingCostMiddleware("gpt-4o", pricing, budget)

    chunks = [_openai_chunk(f"token {i}") for i in range(100)]
    result = list(middleware.wrap_stream(iter(chunks), input_tokens=10))
    assert len(result) == 100
    assert middleware.token_count == 100
