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
    assert middleware.token_count >= 3  # at least 1 token per content chunk


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
    assert middleware.token_count >= 1


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

    assert middleware.token_count >= 2  # "Hello" and " world" have content, None doesn't


def test_no_max_stream_cost_allows_all():
    """Without max_stream_cost, all chunks must pass through."""
    pricing = ModelPricingEngine()
    budget = BudgetEngine("stream_no_max", 10.0, "gpt-4o")
    middleware = StreamingCostMiddleware("gpt-4o", pricing, budget)

    chunks = [_openai_chunk(f"token {i}") for i in range(100)]
    result = list(middleware.wrap_stream(iter(chunks), input_tokens=10))
    assert len(result) == 100
    assert middleware.token_count > 0  # estimated tokens from content


def test_gemini_dict_format():
    """Gemini dict-format chunks must be extracted."""
    pricing = ModelPricingEngine()
    budget = BudgetEngine("stream_gemini", 10.0, "gemini-2.5-pro")
    middleware = StreamingCostMiddleware("gemini-2.5-pro", pricing, budget)

    chunks = [
        {"text": "Hello from Gemini"},
        {"content": "More content"},
        {"text": ""},  # empty
    ]
    costs = []
    for _, cost in middleware.wrap_stream(iter(chunks), input_tokens=10):
        costs.append(cost)

    assert middleware.token_count >= 2


def test_openai_final_usage_chunk():
    """OpenAI final chunk with usage must set exact token counts."""
    pricing = ModelPricingEngine()
    budget = BudgetEngine("stream_usage", 10.0, "gpt-4o")
    middleware = StreamingCostMiddleware("gpt-4o", pricing, budget)

    chunks = [
        _openai_chunk("Hello"),
        _openai_chunk(" world"),
        # Final usage chunk (choices=[], usage populated)
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=100),
        ),
    ]
    results = list(middleware.wrap_stream(iter(chunks), input_tokens=50))
    assert len(results) == 3
    # After usage chunk, token count should be exact
    assert middleware.token_count == 100
    assert middleware.get_final_cost() > 0


def test_estimate_tokens_minimum():
    """Token estimation must return at least 1 for non-empty content."""
    pricing = ModelPricingEngine()
    budget = BudgetEngine("stream_min", 10.0, "gpt-4o")
    middleware = StreamingCostMiddleware("gpt-4o", pricing, budget)
    assert middleware._estimate_tokens("a") == 1
    assert middleware._estimate_tokens("") == 0


def test_stream_with_cache():
    """Streaming with cache must accumulate and store full response."""
    from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit

    pricing = ModelPricingEngine()
    budget = BudgetEngine("stream_cache", 10.0, "gpt-4o")
    middleware = StreamingCostMiddleware("gpt-4o", pricing, budget)
    cache = TwoTierCacheMiddleware()

    chunks = [_openai_chunk("Hello"), _openai_chunk(" from"), _openai_chunk(" cache")]
    msgs = [{"role": "user", "content": "stream cache test unique xyz789"}]

    # Stream with caching
    list(middleware.wrap_stream_with_cache(
        iter(chunks), input_tokens=10,
        model="gpt-4o", messages=msgs, cache=cache,
    ))

    # Verify cached
    result = cache.lookup("gpt-4o", msgs)
    assert isinstance(result, CacheHit)
    assert result.response == "Hello from cache"
