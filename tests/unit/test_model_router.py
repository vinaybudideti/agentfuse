"""
Tests for IntelligentModelRouter — RouteLLM-inspired complexity routing.
"""

from agentfuse.core.model_router import IntelligentModelRouter


def test_simple_query_downrouted():
    """Simple factual queries must route to cheap model."""
    router = IntelligentModelRouter(threshold=0.4)
    msgs = [{"role": "user", "content": "What is Python?"}]
    result = router.route("gpt-4o", msgs)
    assert result == "gpt-4o-mini"


def test_complex_query_keeps_strong():
    """Complex reasoning queries must keep the strong model."""
    router = IntelligentModelRouter(threshold=0.4)
    msgs = [{"role": "user", "content": "Analyze the time complexity of this algorithm step by step and explain why quicksort has O(n log n) average case. Write a detailed implementation with edge case handling."}]
    result = router.route("gpt-4o", msgs)
    assert result == "gpt-4o"


def test_code_query_keeps_strong():
    """Code generation queries must keep the strong model."""
    router = IntelligentModelRouter(threshold=0.4)
    msgs = [{"role": "user", "content": "Implement a binary search tree class with insert, delete, and search methods in Python."}]
    result = router.route("gpt-4o", msgs)
    assert result == "gpt-4o"


def test_greeting_downrouted():
    """Greetings must route to cheap model."""
    router = IntelligentModelRouter(threshold=0.4)
    msgs = [{"role": "user", "content": "Hello!"}]
    result = router.route("gpt-4o", msgs)
    assert result == "gpt-4o-mini"


def test_unknown_model_unchanged():
    """Models without routing pairs must pass through unchanged."""
    router = IntelligentModelRouter()
    msgs = [{"role": "user", "content": "hello"}]
    result = router.route("custom-model-v1", msgs)
    assert result == "custom-model-v1"


def test_anthropic_routing():
    """Anthropic models must route correctly."""
    router = IntelligentModelRouter(threshold=0.4)
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    result = router.route("claude-sonnet-4-6", msgs)
    assert result == "claude-haiku-4-5-20251001"


def test_stats_tracking():
    router = IntelligentModelRouter(threshold=0.4)
    router.route("gpt-4o", [{"role": "user", "content": "Hi"}])
    router.route("gpt-4o", [{"role": "user", "content": "Analyze complex algorithm"}])
    stats = router.get_stats()
    assert stats["total_routed"] == 2
    assert stats["downrouted"] >= 1


def test_multi_turn_complexity():
    """Long multi-turn conversations should increase complexity score."""
    router = IntelligentModelRouter(threshold=0.5)
    msgs = [
        {"role": "system", "content": "You are a senior software architect. " * 20},
        {"role": "user", "content": "Design the database schema"},
        {"role": "assistant", "content": "Here's the schema..."},
        {"role": "user", "content": "Now explain the tradeoffs"},
        {"role": "assistant", "content": "The tradeoffs are..."},
        {"role": "user", "content": "How would you handle sharding?"},
        {"role": "assistant", "content": "For sharding..."},
        {"role": "user", "content": "Compare this approach with event sourcing and analyze the consistency guarantees"},
    ]
    result = router.route("gpt-4o", msgs)
    assert result == "gpt-4o"  # complex multi-turn stays on strong model


def test_custom_pairs():
    """Custom model pairs must override defaults."""
    router = IntelligentModelRouter(threshold=0.3, custom_pairs={
        "my-expensive": "my-cheap",
    })
    msgs = [{"role": "user", "content": "hi"}]
    result = router.route("my-expensive", msgs)
    assert result == "my-cheap"
