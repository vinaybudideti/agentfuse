"""
Loop 6 — PromptCachingMiddleware tests.
"""

from agentfuse.core.prompt_cache import PromptCachingMiddleware


def test_inject_skips_non_claude():
    """Non-Claude models must get messages unchanged."""
    m = PromptCachingMiddleware()
    msgs = [{"role": "system", "content": "x" * 5000}]
    result = m.inject(msgs, "gpt-4o")
    assert result == msgs


def test_inject_adds_cache_control_to_large_static_system():
    """Large static system messages on Claude must get cache_control."""
    m = PromptCachingMiddleware()
    long_text = "You are a helpful assistant. " * 200  # > 1024 tokens
    msgs = [{"role": "system", "content": long_text}]
    result = m.inject(msgs, "claude-sonnet-4-6")
    # Should convert to content block with cache_control
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_inject_skips_small_system():
    """Small system messages must not get cache_control."""
    m = PromptCachingMiddleware()
    msgs = [{"role": "system", "content": "Be helpful"}]
    result = m.inject(msgs, "claude-sonnet-4-6")
    assert isinstance(result[0]["content"], str)


def test_inject_skips_dynamic_content():
    """Dynamic content (dates, session IDs) must not get cache_control."""
    m = PromptCachingMiddleware()
    dynamic_text = ("Today is 2026-03-16. " + "x " * 1000)
    msgs = [{"role": "system", "content": dynamic_text}]
    result = m.inject(msgs, "claude-sonnet-4-6")
    assert isinstance(result[0]["content"], str)


def test_inject_max_4_breakpoints():
    """At most 4 cache breakpoints allowed."""
    m = PromptCachingMiddleware()
    long_text = "Static system prompt. " * 200
    msgs = [{"role": "system", "content": long_text} for _ in range(6)]
    result = m.inject(msgs, "claude-sonnet-4-6")
    breakpoints = sum(
        1 for msg in result
        if isinstance(msg.get("content"), list)
    )
    assert breakpoints <= 4


def test_inject_does_not_mutate_original():
    """inject() must not mutate the original messages list."""
    m = PromptCachingMiddleware()
    long_text = "System prompt text. " * 200
    msgs = [{"role": "system", "content": long_text}]
    original_content = msgs[0]["content"]
    m.inject(msgs, "claude-sonnet-4-6")
    assert msgs[0]["content"] == original_content  # Original unchanged


def test_is_static_detects_timestamps():
    """Content with timestamps must be detected as dynamic."""
    m = PromptCachingMiddleware()
    assert not m._is_static("timestamp=1710523200 is current")
    assert not m._is_static("request_id: abc123xyz")
    assert m._is_static("You are a helpful assistant that answers questions.")


def test_user_messages_not_cached():
    """Only system messages should get cache_control, not user messages."""
    m = PromptCachingMiddleware()
    long_text = "A very long user message. " * 200
    msgs = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": long_text},
    ]
    result = m.inject(msgs, "claude-sonnet-4-6")
    # User message should NOT get cache_control
    assert isinstance(result[1]["content"], str)
