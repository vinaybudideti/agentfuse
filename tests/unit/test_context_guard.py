"""
Tests for ContextWindowGuard — context window overflow prevention.
"""

import pytest

from agentfuse.core.context_guard import ContextWindowGuard, ContextWindowOverflow


def test_short_messages_fit():
    """Short messages must fit any model."""
    guard = ContextWindowGuard()
    msgs = [{"role": "user", "content": "Hello"}]
    result = guard.check(msgs, "gpt-4o")
    assert result["fits"] is True


def test_ensure_fits_returns_original():
    """Messages that fit must be returned unchanged."""
    guard = ContextWindowGuard()
    msgs = [{"role": "user", "content": "Hello"}]
    result = guard.ensure_fits(msgs, "gpt-4o")
    assert result == msgs


def test_long_messages_detected():
    """Messages exceeding context window must be detected."""
    guard = ContextWindowGuard()
    # Use a small model with small context for faster testing
    # gpt-4o-mini has 128K context, need to exceed 124K (128K - 4K output)
    # Each word ≈ 1 token, so generate 130K words
    content = " ".join(f"word{i}" for i in range(130_000))
    msgs = [{"role": "user", "content": content}]
    result = guard.check(msgs, "gpt-4o-mini")
    assert result["fits"] is False


def test_ensure_fits_compresses():
    """Messages that don't fit must be compressed to fewer messages."""
    guard = ContextWindowGuard()
    # Create messages that exceed gpt-4o-mini context (128K)
    msgs = [{"role": "system", "content": "Be helpful."}]
    # Generate enough content to exceed 128K - 4K = 124K tokens
    # Each message: ~130 tokens (1 word ≈ 1 token)
    for i in range(1000):
        words = " ".join(f"word{j}" for j in range(130))
        msgs.append({"role": "user", "content": f"Q{i}: {words}"})

    # Should auto-compress to fit gpt-4o-mini
    result = guard.ensure_fits(msgs, "gpt-4o-mini", max_output_tokens=4096)
    assert len(result) < len(msgs)


def test_get_model_limits():
    """Must return context window limits for known models."""
    guard = ContextWindowGuard()
    limits = guard.get_model_limits("gpt-4o")
    assert limits["context_window"] == 128_000
    assert limits["max_output"] == 16_000


def test_unknown_model_passes():
    """Unknown model with 0 context must pass (can't check)."""
    guard = ContextWindowGuard()
    msgs = [{"role": "user", "content": "Hello"}]
    result = guard.check(msgs, "totally-unknown-model")
    assert result["fits"] is True


def test_headroom_calculation():
    """Headroom must be positive when messages fit."""
    guard = ContextWindowGuard()
    msgs = [{"role": "user", "content": "Hello"}]
    result = guard.check(msgs, "gpt-4o")
    assert result["headroom"] > 0
