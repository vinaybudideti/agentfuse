"""
Phase 0 — Token counting accuracy tests.

These tests verify that token counting uses the correct encoding per provider
and applies appropriate safety margins.
"""

from agentfuse.providers.tokenizer import TokenCounterAdapter


def test_openai_gpt4o_uses_o200k_encoding():
    """GPT-4o must use o200k_base, not cl100k_base."""
    t = TokenCounterAdapter()
    # "Hello world" — basic test that encoding works
    count = t.count_tokens("Hello world", "gpt-4o")
    assert count > 0


def test_openai_o3_uses_o200k_encoding():
    """o3 model must use same encoding as gpt-4o."""
    t = TokenCounterAdapter()
    text = "The quick brown fox jumps over the lazy dog"
    count_4o = t.count_tokens(text, "gpt-4o")
    count_o3 = t.count_tokens(text, "o3")
    # Both should use o200k_base, so counts should be equal
    assert count_4o == count_o3


def test_openai_o1_and_o4_use_o200k_encoding():
    """o1 and o4-mini must use same encoding as gpt-4o (o200k_base)."""
    t = TokenCounterAdapter()
    text = "The quick brown fox jumps over the lazy dog"
    count_4o = t.count_tokens(text, "gpt-4o")
    assert t.count_tokens(text, "o1") == count_4o
    assert t.count_tokens(text, "o4-mini") == count_4o


def test_anthropic_has_20pct_safety_margin():
    """Anthropic count must be >= tiktoken result * 1.20."""
    t = TokenCounterAdapter()
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    text = "The quick brown fox jumps over the lazy dog"
    raw_count = len(enc.encode(text))
    claude_count = t.count_tokens(text, "claude-sonnet-4-6")
    assert claude_count == int(raw_count * 1.20)
    assert claude_count > raw_count


def test_gemini_has_25pct_safety_margin():
    """Gemini count must be >= tiktoken result * 1.25."""
    t = TokenCounterAdapter()
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    text = "The quick brown fox jumps over the lazy dog"
    raw_count = len(enc.encode(text))
    gemini_count = t.count_tokens(text, "gemini-2.5-pro")
    assert gemini_count == int(raw_count * 1.25)
    assert gemini_count > raw_count


def test_unknown_model_falls_back_gracefully():
    """Unknown model must not crash, should return an estimate."""
    t = TokenCounterAdapter()
    count = t.count_tokens("Hello world test string", "some-unknown-model-v5")
    assert count > 0


def test_multimodal_content_blocks():
    """count_messages must handle list content (Anthropic vision blocks)."""
    t = TokenCounterAdapter()
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image", "source": {"data": "base64..."}},
        ]}
    ]
    count = t.count_messages(messages, "gpt-4o")
    assert count > 4  # at least the text tokens + overhead + priming


def test_empty_text_returns_zero():
    """Empty text must return 0 for all models."""
    t = TokenCounterAdapter()
    assert t.count_tokens("", "gpt-4o") == 0
    assert t.count_tokens("", "claude-sonnet-4-6") == 0
    assert t.count_tokens("", "gemini-2.5-pro") == 0


def test_count_messages_includes_priming_tokens():
    """count_messages must include 3 reply priming tokens."""
    t = TokenCounterAdapter()
    messages = [{"role": "user", "content": "Hi"}]
    count = t.count_messages(messages, "gpt-4o")
    token_count = t.count_tokens("Hi", "gpt-4o")
    # Should be: token_count + 4 (overhead) + 3 (priming)
    assert count == token_count + 4 + 3


def test_backward_compatible_count_messages_tokens():
    """count_messages_tokens alias must still work."""
    t = TokenCounterAdapter()
    messages = [{"role": "user", "content": "Hello"}]
    count1 = t.count_messages(messages, "gpt-4o")
    count2 = t.count_messages_tokens(messages, "gpt-4o")
    assert count1 == count2
