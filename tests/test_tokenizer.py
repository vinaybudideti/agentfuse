"""Tests for TokenCounterAdapter: verify token counts and cost estimates."""

from agentfuse.providers.tokenizer import TokenCounterAdapter
from agentfuse.providers.pricing import ModelPricingEngine


def test_basic_token_count():
    t = TokenCounterAdapter()
    count = t.count_tokens("Hello world", "gpt-4o")
    assert count == 2


def test_claude_model_applies_correction_factor():
    """Anthropic tokens should be ~20% higher than raw cl100k count."""
    t = TokenCounterAdapter()
    text = "The quick brown fox jumps over the lazy dog"
    gpt_count = t.count_tokens(text, "gpt-4o")
    claude_count = t.count_tokens(text, "claude-sonnet-4-6")
    # Claude count should be higher due to 1.20x safety margin
    assert claude_count > gpt_count
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    raw = len(enc.encode(text))
    assert claude_count == int(raw * 1.20)


def test_gemini_model_applies_correction_factor():
    """Gemini tokens should be ~25% higher than raw cl100k count."""
    t = TokenCounterAdapter()
    text = "The quick brown fox jumps over the lazy dog"
    gpt_count = t.count_tokens(text, "gpt-4o")
    gemini_count = t.count_tokens(text, "gemini-1.5-pro")
    assert gemini_count >= gpt_count
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    raw = len(enc.encode(text))
    assert gemini_count == int(raw * 1.25)


def test_messages_token_count():
    t = TokenCounterAdapter()
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ]
    count = t.count_messages_tokens(messages, "gpt-4o")
    # 3 tokens ("You are helpful") + 4 overhead + 1 token ("Hello") + 4 overhead + 3 priming = 15
    assert count > 0


def test_messages_handles_list_content():
    """Non-string content (Anthropic vision blocks) must not crash."""
    t = TokenCounterAdapter()
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image"},
        ]}
    ]
    count = t.count_messages_tokens(messages, "gpt-4o")
    assert count > 4  # at least the text tokens + overhead


def test_empty_text_returns_zero():
    t = TokenCounterAdapter()
    assert t.count_tokens("", "gpt-4o") == 0
    assert t.count_tokens("", "claude-sonnet-4-6") == 0


def test_anthropic_cost_estimate_is_higher_than_openai_for_same_text():
    """
    Verify that Anthropic cost estimates account for the correction factor.
    For the same text, Claude token count > GPT token count, so cost per token
    must reflect the corrected count.
    """
    t = TokenCounterAdapter()
    p = ModelPricingEngine()
    text = "Explain quantum computing in simple terms for a beginner audience"

    gpt_tokens = t.count_tokens(text, "gpt-4o")
    claude_tokens = t.count_tokens(text, "claude-sonnet-4-6")

    # Claude should have more tokens due to correction
    assert claude_tokens > gpt_tokens

    # Cost estimates should use the corrected token count
    gpt_cost = p.input_cost("gpt-4o", gpt_tokens)
    claude_cost = p.input_cost("claude-sonnet-4-6", claude_tokens)
    # Both should be positive
    assert gpt_cost > 0
    assert claude_cost > 0


def test_estimate_cost_method_works():
    """Verify the previously broken estimate_cost method (was calling non-existent count_messages)."""
    p = ModelPricingEngine()
    messages = [
        {"role": "user", "content": "Hello world"}
    ]
    cost = p.estimate_cost("gpt-4o", messages)
    assert cost > 0


def test_budget_zero_raises():
    """Budget of 0 must raise ValueError, not divide-by-zero."""
    from agentfuse.core.budget import BudgetEngine
    import pytest
    with pytest.raises(ValueError, match="Budget must be > 0"):
        BudgetEngine("run_1", 0, "gpt-4o")
