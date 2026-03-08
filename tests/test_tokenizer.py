# Tests for TokenCounterAdapter: verify tiktoken counts match official counts for test prompts

from agentfuse.providers.tokenizer import TokenCounterAdapter


def test_basic_token_count():
    t = TokenCounterAdapter()
    count = t.count_tokens("Hello world", "gpt-4o")
    assert count == 2


def test_claude_model_uses_approximation():
    t = TokenCounterAdapter()
    count = t.count_tokens("Hello world", "claude-sonnet-4-6")
    assert count > 0


def test_messages_token_count():
    t = TokenCounterAdapter()
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ]
    count = t.count_messages_tokens(messages, "gpt-4o")
    assert count > 0
