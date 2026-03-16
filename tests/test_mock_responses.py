"""Tests for mock response objects — verify they have all fields downstream code expects."""

import json
from agentfuse.providers.mock_responses import MockOpenAIResponse, MockAnthropicResponse


def test_openai_mock_has_required_fields():
    """LangChain and user code accesses these fields."""
    r = MockOpenAIResponse("Hello", "gpt-4o")

    # Fields that real openai.types.chat.ChatCompletion has
    assert r.id == "agentfuse_cache_hit"
    assert r.model == "gpt-4o"
    assert r.object == "chat.completion"
    assert r.created > 0

    # Choices
    assert len(r.choices) == 1
    choice = r.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.index == 0
    assert choice.message.content == "Hello"
    assert choice.message.role == "assistant"
    assert choice.message.tool_calls is None  # LangChain checks this
    assert choice.message.function_call is None

    # Usage
    assert r.usage.prompt_tokens == 0
    assert r.usage.completion_tokens == 0
    assert r.usage.total_tokens == 0


def test_openai_mock_model_dump():
    """LangChain calls .model_dump() or .dict() on responses."""
    r = MockOpenAIResponse("Hello", "gpt-4o")

    d = r.model_dump()
    assert isinstance(d, dict)
    assert d["model"] == "gpt-4o"
    assert d["choices"][0]["message"]["content"] == "Hello"
    assert d["usage"]["prompt_tokens"] == 0

    # .dict() should also work (older pydantic compat)
    assert r.dict() == d

    # .json() should produce valid JSON
    j = r.json()
    parsed = json.loads(j)
    assert parsed["model"] == "gpt-4o"


def test_anthropic_mock_has_required_fields():
    r = MockAnthropicResponse("Hello", "claude-sonnet-4-6")

    assert r.id == "agentfuse_cache_hit"
    assert r.type == "message"
    assert r.role == "assistant"
    assert r.model == "claude-sonnet-4-6"
    assert r.stop_reason == "end_turn"
    assert r.stop_sequence is None

    # Content blocks
    assert len(r.content) == 1
    assert r.content[0].type == "text"
    assert r.content[0].text == "Hello"

    # Usage
    assert r.usage.input_tokens == 0
    assert r.usage.output_tokens == 0


def test_anthropic_mock_model_dump():
    r = MockAnthropicResponse("Hello", "claude-sonnet-4-6")

    d = r.model_dump()
    assert d["content"][0]["text"] == "Hello"
    assert d["usage"]["input_tokens"] == 0

    j = r.json()
    parsed = json.loads(j)
    assert parsed["model"] == "claude-sonnet-4-6"
