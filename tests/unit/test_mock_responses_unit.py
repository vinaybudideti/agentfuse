"""
Loop 41 — Mock response object tests.
"""

from agentfuse.providers.mock_responses import MockOpenAIResponse, MockAnthropicResponse


def test_openai_mock_response():
    """MockOpenAIResponse must have all required fields."""
    resp = MockOpenAIResponse("Hello", "gpt-4o")
    assert resp.choices[0].message.content == "Hello"
    assert resp.model == "gpt-4o"
    assert resp.usage.prompt_tokens == 0
    assert resp.usage.completion_tokens == 0


def test_openai_mock_model_dump():
    """MockOpenAIResponse.model_dump must return a dict."""
    resp = MockOpenAIResponse("Test", "gpt-4o")
    dumped = resp.model_dump()
    assert isinstance(dumped, dict)
    assert "choices" in dumped
    assert dumped["model"] == "gpt-4o"


def test_anthropic_mock_response():
    """MockAnthropicResponse must have all required fields."""
    resp = MockAnthropicResponse("Hello", "claude-sonnet-4-6")
    assert resp.content[0].text == "Hello"
    assert resp.model == "claude-sonnet-4-6"
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0


def test_anthropic_mock_model_dump():
    """MockAnthropicResponse.model_dump must return a dict."""
    resp = MockAnthropicResponse("Test", "claude-sonnet-4-6")
    dumped = resp.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["model"] == "claude-sonnet-4-6"
