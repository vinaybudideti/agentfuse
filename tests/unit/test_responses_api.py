"""
Tests for OpenAI Responses API adapter.
"""

from types import SimpleNamespace
from agentfuse.providers.responses_api import (
    is_responses_api_model, normalize_response, build_responses_params,
)


def test_gpt54_is_responses_api():
    """GPT-5.4 must be detected as Responses API model."""
    assert is_responses_api_model("gpt-5.4") is True


def test_gpt53_codex_is_responses_api():
    """GPT-5.3-Codex must be detected as Responses API model."""
    assert is_responses_api_model("gpt-5.3-codex") is True


def test_gpt4o_is_not_responses_api():
    """GPT-4o must NOT be Responses API."""
    assert is_responses_api_model("gpt-4o") is False


def test_normalize_responses_format():
    """Responses API format must be normalized to Chat Completions."""
    raw = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="text", text="Hello from Responses API")],
            ),
        ],
        usage=SimpleNamespace(input_tokens=50, output_tokens=10),
        response_id="resp_123",
    )
    normalized = normalize_response(raw)
    assert normalized.choices[0].message.content == "Hello from Responses API"
    assert normalized.usage.prompt_tokens == 50
    assert normalized.usage.completion_tokens == 10


def test_normalize_passthrough_chat_completions():
    """Chat Completions format must pass through unchanged."""
    chat = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Already chat format"),
            finish_reason="stop",
        )],
    )
    result = normalize_response(chat)
    assert result.choices[0].message.content == "Already chat format"


def test_normalize_none():
    """None input must return None."""
    assert normalize_response(None) is None


def test_build_params_basic():
    """Basic params must be converted correctly."""
    params = build_responses_params(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
    )
    assert params["model"] == "gpt-5.4"
    assert params["max_output_tokens"] == 100
    assert "input" in params


def test_build_params_verbosity():
    """Verbosity must be passed through."""
    params = build_responses_params(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        verbosity="low",
    )
    assert params["verbosity"] == "low"


def test_build_params_tools():
    """Tools must be included in params."""
    tools = [{"function": {"name": "search"}}]
    params = build_responses_params(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "search for X"}],
        tools=tools,
    )
    assert params["tools"] == tools
