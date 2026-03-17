"""
Tests for provider wrappers with mocked SDK clients.
"""

from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import pytest
import os


def test_openai_context_tracking():
    """OpenAI _run_contexts and _active_openai_run must be module-level."""
    from agentfuse.providers.openai import _run_contexts, _active_openai_run
    assert isinstance(_run_contexts, dict)


def test_anthropic_context_tracking():
    """Anthropic _run_contexts must be module-level dict."""
    from agentfuse.providers.anthropic import _run_contexts
    assert isinstance(_run_contexts, dict)


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-fake-key-123"})
def test_wrap_openai_with_mock_env():
    """wrap_openai with env key must register context."""
    from agentfuse.providers.openai import wrap_openai, _run_contexts
    saved = dict(_run_contexts)
    try:
        wrap_openai(budget_usd=5.0, run_id="env_test")
        assert "env_test" in _run_contexts
        assert _run_contexts["env_test"]["engine"].budget == 5.0
    finally:
        _run_contexts.clear()
        _run_contexts.update(saved)


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-fake-key-123"})
def test_wrap_anthropic_with_mock_env():
    """wrap_anthropic with env key must register context."""
    from agentfuse.providers.anthropic import wrap_anthropic, _run_contexts
    saved = dict(_run_contexts)
    try:
        wrap_anthropic(budget_usd=3.0, run_id="env_anthro")
        assert "env_anthro" in _run_contexts
        assert _run_contexts["env_anthro"]["engine"].budget == 3.0
    finally:
        _run_contexts.clear()
        _run_contexts.update(saved)


def test_mock_responses_openai():
    """MockOpenAIResponse must create valid response objects."""
    from agentfuse.providers.mock_responses import MockOpenAIResponse
    mock = MockOpenAIResponse("test content", "gpt-4o")
    assert mock.choices[0].message.content == "test content"
    assert mock._agentfuse_cache_hit is True


def test_mock_responses_anthropic():
    """MockAnthropicResponse must create valid response objects."""
    from agentfuse.providers.mock_responses import MockAnthropicResponse
    mock = MockAnthropicResponse("claude content", "claude-sonnet-4-6")
    assert mock.content[0].text == "claude content"
    assert mock._agentfuse_cache_hit is True
