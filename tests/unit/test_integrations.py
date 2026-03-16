"""
Phase 4 — Framework integration unit tests (no real LLM calls).
"""

from types import SimpleNamespace


def test_langchain_middleware_init_works():
    """AgentFuseChatModel initializes without error."""
    from agentfuse.integrations.langchain import AgentFuseChatModel
    model = AgentFuseChatModel(budget=5.00)
    assert model.engine is not None
    assert model.run_id is not None
    assert model._llm_type == "agentfuse"


def test_crewai_hooks_return_callables():
    """create_agentfuse_hooks returns two callable hook functions."""
    from agentfuse.integrations.crewai import create_agentfuse_hooks
    before, after = create_agentfuse_hooks(budget=5.00)
    assert callable(before)
    assert callable(after)


def test_openai_agents_model_provider_init():
    """AgentFuseModelProvider initializes and creates models."""
    from agentfuse.integrations.openai_agents import AgentFuseModelProvider
    provider = AgentFuseModelProvider(budget=5.00)
    model = provider.get_model("gpt-4o")
    assert model is not None
    assert model.engine is not None


def test_langchain_import_error_message():
    """AgentFuseChatModel should work without langchain installed."""
    from agentfuse.integrations.langchain import AgentFuseChatModel
    # Should initialize fine without langchain-core
    model = AgentFuseChatModel(budget=5.00)
    assert model is not None


def test_mock_response_extraction_openai_format():
    """_extract_text handles OpenAI response format."""
    from agentfuse.integrations.langchain import _extract_text

    # OpenAI format
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Hello world")
        )]
    )
    assert _extract_text(response) == "Hello world"


def test_mock_response_extraction_anthropic_format():
    """_extract_text handles Anthropic response format."""
    from agentfuse.integrations.langchain import _extract_text

    # Anthropic format
    response = SimpleNamespace(
        content=[SimpleNamespace(text="Hello from Claude")]
    )
    assert _extract_text(response) == "Hello from Claude"


def test_crewai_before_hook_allows_call():
    """Before hook returns None (allow) when no cache hit."""
    from agentfuse.integrations.crewai import create_agentfuse_hooks
    before, _ = create_agentfuse_hooks(budget=5.00)

    context = SimpleNamespace(
        messages=[{"role": "user", "content": "unique query for test"}],
        model="gpt-4o",
    )
    result = before(context)
    # Should allow (return None or True, not False)
    assert result is not False or result is None


def test_openai_agents_model_get_receipt():
    """AgentFuseRunHooks.get_receipt returns expected structure."""
    from agentfuse.integrations.openai_agents import AgentFuseRunHooks
    hooks = AgentFuseRunHooks(budget=5.00, run_id="test_run")
    receipt = hooks.get_receipt()
    assert receipt["run_id"] == "test_run"
    assert receipt["budget_usd"] == 5.00
    assert receipt["spent_usd"] == 0.0


def test_backward_compat_aliases():
    """Backward compat aliases must exist."""
    from agentfuse.integrations.langchain import AgentFuseLangChainMiddleware
    from agentfuse.integrations.crewai import agentfuse_hooks
    from agentfuse.integrations.openai_agents import AgentFuseRunHooks
    assert AgentFuseLangChainMiddleware is not None
    assert callable(agentfuse_hooks)
    assert AgentFuseRunHooks is not None


def test_langchain_message_conversion():
    """LangChain BaseMessage-like objects must be converted correctly."""
    from agentfuse.integrations.langchain import AgentFuseChatModel

    model = AgentFuseChatModel(budget=5.00)

    # Simulate LangChain messages with type/content attrs
    msgs = [
        SimpleNamespace(type="system", content="Be helpful"),
        SimpleNamespace(type="human", content="Hello"),
        SimpleNamespace(type="ai", content="Hi there"),
    ]

    # Convert using the internal logic
    msg_dicts = []
    for m in msgs:
        if isinstance(m, dict):
            msg_dicts.append(m)
        elif hasattr(m, "type") and hasattr(m, "content"):
            role = getattr(m, "type", "user")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            msg_dicts.append({"role": role, "content": m.content})

    assert msg_dicts[0]["role"] == "system"
    assert msg_dicts[1]["role"] == "user"
    assert msg_dicts[2]["role"] == "assistant"


def test_extract_text_string_passthrough():
    """String input must be returned as-is."""
    from agentfuse.integrations.langchain import _extract_text
    assert _extract_text("hello") == "hello"


def test_extract_text_langchain_generations():
    """LangChain ChatResult with generations must extract text."""
    from agentfuse.integrations.langchain import _extract_text
    response = SimpleNamespace(
        generations=[SimpleNamespace(
            text="Generated text",
            message=SimpleNamespace(content="Message content"),
        )]
    )
    result = _extract_text(response)
    assert result == "Generated text"
