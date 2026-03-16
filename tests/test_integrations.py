"""
Integration tests — verify actual behavior, not just construction.
Updated for v0.2.0 API (BaseChatModel wrapper, new hook signatures).
"""

import pytest
from types import SimpleNamespace


# --- LangChain ---

def test_langchain_middleware_init():
    from agentfuse.integrations.langchain import AgentFuseChatModel
    m = AgentFuseChatModel(budget=5.00, model="gpt-4o")
    assert m.engine.budget == 5.00
    assert m.engine.model == "gpt-4o"
    assert m.run_id is not None


def test_langchain_middleware_receipt():
    from agentfuse.integrations.langchain import AgentFuseChatModel
    m = AgentFuseChatModel(budget=5.00)
    receipt = m.get_receipt()
    assert receipt["run_id"] == m.run_id
    assert receipt["spent_usd"] == 0.0
    assert receipt["budget_usd"] == 5.00


def test_langchain_extract_text_openai_format():
    """_extract_text must handle OpenAI response format."""
    from agentfuse.integrations.langchain import _extract_text
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello"))]
    )
    assert _extract_text(response) == "Hello"


def test_langchain_extract_text_anthropic_format():
    """_extract_text must handle Anthropic response format."""
    from agentfuse.integrations.langchain import _extract_text
    response = SimpleNamespace(
        content=[SimpleNamespace(text="Hello from Claude")]
    )
    assert _extract_text(response) == "Hello from Claude"


# --- CrewAI ---

def test_crewai_hooks_return_callables():
    from agentfuse.integrations.crewai import create_agentfuse_hooks
    before, after = create_agentfuse_hooks(budget=5.00)
    assert callable(before)
    assert callable(after)


def test_crewai_before_allows_on_cache_miss():
    """Before hook returns None (allow) on cache miss."""
    from agentfuse.integrations.crewai import create_agentfuse_hooks
    before, after = create_agentfuse_hooks(budget=10.00)
    context = SimpleNamespace(
        messages=[{"role": "user", "content": "unique prompt xyz 12345"}],
        model="gpt-4o"
    )
    result = before(context)
    assert result is not False, "Expected allow (None) on cache miss"


def test_crewai_after_hook_single_arg():
    """After hook takes single context arg and returns None on first call."""
    from agentfuse.integrations.crewai import create_agentfuse_hooks
    _, after = create_agentfuse_hooks(budget=5.00, model="gpt-4o")
    context = SimpleNamespace(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-4o",
        response=None,
    )
    result = after(context)
    assert result is None  # No cached response to return


# --- OpenAI Agents SDK ---

def test_openai_agents_hooks_init():
    from agentfuse.integrations.openai_agents import AgentFuseRunHooks
    h = AgentFuseRunHooks(budget=5.00, model="gpt-4o")
    assert h.engine.budget == 5.00
    assert h.run_id is not None


def test_openai_agents_hooks_receipt():
    from agentfuse.integrations.openai_agents import AgentFuseRunHooks
    h = AgentFuseRunHooks(budget=3.00)
    receipt = h.get_receipt()
    assert receipt["budget_usd"] == 3.00
    assert receipt["spent_usd"] == 0.0


def test_openai_agents_model_provider():
    """AgentFuseModelProvider must create model instances."""
    from agentfuse.integrations.openai_agents import AgentFuseModelProvider
    provider = AgentFuseModelProvider(budget=5.00)
    model = provider.get_model("gpt-4o")
    assert model is not None
    assert model.engine.budget == 5.00


def test_openai_agents_budget_exhaustion():
    """BudgetEngine must raise BudgetExhaustedGracefully when budget is fully spent."""
    from agentfuse.core.budget import BudgetEngine, BudgetExhaustedGracefully

    engine = BudgetEngine("test_exhaust", budget_usd=0.001, model="gpt-4o")
    engine.spent = 0.0009
    messages = [{"role": "user", "content": "hello"}]

    with pytest.raises(BudgetExhaustedGracefully):
        engine.check_and_act(0.002, messages)
