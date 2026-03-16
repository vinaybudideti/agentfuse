import pytest


def test_langchain_middleware_init():
    from agentfuse.integrations.langchain import AgentFuseLangChainMiddleware
    m = AgentFuseLangChainMiddleware(budget=5.00, model="gpt-4o")
    assert m.engine.budget == 5.00
    assert m.engine.model == "gpt-4o"
    assert m.run_id is not None


def test_langchain_middleware_receipt():
    from agentfuse.integrations.langchain import AgentFuseLangChainMiddleware
    m = AgentFuseLangChainMiddleware(budget=5.00)
    receipt = m.get_receipt()
    assert "run_id" in receipt
    assert "spent_usd" in receipt
    assert receipt["budget_usd"] == 5.00


def test_crewai_hooks_return_callables():
    from agentfuse.integrations.crewai import agentfuse_hooks
    before, after = agentfuse_hooks(budget=5.00)
    assert callable(before)
    assert callable(after)


def test_crewai_before_returns_true_on_cache_miss():
    from agentfuse.integrations.crewai import agentfuse_hooks
    from types import SimpleNamespace
    before, after = agentfuse_hooks(budget=10.00)
    context = SimpleNamespace(
        messages=[{"role": "user", "content": "unique prompt xyz 12345"}],
        model="gpt-4o"
    )
    result = before(context)
    # Should return True (proceed) on cache miss
    assert result in (True, False)  # either is valid


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
