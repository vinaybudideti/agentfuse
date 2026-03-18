"""Tests for Pydantic AI integration."""

from types import SimpleNamespace
from agentfuse.integrations.pydantic_ai import wrap_pydantic_agent, get_agent_receipt


def test_wrap_agent_adds_engine():
    """Wrapping must attach a BudgetEngine."""
    agent = SimpleNamespace(run=None, run_sync=None)
    wrapped = wrap_pydantic_agent(agent, budget_usd=5.0)
    assert hasattr(wrapped, "_agentfuse_engine")
    assert wrapped._agentfuse_engine.budget == 5.0


def test_get_receipt_from_wrapped():
    """get_agent_receipt must return budget info."""
    agent = SimpleNamespace(run=None, run_sync=None)
    wrapped = wrap_pydantic_agent(agent, budget_usd=3.0, run_id="pydantic_test")
    receipt = get_agent_receipt(wrapped)
    assert receipt["budget_usd"] == 3.0
    assert receipt["run_id"] == "pydantic_test"
    assert receipt["state"] == "normal"


def test_get_receipt_unwrapped():
    """get_agent_receipt on unwrapped agent must return error."""
    agent = SimpleNamespace()
    receipt = get_agent_receipt(agent)
    assert "error" in receipt


def test_custom_model():
    """Wrapper must use specified model."""
    agent = SimpleNamespace(run=None, run_sync=None)
    wrapped = wrap_pydantic_agent(agent, model="claude-sonnet-4-6")
    assert wrapped._agentfuse_engine.model == "claude-sonnet-4-6"
