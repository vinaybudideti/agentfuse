"""
Tests for LangGraph integration.
"""

from agentfuse.integrations.langgraph import AgentFuseNode, budget_guard, cost_tracking_callback
from agentfuse.gateway import cleanup, _get_engine


def setup_function():
    cleanup()


def test_node_wraps_function():
    """AgentFuseNode must wrap and call the node function."""
    def my_node(state):
        return {"result": "done"}

    node = AgentFuseNode(my_node, budget_id="lg_test", budget_usd=5.0)
    result = node({"input": "test"})
    assert result["result"] == "done"
    assert node._call_count == 1


def test_node_tracks_calls():
    """Multiple calls must be tracked."""
    node = AgentFuseNode(lambda s: s, budget_id="lg_count")
    node({})
    node({})
    node({})
    assert node.get_stats()["call_count"] == 3


def test_budget_guard_continue():
    """Budget guard must return 'continue' when budget OK."""
    _get_engine("lg_guard", 10.0, "gpt-4o")
    guard = budget_guard("lg_guard")
    assert guard({}) == "continue"


def test_budget_guard_stop():
    """Budget guard must return 'stop' when budget exceeded."""
    engine = _get_engine("lg_stop", 1.0, "gpt-4o")
    engine.record_cost(0.95)  # 95% spent
    guard = budget_guard("lg_stop", threshold=0.90)
    assert guard({}) == "stop"


def test_budget_guard_unknown_run():
    """Budget guard for unknown run must return 'continue'."""
    guard = budget_guard("nonexistent")
    assert guard({}) == "continue"


def test_callback_creation():
    """cost_tracking_callback must return dict with handlers."""
    cb = cost_tracking_callback("lg_cb")
    assert "on_llm_start" in cb
    assert "on_llm_end" in cb
    assert callable(cb["on_llm_start"])
    assert callable(cb["on_llm_end"])


def test_callback_handlers_dont_crash():
    """Callback handlers must not crash."""
    cb = cost_tracking_callback("lg_cb2")
    cb["on_llm_start"]({})
    cb["on_llm_end"]({"usage": {}})
    cb["on_llm_end"]({"usage": {"total_cost": 0.05}})
