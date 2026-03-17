"""
Tests for AgentSession — all-in-one session context manager.
"""

from agentfuse.core.session import AgentSession
from agentfuse.gateway import cleanup, _cache


def setup_function():
    cleanup()


def test_session_context_manager():
    """Session must work as a context manager."""
    with AgentSession("test", budget_usd=5.0) as session:
        assert session.run_id is not None
        assert session._started_at is not None
    assert session._ended_at is not None


def test_session_completion_cache_hit():
    """Session completion must return cached responses."""
    msgs = [{"role": "user", "content": "session cache test unique 8765"}]
    _cache.store(model="gpt-4o", messages=msgs, response="Session cached response")

    with AgentSession("cache_test", budget_usd=5.0) as session:
        result = session.completion(messages=msgs)
        assert result.choices[0].message.content == "Session cached response"
        assert session._call_count == 1


def test_session_tool_call_tracking():
    """Session must track tool call costs."""
    with AgentSession("tool_test", budget_usd=5.0) as session:
        session.record_tool_call("web_search", cost=0.01)
        session.record_tool_call("web_search", cost=0.01)

    receipt = session.get_receipt()
    assert receipt["tool_cost_usd"] == 0.02
    assert receipt["tool_report"]["total_tool_calls"] == 2


def test_session_receipt_structure():
    """Receipt must contain all expected keys."""
    with AgentSession("receipt_test", budget_usd=5.0) as session:
        pass  # empty session

    receipt = session.get_receipt()
    assert "run_id" in receipt
    assert "name" in receipt
    assert "budget_usd" in receipt
    assert "total_cost_usd" in receipt
    assert "llm_cost_usd" in receipt
    assert "tool_cost_usd" in receipt
    assert "calls" in receipt
    assert "cache_hits" in receipt
    assert "duration_seconds" in receipt
    assert "cost_projection" in receipt
    assert "tool_report" in receipt
    assert receipt["budget_usd"] == 5.0


def test_session_estimate_remaining():
    """estimate_remaining must return projection data."""
    with AgentSession("est_test", budget_usd=10.0) as session:
        proj = session.estimate_remaining()
        assert "projected_total_cost" in proj
        assert "pattern" in proj


def test_session_custom_run_id():
    """Custom run_id must be preserved."""
    with AgentSession("custom", budget_usd=5.0, run_id="my_custom_id") as session:
        assert session.run_id == "my_custom_id"


def test_session_auto_route():
    """auto_route flag must be passed to completion."""
    session = AgentSession("route_test", budget_usd=5.0, auto_route=True)
    assert session.auto_route is True
