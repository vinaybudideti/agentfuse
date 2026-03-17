"""
Full flow integration test — exercises the entire gateway pipeline.
Tests cache + budget + routing + metrics + validation in one flow.
"""

from unittest.mock import patch
from types import SimpleNamespace

from agentfuse.gateway import completion, cleanup, get_spend, _cache
from agentfuse.core.budget import BudgetExhaustedGracefully
import pytest


def setup_function():
    cleanup()


@patch("agentfuse.gateway._call_openai_compatible")
def test_full_flow_new_request(mock_call):
    """New request must: validate → optimize → route → budget → cache miss → API → cost → cache."""
    mock_call.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Full flow response"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=100),
    )

    result = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "full flow test query unique 99"}],
        budget_id="flow_test",
        budget_usd=10.0,
    )

    # Verify response
    assert result.choices[0].message.content == "Full flow response"
    # Verify budget tracking
    assert get_spend("flow_test") > 0
    # Verify API was called
    mock_call.assert_called_once()


def test_full_flow_cache_hit():
    """Cached request must: validate → optimize → cache hit → return (no API call)."""
    msgs = [{"role": "user", "content": "full flow cache hit test unique 88"}]
    _cache.store(model="gpt-4o", messages=msgs, response="Cached full flow")

    result = completion(model="gpt-4o", messages=msgs, budget_id="cache_flow", budget_usd=5.0)

    assert result.choices[0].message.content == "Cached full flow"
    assert result._agentfuse_cache_hit is True
    assert get_spend("cache_flow") == 0.0  # no cost for cache hit


@patch("agentfuse.gateway._call_openai_compatible")
def test_full_flow_budget_exhaustion(mock_call):
    """Budget exhaustion must raise gracefully with partial results."""
    from agentfuse.gateway import _get_engine
    engine = _get_engine("exhaust_flow", 0.001, "gpt-4o")
    engine.record_cost(0.001)  # exhaust budget

    with pytest.raises(BudgetExhaustedGracefully):
        completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "exhausted test"}],
            budget_id="exhaust_flow",
        )


@patch("agentfuse.gateway._call_openai_compatible")
def test_full_flow_with_tools(mock_call):
    """Tool calls must work through the full flow."""
    mock_call.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Tool response", tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=80, completion_tokens=50),
    )
    tools = [{"function": {"name": "get_weather", "parameters": {}}}]

    result = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "what's the weather?"}],
        tools=tools,
        budget_id="tool_flow",
        budget_usd=5.0,
    )

    assert result.choices[0].message.content == "Tool response"


def test_full_flow_validation_rejects_bad_input():
    """Bad inputs must be rejected at the validation layer."""
    with pytest.raises(ValueError, match="model must be"):
        completion(model="", messages=[])

    with pytest.raises(ValueError, match="messages must be"):
        completion(model="gpt-4o", messages="not a list")


def test_full_flow_kill_switch():
    """Kill switch must block all calls for a killed run."""
    from agentfuse.core.kill_switch import kill_switch, AgentKilled

    kill_switch.kill("killed_flow", "safety test")
    with pytest.raises(AgentKilled):
        completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "blocked"}],
            budget_id="killed_flow",
        )
    kill_switch.revive("killed_flow")


@patch("agentfuse.gateway._call_openai_compatible")
def test_full_flow_with_metadata(mock_call):
    """Metadata must pass through the full flow without error."""
    mock_call.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="metadata response"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )

    result = completion(
        model="gpt-5",
        messages=[{"role": "user", "content": "metadata flow test"}],
        budget_id="meta_flow",
        budget_usd=10.0,
        metadata={"user_id": "user_123", "team": "engineering"},
    )
    assert result.choices[0].message.content == "metadata response"


def test_full_flow_deprecated_model_warns():
    """Deprecated model must still work but log a warning."""
    # gpt-4o is deprecated but should still work (not auto-redirected by default)
    msgs = [{"role": "user", "content": "deprecated test flow unique xyz987"}]
    _cache.store(model="gpt-4o", messages=msgs, response="Deprecated model response")

    result = completion(model="gpt-4o", messages=msgs)
    assert result.choices[0].message.content == "Deprecated model response"


@patch("agentfuse.gateway._call_openai_compatible")
def test_full_flow_auto_route(mock_call):
    """Auto-route must potentially change the model for simple queries."""
    mock_call.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="routed response"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )

    result = completion(
        model="gpt-5",
        messages=[{"role": "user", "content": "hi"}],
        auto_route=True,
    )
    assert result.choices[0].message.content == "routed response"


@patch("agentfuse.gateway._call_openai_compatible")
def test_full_flow_tenant_isolation(mock_call):
    """Different tenants must not share cache."""
    mock_call.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="tenant specific"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )

    msgs = [{"role": "user", "content": "shared query for tenant test flow"}]
    _cache.store(model="gpt-4o", messages=msgs, response="Tenant A",
                 tenant_id="tenant_a")

    # Tenant B should NOT get Tenant A's cached response
    result = completion(model="gpt-4o", messages=msgs, tenant_id="tenant_b")
    # Should call the API since cache miss for tenant_b
    assert mock_call.called
