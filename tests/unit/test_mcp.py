"""
Tests for MCP (Model Context Protocol) integration.
"""

from agentfuse.integrations.mcp import (
    handle_mcp_tool_call, get_mcp_tool_definitions, MCP_TOOLS,
)
from agentfuse.gateway import cleanup


def setup_function():
    cleanup()


def test_tool_definitions_valid():
    """MCP tool definitions must be valid."""
    tools = get_mcp_tool_definitions()
    assert len(tools) >= 5
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool


def test_budget_check_not_found():
    """Budget check for unknown ID must return not_found."""
    result = handle_mcp_tool_call("agentfuse_budget_check", {"budget_id": "unknown"})
    assert result["status"] == "not_found"


def test_budget_check_existing():
    """Budget check for existing run must return budget info."""
    from agentfuse.gateway import _get_engine
    _get_engine("mcp_test", 5.0, "gpt-4o")
    result = handle_mcp_tool_call("agentfuse_budget_check", {"budget_id": "mcp_test"})
    assert result["budget_usd"] == 5.0
    assert result["state"] == "normal"


def test_estimate_cost():
    """Estimate cost must return valid estimate."""
    result = handle_mcp_tool_call("agentfuse_estimate_cost", {
        "model": "gpt-4o",
        "prompt": "What is Python?",
    })
    assert result["model"] == "gpt-4o"
    assert result["estimated_total_cost_usd"] > 0


def test_spend_report():
    """Spend report must return valid structure."""
    result = handle_mcp_tool_call("agentfuse_spend_report", {})
    assert "total_usd" in result
    assert "by_model" in result


def test_kill():
    """Kill must activate kill switch."""
    result = handle_mcp_tool_call("agentfuse_kill", {
        "run_id": "mcp_kill_test",
        "reason": "test kill",
    })
    assert result["status"] == "killed"

    # Verify it's actually killed
    from agentfuse.core.kill_switch import kill_switch
    assert kill_switch.is_killed("mcp_kill_test") is True
    kill_switch.revive("mcp_kill_test")


def test_recommend():
    """Recommend must return a valid model."""
    result = handle_mcp_tool_call("agentfuse_recommend_model", {
        "budget_remaining": 5.0,
        "task_type": "code",
    })
    assert "recommended_model" in result
    assert len(result["recommended_model"]) > 0


def test_unknown_tool():
    """Unknown tool must return error."""
    result = handle_mcp_tool_call("unknown_tool", {})
    assert "error" in result


def test_tool_names():
    """All tool names must start with agentfuse_."""
    for tool in MCP_TOOLS:
        assert tool["name"].startswith("agentfuse_")
