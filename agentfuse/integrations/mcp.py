"""
MCP (Model Context Protocol) integration for AgentFuse.

MCP is the industry standard for agent-tool integration (97M+ monthly downloads,
backed by Anthropic, OpenAI, Google, Microsoft, Linux Foundation).

This module exposes AgentFuse capabilities as MCP tools that any MCP-compatible
agent can use:
- budget_check: Check remaining budget before making a call
- estimate_cost: Preview what a call will cost
- get_spend: Get current spend for a run
- kill_agent: Emergency stop an agent run

Usage with any MCP client:
    # AgentFuse registers as an MCP tool server
    tools = [
        {"name": "agentfuse_budget_check", "description": "Check remaining budget"},
        {"name": "agentfuse_estimate_cost", "description": "Estimate LLM call cost"},
        {"name": "agentfuse_spend_report", "description": "Get spend breakdown"},
        {"name": "agentfuse_kill", "description": "Emergency stop agent"},
    ]
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


# MCP Tool Definitions (JSON Schema format for tool registration)
MCP_TOOLS = [
    {
        "name": "agentfuse_budget_check",
        "description": "Check remaining budget for an agent run. Returns budget, spent, remaining, and state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "budget_id": {"type": "string", "description": "The budget/run ID to check"},
            },
            "required": ["budget_id"],
        },
    },
    {
        "name": "agentfuse_estimate_cost",
        "description": "Estimate the cost of an LLM call before making it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Model name (e.g., gpt-4o)"},
                "prompt": {"type": "string", "description": "The prompt text"},
                "max_output_tokens": {"type": "integer", "description": "Max output tokens", "default": 1000},
            },
            "required": ["model", "prompt"],
        },
    },
    {
        "name": "agentfuse_spend_report",
        "description": "Get a detailed spend report (total, by model, by provider, by run).",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "agentfuse_kill",
        "description": "Emergency stop an agent run. Blocks all future LLM calls for this run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The run ID to kill"},
                "reason": {"type": "string", "description": "Reason for killing", "default": "MCP kill request"},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "agentfuse_recommend_model",
        "description": "Get a model recommendation based on budget and task type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "budget_remaining": {"type": "number", "description": "Remaining budget in USD"},
                "task_type": {"type": "string", "enum": ["code", "reasoning", "factual", "creative"]},
                "latency_priority": {"type": "string", "enum": ["low", "balanced", "high"], "default": "balanced"},
            },
            "required": ["budget_remaining"],
        },
    },
]


def handle_mcp_tool_call(tool_name: str, arguments: dict) -> Any:
    """
    Handle an MCP tool call from any MCP-compatible agent.

    Args:
        tool_name: The MCP tool name
        arguments: Tool arguments dict

    Returns:
        Tool result (dict or string)
    """
    if tool_name == "agentfuse_budget_check":
        return _handle_budget_check(arguments)
    elif tool_name == "agentfuse_estimate_cost":
        return _handle_estimate_cost(arguments)
    elif tool_name == "agentfuse_spend_report":
        return _handle_spend_report()
    elif tool_name == "agentfuse_kill":
        return _handle_kill(arguments)
    elif tool_name == "agentfuse_recommend_model":
        return _handle_recommend(arguments)
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def _handle_budget_check(args: dict) -> dict:
    from agentfuse.gateway import get_engine
    budget_id = args["budget_id"]
    engine = get_engine(budget_id)
    if engine is None:
        return {"budget_id": budget_id, "status": "not_found"}
    return {
        "budget_id": budget_id,
        "budget_usd": engine.budget,
        "spent_usd": round(engine.spent, 6),
        "remaining_usd": round(engine.budget - engine.spent, 6),
        "state": engine.state.value,
        "model": engine.model,
    }


def _handle_estimate_cost(args: dict) -> dict:
    from agentfuse.gateway import estimate_cost
    model = args["model"]
    prompt = args["prompt"]
    max_output = args.get("max_output_tokens", 1000)
    messages = [{"role": "user", "content": prompt}]
    return estimate_cost(model, messages, max_output_tokens=max_output)


def _handle_spend_report() -> dict:
    from agentfuse.gateway import get_spend_report
    return get_spend_report()


def _handle_kill(args: dict) -> dict:
    from agentfuse.core.kill_switch import kill_switch
    run_id = args["run_id"]
    reason = args.get("reason", "MCP kill request")
    kill_switch.kill(run_id, reason, killed_by="mcp")
    return {"status": "killed", "run_id": run_id, "reason": reason}


def _handle_recommend(args: dict) -> dict:
    from agentfuse.core.model_recommender import ModelRecommender
    recommender = ModelRecommender()
    model = recommender.recommend(
        budget_remaining=args["budget_remaining"],
        task_type=args.get("task_type"),
        latency_priority=args.get("latency_priority", "balanced"),
    )
    return {"recommended_model": model}


def get_mcp_tool_definitions() -> list[dict]:
    """Get MCP tool definitions for registration with an MCP server."""
    return MCP_TOOLS
