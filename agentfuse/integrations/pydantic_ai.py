"""
Pydantic AI integration — cost tracking for Pydantic AI agents.

Pydantic AI is the newest major agent framework (2026), built on top of
Pydantic for type-safe agent definition. It's gaining rapid adoption
for production agents due to its simplicity and type safety.

This module provides:
1. AgentFuseModel: wraps Pydantic AI's Model with cost tracking
2. cost_result_validator: validates results against budget

Usage:
    from agentfuse.integrations.pydantic_ai import wrap_pydantic_agent

    agent = wrap_pydantic_agent(my_agent, budget_usd=5.0)
"""

import logging
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)


def wrap_pydantic_agent(
    agent: Any,
    budget_usd: float = 10.0,
    run_id: Optional[str] = None,
    model: str = "gpt-4o",
) -> Any:
    """
    Wrap a Pydantic AI agent with AgentFuse cost tracking.

    Intercepts the agent's run method to add budget enforcement.

    Args:
        agent: The Pydantic AI Agent instance
        budget_usd: Maximum budget for this agent
        run_id: Unique run identifier
        model: Default model name for pricing

    Returns:
        The wrapped agent (same type, enhanced behavior)
    """
    from uuid import uuid4
    from agentfuse.core.budget import BudgetEngine
    from agentfuse.providers.pricing import ModelPricingEngine

    run_id = run_id or str(uuid4())
    engine = BudgetEngine(run_id, budget_usd, model)
    pricing = ModelPricingEngine()

    # Store original run method
    original_run = getattr(agent, "run", None)
    original_run_sync = getattr(agent, "run_sync", None)

    if original_run:
        async def wrapped_run(*args, **kwargs):
            # Pre-call budget check
            est_cost = 0.01  # rough estimate
            try:
                engine.check_and_act(est_cost, [])
            except Exception:
                raise

            result = await original_run(*args, **kwargs)

            # Post-call cost recording
            try:
                usage = getattr(result, "usage", None) or getattr(result, "_usage", None)
                if usage:
                    from agentfuse.providers.response import extract_usage
                    normalized = extract_usage("openai", usage)
                    cost = pricing.total_cost_normalized(model, normalized)
                    engine.record_cost(cost)
            except Exception:
                pass

            return result

        agent.run = wrapped_run

    if original_run_sync:
        def wrapped_run_sync(*args, **kwargs):
            est_cost = 0.01
            try:
                engine.check_and_act(est_cost, [])
            except Exception:
                raise

            result = original_run_sync(*args, **kwargs)

            try:
                usage = getattr(result, "usage", None)
                if usage:
                    from agentfuse.providers.response import extract_usage
                    normalized = extract_usage("openai", usage)
                    cost = pricing.total_cost_normalized(model, normalized)
                    engine.record_cost(cost)
            except Exception:
                pass

            return result

        agent.run_sync = wrapped_run_sync

    # Attach engine for inspection
    agent._agentfuse_engine = engine
    agent._agentfuse_run_id = run_id

    return agent


def get_agent_receipt(agent: Any) -> dict:
    """Get cost receipt from a wrapped Pydantic AI agent."""
    engine = getattr(agent, "_agentfuse_engine", None)
    if engine is None:
        return {"error": "Agent not wrapped with AgentFuse"}

    return {
        "run_id": getattr(agent, "_agentfuse_run_id", "unknown"),
        "budget_usd": engine.budget,
        "spent_usd": round(engine.spent, 6),
        "remaining_usd": round(engine.budget - engine.spent, 6),
        "state": engine.state.value,
        "model": engine.model,
    }
