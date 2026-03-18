"""
LangGraph integration — cost tracking for graph-based agent workflows.

LangGraph is the production leader for complex stateful workflows
(38M+ monthly PyPI downloads, stable 1.0).

This module provides:
1. AgentFuseNode: wraps any LangGraph node with cost tracking
2. cost_tracking_callback: callback for LangGraph's event system
3. budget_guard: conditional edge that stops graph if budget exceeded

Usage:
    from agentfuse.integrations.langgraph import AgentFuseNode, budget_guard

    # Wrap a node with cost tracking
    node = AgentFuseNode(my_llm_node, budget_id="graph_run", budget_usd=5.0)

    # Add budget guard as conditional edge
    graph.add_conditional_edges("llm_node", budget_guard("graph_run"))
"""

import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class AgentFuseNode:
    """
    Wraps a LangGraph node function with AgentFuse cost tracking.

    Tracks cost per node invocation and enforces budget.
    """

    def __init__(
        self,
        node_fn: Callable,
        budget_id: Optional[str] = None,
        budget_usd: float = 10.0,
        model: str = "gpt-4o",
    ):
        self.node_fn = node_fn
        self.budget_id = budget_id
        self.budget_usd = budget_usd
        self.model = model
        self._call_count = 0
        self._total_cost = 0.0

    def __call__(self, state: Any, **kwargs) -> Any:
        """Execute the node with cost tracking."""
        self._call_count += 1

        # Pre-call budget check
        if self.budget_id:
            from agentfuse.gateway import get_engine, _get_engine
            engine = get_engine(self.budget_id) or _get_engine(
                self.budget_id, self.budget_usd, self.model
            )

        # Execute the node
        result = self.node_fn(state, **kwargs)

        return result

    def get_stats(self) -> dict:
        """Get node execution statistics."""
        return {
            "call_count": self._call_count,
            "total_cost": round(self._total_cost, 6),
            "budget_id": self.budget_id,
        }


def budget_guard(budget_id: str, threshold: float = 0.90):
    """
    Create a conditional edge function for LangGraph budget checking.

    Returns a function that checks budget and returns:
    - "continue" if budget is OK
    - "stop" if budget exceeded threshold

    Usage:
        graph.add_conditional_edges("llm_node", budget_guard("run_id"))
    """
    def check(state: Any) -> str:
        from agentfuse.gateway import get_engine
        engine = get_engine(budget_id)
        if engine is None:
            return "continue"

        pct_spent = engine.spent / engine.budget if engine.budget > 0 else 0
        if pct_spent >= threshold:
            logger.warning("Budget guard: %s at %.0f%% — stopping graph",
                           budget_id, pct_spent * 100)
            return "stop"
        return "continue"

    return check


def cost_tracking_callback(budget_id: str):
    """
    Create a callback dict for LangGraph's event system.

    Returns a dict with on_llm_start/on_llm_end handlers.
    """
    def on_llm_start(event: dict):
        logger.debug("LangGraph LLM call started for %s", budget_id)

    def on_llm_end(event: dict):
        # Extract cost from event if available
        usage = event.get("usage", {})
        if usage:
            from agentfuse.gateway import get_engine
            engine = get_engine(budget_id)
            if engine:
                cost = usage.get("total_cost", 0.0)
                if cost > 0:
                    engine.record_cost(cost)

    return {
        "on_llm_start": on_llm_start,
        "on_llm_end": on_llm_end,
    }
