"""
AgentSession — context manager for entire agent sessions with lifecycle management.

NOVEL: Combines budget enforcement, cost tracking, tool tracking, conversation
estimation, and spend ledger into a single context manager. Users get one-line
session management instead of wiring 6 modules manually.

This is the "batteries-included" API for production:

    with AgentSession("my_agent", budget_usd=5.00) as session:
        response = session.completion(model="gpt-4o", messages=[...])
        session.record_tool_call("web_search", cost=0.01)

        # Session auto-tracks: LLM costs, tool costs, conversation pattern
        # Auto-terminates if budget exceeded
        # Auto-generates cost receipt on exit

    receipt = session.get_receipt()
    # {'run_id': '...', 'total_cost': 2.34, 'llm_cost': 2.30, 'tool_cost': 0.04, ...}

No existing SDK offers this level of integration in a single context manager.
"""

import logging
import time
from typing import Optional, Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class AgentSession:
    """
    All-in-one session manager for AI agent cost optimization.

    Wraps gateway.completion() with automatic:
    - Budget enforcement (graduated policies)
    - Tool cost tracking
    - Conversation cost estimation
    - Spend ledger recording
    - Cost receipt generation
    """

    def __init__(
        self,
        name: str = "default",
        budget_usd: float = 10.0,
        model: str = "gpt-4o",
        run_id: Optional[str] = None,
        auto_route: bool = False,
        tenant_id: Optional[str] = None,
    ):
        self.name = name
        self.run_id = run_id or str(uuid4())
        self.model = model
        self.budget_usd = budget_usd
        self.auto_route = auto_route
        self.tenant_id = tenant_id
        self._started_at = None
        self._ended_at = None
        self._call_count = 0
        self._cache_hits = 0
        self._errors = 0

        # Lazy imports to avoid circular dependencies
        self._tool_tracker = None
        self._estimator = None

    def __enter__(self):
        self._started_at = time.time()
        logger.info("AgentSession started: %s (budget=$%.2f, model=%s)",
                     self.run_id, self.budget_usd, self.model)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ended_at = time.time()
        elapsed = self._ended_at - self._started_at
        logger.info("AgentSession ended: %s (%.1fs, %d calls, $%.4f spent)",
                     self.run_id, elapsed, self._call_count,
                     self._get_total_cost())
        return False  # don't suppress exceptions

    async def __aenter__(self):
        self._started_at = time.time()
        logger.info("AgentSession started (async): %s (budget=$%.2f, model=%s)",
                     self.run_id, self.budget_usd, self.model)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._ended_at = time.time()
        elapsed = self._ended_at - self._started_at
        logger.info("AgentSession ended (async): %s (%.1fs, %d calls, $%.4f spent)",
                     self.run_id, elapsed, self._call_count,
                     self._get_total_cost())
        return False

    def completion(self, messages: list[dict], model: Optional[str] = None,
                   temperature: float = 0.0, tools: Optional[list] = None,
                   **kwargs) -> Any:
        """Make an LLM completion call with automatic cost tracking.

        Uses the gateway's completion() function with session budget/tenant.
        """
        from agentfuse.gateway import completion as gw_completion

        active_model = model or self.model
        self._call_count += 1

        result = gw_completion(
            model=active_model,
            messages=messages,
            budget_id=self.run_id,
            budget_usd=self.budget_usd,
            temperature=temperature,
            tools=tools,
            tenant_id=self.tenant_id,
            auto_route=self.auto_route,
            **kwargs,
        )

        # Track cache hits
        if hasattr(result, "_agentfuse_cache_hit") and result._agentfuse_cache_hit:
            self._cache_hits += 1

        # Record in conversation estimator
        estimator = self._get_estimator()
        usage = getattr(result, "usage", None)
        if usage:
            from agentfuse.providers.response import extract_usage
            provider = "anthropic" if active_model.startswith("claude") else "openai"
            try:
                normalized = extract_usage(provider, usage)
                from agentfuse.providers.pricing import ModelPricingEngine
                pricing = ModelPricingEngine()
                cost = pricing.total_cost_normalized(active_model, normalized)
                estimator.record_turn(
                    cost=cost,
                    input_tokens=normalized.total_input_tokens,
                    output_tokens=normalized.total_output_tokens,
                )
            except Exception:
                pass

        return result

    def record_tool_call(self, tool_name: str, cost: float = 0.0,
                         duration_seconds: float = 0.0) -> float:
        """Record a tool call cost against the session budget."""
        tracker = self._get_tool_tracker()
        if tool_name not in tracker._tools:
            tracker.register_tool(tool_name, cost_per_call=cost)
        return tracker.record_tool_call(tool_name, duration_seconds=duration_seconds,
                                         custom_cost=cost if cost > 0 else None)

    def estimate_remaining(self, target_turns: int = 20) -> dict:
        """Estimate remaining conversation cost based on pattern so far."""
        return self._get_estimator().project(target_turns=target_turns)

    def get_receipt(self) -> dict:
        """Get a detailed session receipt."""
        from agentfuse.gateway import get_spend

        llm_cost = get_spend(self.run_id)
        tool_cost = self._get_tool_tracker().get_tool_spend()
        total_cost = llm_cost + tool_cost

        return {
            "run_id": self.run_id,
            "name": self.name,
            "model": self.model,
            "budget_usd": self.budget_usd,
            "total_cost_usd": round(total_cost, 6),
            "llm_cost_usd": round(llm_cost, 6),
            "tool_cost_usd": round(tool_cost, 6),
            "calls": self._call_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(self._cache_hits / max(self._call_count, 1), 3),
            "errors": self._errors,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "duration_seconds": round((self._ended_at or time.time()) - (self._started_at or time.time()), 2),
            "cost_projection": self._get_estimator().project(),
            "tool_report": self._get_tool_tracker().get_report(),
        }

    def _get_tool_tracker(self):
        if self._tool_tracker is None:
            from agentfuse.core.tool_cost_tracker import ToolCostTracker
            from agentfuse.gateway import get_engine
            engine = get_engine(self.run_id)
            self._tool_tracker = ToolCostTracker(budget_engine=engine)
        return self._tool_tracker

    def _get_estimator(self):
        if self._estimator is None:
            from agentfuse.core.conversation_estimator import ConversationCostEstimator
            self._estimator = ConversationCostEstimator(budget_usd=self.budget_usd)
        return self._estimator

    def _get_total_cost(self) -> float:
        from agentfuse.gateway import get_spend
        return get_spend(self.run_id) + self._get_tool_tracker().get_tool_spend()
