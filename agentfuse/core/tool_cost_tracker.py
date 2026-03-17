"""
ToolCostTracker — track costs for tool/function calls alongside LLM costs.

Production AI agents don't just make LLM calls — they use tools:
- Web search APIs ($0.01-0.05 per search)
- Code execution (compute time)
- Database queries
- External API calls (e.g., weather, maps, payments)

These tool costs can exceed LLM costs in complex agent workflows.
No existing Python SDK tracks both in a unified budget.

This module provides:
1. Per-tool cost registration (set expected cost per tool)
2. Automatic cost recording when tools are called
3. Unified budget with BudgetEngine (LLM + tool costs in one balance)
4. Cost attribution per tool for analytics

Usage:
    tracker = ToolCostTracker(budget_engine)
    tracker.register_tool("web_search", cost_per_call=0.01)
    tracker.register_tool("code_exec", cost_per_second=0.001)

    # When a tool is called:
    tracker.record_tool_call("web_search")  # $0.01 deducted from budget
    tracker.record_tool_call("code_exec", duration_seconds=5.0)  # $0.005 deducted

    # Analytics:
    report = tracker.get_report()
    # {'web_search': {'calls': 5, 'total_cost': 0.05}, ...}
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolConfig:
    """Configuration for a registered tool."""
    name: str
    cost_per_call: float = 0.0
    cost_per_second: float = 0.0  # for compute-time tools
    max_calls: Optional[int] = None  # rate limit per session


@dataclass
class ToolUsage:
    """Usage statistics for a tool."""
    calls: int = 0
    total_cost: float = 0.0
    total_duration: float = 0.0
    last_called: float = 0.0


class ToolCostExceeded(Exception):
    """Raised when tool budget or rate limit is exceeded."""
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool cost exceeded for '{tool_name}': {message}")


# Pre-configured costs for common LLM tools (March 2026 pricing)
KNOWN_TOOL_COSTS = {
    # Anthropic server tools
    "web_search": 0.01,          # $10 per 1,000 searches
    "web_fetch": 0.005,          # estimated
    "text_editor": 0.0,          # no extra cost (token-based)
    "code_execution": 0.0,       # free with web search/fetch
    # Common external tools
    "google_search": 0.005,      # $5 per 1,000 searches (SerpAPI)
    "bing_search": 0.005,        # similar
    "wolfram_alpha": 0.01,       # Wolfram API
    "weather_api": 0.001,        # weather APIs
    "database_query": 0.0,       # internal, no external cost
}


class ToolCostTracker:
    """
    Tracks tool/function call costs alongside LLM costs.

    Integrates with BudgetEngine so LLM + tool costs share one budget.
    This is critical for production agents that make 10-100+ tool calls
    per session — tool costs can exceed LLM costs.
    """

    def __init__(self, budget_engine=None, tool_budget_usd: Optional[float] = None):
        """
        Args:
            budget_engine: Optional BudgetEngine for unified LLM+tool budget
            tool_budget_usd: Separate tool-only budget (if no budget_engine)
        """
        self._budget_engine = budget_engine
        self._tool_budget = tool_budget_usd
        self._tool_spend = 0.0
        self._tools: dict[str, ToolConfig] = {}
        self._usage: dict[str, ToolUsage] = {}
        self._lock = threading.Lock()

    def register_tool(
        self,
        name: str,
        cost_per_call: Optional[float] = None,
        cost_per_second: float = 0.0,
        max_calls: Optional[int] = None,
    ):
        """Register a tool with its cost configuration.

        If cost_per_call is None, auto-detects from KNOWN_TOOL_COSTS.
        """
        if cost_per_call is None:
            cost_per_call = KNOWN_TOOL_COSTS.get(name, 0.0)

        self._tools[name] = ToolConfig(
            name=name,
            cost_per_call=cost_per_call,
            cost_per_second=cost_per_second,
            max_calls=max_calls,
        )
        if name not in self._usage:
            self._usage[name] = ToolUsage()

    def record_tool_call(
        self,
        tool_name: str,
        duration_seconds: float = 0.0,
        custom_cost: Optional[float] = None,
    ) -> float:
        """
        Record a tool call and deduct cost from budget.

        Args:
            tool_name: Name of the tool called
            duration_seconds: Duration for compute-time-based billing
            custom_cost: Override the registered cost for this call

        Returns:
            The cost of this tool call

        Raises:
            ToolCostExceeded: If tool budget or rate limit exceeded
        """
        with self._lock:
            config = self._tools.get(tool_name)

            # Calculate cost
            if custom_cost is not None:
                cost = custom_cost
            elif config:
                cost = config.cost_per_call + (config.cost_per_second * duration_seconds)
            else:
                cost = 0.0
                logger.warning("Unregistered tool '%s' called — cost unknown", tool_name)

            # Check rate limit
            if config and config.max_calls is not None:
                usage = self._usage.get(tool_name, ToolUsage())
                if usage.calls >= config.max_calls:
                    raise ToolCostExceeded(
                        tool_name,
                        f"Rate limit exceeded ({config.max_calls} max calls)"
                    )

            # Check budget
            if self._tool_budget is not None:
                if self._tool_spend + cost > self._tool_budget:
                    raise ToolCostExceeded(
                        tool_name,
                        f"Tool budget exceeded (${self._tool_spend:.4f} + ${cost:.4f} > ${self._tool_budget:.2f})"
                    )

            # Record usage
            if tool_name not in self._usage:
                self._usage[tool_name] = ToolUsage()
            usage = self._usage[tool_name]
            usage.calls += 1
            usage.total_cost += cost
            usage.total_duration += duration_seconds
            usage.last_called = time.time()

            self._tool_spend += cost

            # Deduct from unified budget if available
            if self._budget_engine and cost > 0:
                self._budget_engine.record_cost(cost)

            return cost

    def get_tool_spend(self) -> float:
        """Get total spend on tool calls."""
        with self._lock:
            return self._tool_spend

    def get_report(self) -> dict:
        """Get detailed cost report per tool."""
        with self._lock:
            tools_report = {}
            for name, usage in self._usage.items():
                config = self._tools.get(name)
                tools_report[name] = {
                    "calls": usage.calls,
                    "total_cost": round(usage.total_cost, 6),
                    "total_duration": round(usage.total_duration, 3),
                    "cost_per_call": config.cost_per_call if config else 0.0,
                    "cost_per_second": config.cost_per_second if config else 0.0,
                }
            return {
                "total_tool_spend": round(self._tool_spend, 6),
                "total_tool_calls": sum(u.calls for u in self._usage.values()),
                "tools": tools_report,
            }

    def reset(self):
        """Reset all usage tracking."""
        with self._lock:
            self._tool_spend = 0.0
            self._usage.clear()
