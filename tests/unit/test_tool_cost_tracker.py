"""
Tests for ToolCostTracker — per-tool cost tracking.
"""

import pytest

from agentfuse.core.tool_cost_tracker import ToolCostTracker, ToolCostExceeded
from agentfuse.core.budget import BudgetEngine


def test_register_and_record():
    """Registering a tool and recording calls must track cost."""
    tracker = ToolCostTracker()
    tracker.register_tool("web_search", cost_per_call=0.01)
    cost = tracker.record_tool_call("web_search")
    assert cost == 0.01
    assert tracker.get_tool_spend() == 0.01


def test_multiple_calls():
    """Multiple tool calls must accumulate cost."""
    tracker = ToolCostTracker()
    tracker.register_tool("web_search", cost_per_call=0.01)
    for _ in range(5):
        tracker.record_tool_call("web_search")
    assert abs(tracker.get_tool_spend() - 0.05) < 1e-9


def test_compute_time_billing():
    """Compute-time tools must bill per second."""
    tracker = ToolCostTracker()
    tracker.register_tool("code_exec", cost_per_second=0.001)
    cost = tracker.record_tool_call("code_exec", duration_seconds=10.0)
    assert abs(cost - 0.01) < 1e-9


def test_custom_cost_override():
    """Custom cost must override registered cost."""
    tracker = ToolCostTracker()
    tracker.register_tool("api_call", cost_per_call=0.05)
    cost = tracker.record_tool_call("api_call", custom_cost=0.10)
    assert cost == 0.10


def test_rate_limit_enforcement():
    """Exceeding max_calls must raise ToolCostExceeded."""
    tracker = ToolCostTracker()
    tracker.register_tool("search", cost_per_call=0.01, max_calls=3)
    tracker.record_tool_call("search")
    tracker.record_tool_call("search")
    tracker.record_tool_call("search")
    with pytest.raises(ToolCostExceeded, match="Rate limit"):
        tracker.record_tool_call("search")


def test_tool_budget_enforcement():
    """Exceeding tool budget must raise ToolCostExceeded."""
    tracker = ToolCostTracker(tool_budget_usd=0.05)
    tracker.register_tool("search", cost_per_call=0.02)
    tracker.record_tool_call("search")
    tracker.record_tool_call("search")
    with pytest.raises(ToolCostExceeded, match="budget exceeded"):
        tracker.record_tool_call("search")  # $0.06 > $0.05


def test_unified_budget_integration():
    """Tool costs must deduct from BudgetEngine budget."""
    engine = BudgetEngine("unified_run", 10.0, "gpt-4o")
    tracker = ToolCostTracker(budget_engine=engine)
    tracker.register_tool("web_search", cost_per_call=1.0)
    tracker.record_tool_call("web_search")
    assert engine.spent == 1.0  # tool cost recorded in engine


def test_report_structure():
    """Report must contain expected keys and values."""
    tracker = ToolCostTracker()
    tracker.register_tool("search", cost_per_call=0.01)
    tracker.register_tool("code", cost_per_second=0.001)
    tracker.record_tool_call("search")
    tracker.record_tool_call("code", duration_seconds=5.0)

    report = tracker.get_report()
    assert report["total_tool_spend"] > 0
    assert report["total_tool_calls"] == 2
    assert "search" in report["tools"]
    assert report["tools"]["search"]["calls"] == 1
    assert report["tools"]["code"]["total_duration"] == 5.0


def test_unregistered_tool_no_crash():
    """Calling an unregistered tool must not crash (zero cost)."""
    tracker = ToolCostTracker()
    cost = tracker.record_tool_call("unknown_tool")
    assert cost == 0.0


def test_reset_clears_usage():
    """Reset must clear all usage tracking."""
    tracker = ToolCostTracker()
    tracker.register_tool("search", cost_per_call=0.01)
    tracker.record_tool_call("search")
    tracker.reset()
    assert tracker.get_tool_spend() == 0.0
    assert tracker.get_report()["total_tool_calls"] == 0


def test_known_tool_auto_pricing():
    """Known tools must auto-detect pricing when cost_per_call is None."""
    tracker = ToolCostTracker()
    tracker.register_tool("web_search")  # no cost specified
    cost = tracker.record_tool_call("web_search")
    assert cost == 0.01  # auto-detected from KNOWN_TOOL_COSTS


def test_unknown_tool_zero_cost():
    """Unknown tool without explicit cost must default to zero."""
    tracker = ToolCostTracker()
    tracker.register_tool("custom_tool")
    cost = tracker.record_tool_call("custom_tool")
    assert cost == 0.0


def test_explicit_cost_overrides_known():
    """Explicit cost must override known tool cost."""
    tracker = ToolCostTracker()
    tracker.register_tool("web_search", cost_per_call=0.05)  # override
    cost = tracker.record_tool_call("web_search")
    assert cost == 0.05  # not the default 0.01


def test_thread_safety():
    """Concurrent tool calls must not corrupt state."""
    import threading
    tracker = ToolCostTracker()
    tracker.register_tool("search", cost_per_call=0.001)
    errors = []

    def worker():
        try:
            for _ in range(100):
                tracker.record_tool_call("search")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert abs(tracker.get_tool_spend() - 0.5) < 1e-6  # 500 calls × $0.001
