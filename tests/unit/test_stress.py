"""
Stress tests — verify correctness under heavy concurrent load.
"""

import threading
import time

from agentfuse.core.budget import BudgetEngine
from agentfuse.core.cost_tracker import CostTracker
from agentfuse.core.tool_cost_tracker import ToolCostTracker
from agentfuse.core.conversation_estimator import ConversationCostEstimator
from agentfuse.core.predictive_router import CostPredictiveRouter
from agentfuse.core.kill_switch import KillSwitch
from agentfuse.core.hierarchical_budget import HierarchicalBudget
from agentfuse.core.guardrails import ContentGuardrails
from agentfuse.core.dedup import RequestDeduplicator


def _run_threaded(fn, n_threads=10, iterations=100):
    """Run fn in n_threads, each calling it `iterations` times."""
    errors = []
    def worker():
        try:
            for _ in range(iterations):
                fn()
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_budget_engine_concurrent_record():
    """Budget engine must handle concurrent record_cost correctly."""
    engine = BudgetEngine("stress_budget", 1000.0, "gpt-4o")
    errors = _run_threaded(lambda: engine.record_cost(0.001), n_threads=10, iterations=100)
    assert len(errors) == 0
    assert abs(engine.spent - 1.0) < 0.01  # 10*100*0.001 = 1.0


def test_cost_tracker_concurrent():
    """CostTracker must handle concurrent record_call correctly."""
    tracker = CostTracker()
    errors = _run_threaded(
        lambda: tracker.record_call("gpt-4o", "openai", "run1", cost_usd=0.001),
        n_threads=5, iterations=50
    )
    assert len(errors) == 0


def test_tool_tracker_concurrent():
    """ToolCostTracker must be thread-safe."""
    tracker = ToolCostTracker()
    tracker.register_tool("search", cost_per_call=0.001)
    errors = _run_threaded(lambda: tracker.record_tool_call("search"), n_threads=5, iterations=50)
    assert len(errors) == 0
    assert abs(tracker.get_tool_spend() - 0.25) < 0.001  # 250 * 0.001


def test_estimator_concurrent():
    """ConversationCostEstimator must be thread-safe."""
    est = ConversationCostEstimator(budget_usd=100.0)
    errors = _run_threaded(lambda: est.record_turn(cost=0.01), n_threads=5, iterations=50)
    assert len(errors) == 0


def test_predictive_router_concurrent():
    """CostPredictiveRouter must be thread-safe."""
    router = CostPredictiveRouter(budget_usd=100.0)
    errors = _run_threaded(
        lambda: (router.record_cost(0.001), router.predict_and_route("gpt-4o")),
        n_threads=5, iterations=50
    )
    assert len(errors) == 0


def test_kill_switch_concurrent():
    """KillSwitch must handle concurrent operations."""
    ks = KillSwitch()
    errors = _run_threaded(
        lambda: (ks.kill("run_stress", "test"), ks.revive("run_stress")),
        n_threads=5, iterations=20
    )
    assert len(errors) == 0


def test_guardrails_concurrent():
    """ContentGuardrails must be thread-safe."""
    g = ContentGuardrails()
    g.add_rule("max_length", max_chars=10000)
    g.add_rule("no_pii", patterns=["email"])
    errors = _run_threaded(
        lambda: g.validate("This is a test response with no PII."),
        n_threads=5, iterations=50
    )
    assert len(errors) == 0


def test_dedup_concurrent():
    """RequestDeduplicator must handle concurrent execute correctly."""
    dedup = RequestDeduplicator()
    counter = {"value": 0}
    lock = threading.Lock()

    def expensive_fn():
        with lock:
            counter["value"] += 1
        return "result"

    errors = _run_threaded(
        lambda: dedup.execute(f"key_{threading.current_thread().name}", expensive_fn),
        n_threads=5, iterations=10
    )
    assert len(errors) == 0
