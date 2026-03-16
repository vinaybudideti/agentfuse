"""
Phase 0 — BudgetEngine behavioral tests.

Tests verify actual behavior: threshold boundaries, state machine correctness,
context compression, graceful termination, and concurrent safety.
"""

import asyncio
import threading

import pytest

from agentfuse.core.budget import (
    BudgetEngine,
    BudgetState,
    BudgetExhaustedGracefully,
    _current_run_id,
)


def test_downgrade_at_exactly_80_pct():
    """At exactly 80% budget, model must be downgraded."""
    engine = BudgetEngine("run_80", 1.00, "gpt-4o")
    engine.spent = 0.79
    messages = [{"role": "user", "content": "hi"}]
    _, model = engine.check_and_act(0.01, messages)
    assert model == "gpt-4o-mini"
    assert engine.state == BudgetState.DOWNGRADED


def test_no_double_downgrade():
    """Once downgraded, model must not downgrade again on subsequent calls."""
    engine = BudgetEngine("run_dd", 1.00, "gpt-4o")
    engine.spent = 0.75
    messages = [{"role": "user", "content": "hi"}]
    engine.check_and_act(0.06, messages)
    assert engine.model == "gpt-4o-mini"
    engine.check_and_act(0.01, messages)
    assert engine.model == "gpt-4o-mini"


def test_compression_keeps_system_plus_6():
    """Compression must keep all system messages + last 6 non-system messages."""
    engine = BudgetEngine("run_comp", 1.00, "gpt-4o")
    engine.spent = 0.85
    messages = (
        [{"role": "system", "content": "You are helpful"}]
        + [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    )
    compressed, _ = engine.check_and_act(0.06, messages)
    non_system = [m for m in compressed if m["role"] != "system"]
    assert len(non_system) == 6
    assert compressed[0]["role"] == "system"
    # Should keep the LAST 6 messages
    assert non_system[0]["content"] == "msg 4"
    assert non_system[-1]["content"] == "msg 9"


def test_terminate_raises_with_partial_results():
    """At 100% budget, must raise BudgetExhaustedGracefully with partial results."""
    engine = BudgetEngine("run_term", 1.00, "gpt-4o")
    engine.spent = 0.95
    engine.add_partial_result("partial answer")
    with pytest.raises(BudgetExhaustedGracefully) as exc_info:
        engine.check_and_act(0.10, [{"role": "user", "content": "hi"}])
    assert exc_info.value.partial_results == ["partial answer"]


def test_concurrent_budget_check_no_bleed():
    """10 threads doing check_and_act on same engine must not corrupt state."""
    engine = BudgetEngine("run_conc", 10.0, "gpt-4o")
    errors = []

    def check_many():
        try:
            for _ in range(50):
                messages = [{"role": "user", "content": "hello"}]
                engine.check_and_act(0.01, messages)
        except BudgetExhaustedGracefully:
            pass
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=check_many) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threads raised unexpected errors: {errors}"
    assert engine.state in list(BudgetState)


def test_concurrent_record_cost_no_lost_updates():
    """10 threads recording cost must converge to correct total."""
    engine = BudgetEngine("run_cost", 100.0, "gpt-4o")
    errors = []

    def record_many():
        try:
            for _ in range(100):
                engine.record_cost(0.01)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=record_many) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # 10 threads * 100 iterations * 0.01 = 10.0
    assert abs(engine.spent - 10.0) < 0.001, f"Expected 10.0, got {engine.spent}"


def test_instance_level_locks_dont_block_other_engines():
    """Two different BudgetEngines must not block each other."""
    engine_a = BudgetEngine("run_a_lock", 10.0, "gpt-4o")
    engine_b = BudgetEngine("run_b_lock", 10.0, "gpt-4o")
    # They should have separate lock objects
    assert engine_a._sync_lock is not engine_b._sync_lock


def test_async_check_and_act():
    """Async version must work correctly."""
    async def run():
        engine = BudgetEngine("run_async", 1.00, "gpt-4o")
        messages = [{"role": "user", "content": "hi"}]
        result_messages, result_model = await engine.check_and_act_async(0.50, messages)
        assert result_model == "gpt-4o"
        return True

    assert asyncio.run(run())


def test_async_record_cost():
    """Async record_cost must work correctly."""
    async def run():
        engine = BudgetEngine("run_async_cost", 10.0, "gpt-4o")
        await engine.record_cost_async(0.50)
        await engine.record_cost_async(0.25)
        assert abs(engine.spent - 0.75) < 0.001
        return True

    assert asyncio.run(run())


def test_contextvar_set_on_init():
    """Creating a BudgetEngine must set the ContextVar in current context."""
    engine = BudgetEngine("run_cv_a", 1.00, "gpt-4o")
    assert BudgetEngine.get_current_run_id() == "run_cv_a"

    engine2 = BudgetEngine("run_cv_b", 1.00, "gpt-4o")
    assert BudgetEngine.get_current_run_id() == "run_cv_b"


def test_contextvar_thread_isolation():
    """ContextVar must give each thread its own run_id."""
    results = {}

    def thread_fn(run_id):
        engine = BudgetEngine(run_id, 1.00, "gpt-4o")
        # Small delay to interleave threads
        import time
        time.sleep(0.01)
        results[run_id] = BudgetEngine.get_current_run_id()

    t1 = threading.Thread(target=thread_fn, args=("thread_run_1",))
    t2 = threading.Thread(target=thread_fn, args=("thread_run_2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["thread_run_1"] == "thread_run_1"
    assert results["thread_run_2"] == "thread_run_2"


def test_alert_at_60_pct():
    """At 60% budget, alert callback must be triggered."""
    alerts = []
    engine = BudgetEngine("run_alert", 1.00, "gpt-4o",
                          alert_cb=lambda pct, event: alerts.append(event))
    engine.spent = 0.55
    engine.check_and_act(0.06, [{"role": "user", "content": "hi"}])
    assert "budget_alert" in alerts


def test_normal_below_60_pct():
    """Below 60% budget, state should remain NORMAL."""
    engine = BudgetEngine("run_normal", 1.00, "gpt-4o")
    messages = [{"role": "user", "content": "hello"}]
    result_messages, result_model = engine.check_and_act(0.50, messages)
    assert result_model == "gpt-4o"
    assert engine.state == BudgetState.NORMAL


def test_no_downgrade_at_79_pct():
    """At 79% budget (just below 80%), model must NOT be downgraded."""
    engine = BudgetEngine("run_79", 1.00, "gpt-4o")
    engine.spent = 0.78
    messages = [{"role": "user", "content": "hi"}]
    _, model = engine.check_and_act(0.01, messages)
    assert model == "gpt-4o"  # Still the original model


def test_compression_keeps_exactly_6_non_system_messages():
    """Compression must keep exactly 6 non-system messages, regardless of how many there are."""
    engine = BudgetEngine("run_comp_exact", 1.00, "gpt-4o")
    engine.spent = 0.85
    messages = (
        [{"role": "system", "content": "sys"}]
        + [{"role": "user", "content": f"m{i}"} for i in range(20)]
    )
    compressed, _ = engine.check_and_act(0.06, messages)
    non_system = [m for m in compressed if m["role"] != "system"]
    assert len(non_system) == 6


def test_terminate_raises_BudgetExhaustedGracefully_not_generic():
    """Must raise BudgetExhaustedGracefully specifically, not a generic Exception."""
    engine = BudgetEngine("run_term_type", 1.00, "gpt-4o")
    engine.spent = 1.00
    with pytest.raises(BudgetExhaustedGracefully):
        engine.check_and_act(0.01, [{"role": "user", "content": "hi"}])


def test_partial_results_preserved_in_exception():
    """Partial results accumulated before termination must be in the exception."""
    engine = BudgetEngine("run_partial", 1.00, "gpt-4o")
    engine.add_partial_result("result_1")
    engine.add_partial_result("result_2")
    engine.spent = 1.00
    with pytest.raises(BudgetExhaustedGracefully) as exc_info:
        engine.check_and_act(0.01, [{"role": "user", "content": "hi"}])
    assert len(exc_info.value.partial_results) == 2
    assert "result_1" in exc_info.value.partial_results
    assert "result_2" in exc_info.value.partial_results


def test_invalid_budget_raises():
    """Budget <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        BudgetEngine("run_invalid", 0.0, "gpt-4o")
    with pytest.raises(ValueError):
        BudgetEngine("run_invalid2", -1.0, "gpt-4o")


def test_downgrade_o3_to_o4_mini():
    """o3 at 80%+ budget must downgrade to o4-mini."""
    engine = BudgetEngine("run_o3", 1.00, "o3")
    engine.spent = 0.79
    _, model = engine.check_and_act(0.01, [{"role": "user", "content": "hi"}])
    assert model == "o4-mini"


def test_downgrade_gpt41_to_o4_mini():
    """gpt-4.1 at 80%+ budget must downgrade to o4-mini."""
    engine = BudgetEngine("run_41", 1.00, "gpt-4.1")
    engine.spent = 0.79
    _, model = engine.check_and_act(0.01, [{"role": "user", "content": "hi"}])
    assert model == "o4-mini"


def test_no_downgrade_unknown_model():
    """Unknown model with no downgrade path stays as-is."""
    engine = BudgetEngine("run_unk", 1.00, "custom-model")
    engine.spent = 0.79
    _, model = engine.check_and_act(0.01, [{"role": "user", "content": "hi"}])
    assert model == "custom-model"
