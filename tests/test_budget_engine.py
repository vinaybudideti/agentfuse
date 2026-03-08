# Tests for BudgetEngine: graduated policies (downgrade at 80%, compression at 90%, termination at 100%)

import pytest
from agentfuse.core.budget import BudgetEngine, BudgetState, BudgetExhaustedGracefully


def test_normal_below_60_pct():
    engine = BudgetEngine("run_1", budget_usd=1.00, model="gpt-4o")
    messages = [{"role": "user", "content": "hello"}]
    result_messages, result_model = engine.check_and_act(0.50, messages)
    assert result_model == "gpt-4o"
    assert engine.state == BudgetState.NORMAL


def test_alert_at_60_pct():
    alerts = []
    engine = BudgetEngine("run_1", 1.00, "gpt-4o",
                          alert_cb=lambda pct, event: alerts.append(event))
    engine.spent = 0.55
    engine.check_and_act(0.06, [{"role": "user", "content": "hi"}])
    assert "budget_alert" in alerts


def test_downgrade_at_80_pct():
    engine = BudgetEngine("run_1", 1.00, "gpt-4o")
    engine.spent = 0.75
    messages = [{"role": "user", "content": "hi"}]
    _, model = engine.check_and_act(0.06, messages)
    assert model == "gpt-4o-mini"
    assert engine.state == BudgetState.DOWNGRADED


def test_compression_at_90_pct():
    engine = BudgetEngine("run_1", 1.00, "gpt-4o")
    engine.spent = 0.85
    messages = (
        [{"role": "system", "content": "You are helpful"}] +
        [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    )
    compressed, _ = engine.check_and_act(0.06, messages)
    non_system = [m for m in compressed if m["role"] != "system"]
    assert len(non_system) == 6
    assert compressed[0]["role"] == "system"


def test_termination_at_100_pct():
    engine = BudgetEngine("run_1", 1.00, "gpt-4o")
    engine.spent = 0.95
    with pytest.raises(BudgetExhaustedGracefully):
        engine.check_and_act(0.10, [{"role": "user", "content": "hi"}])


def test_no_double_downgrade():
    engine = BudgetEngine("run_1", 1.00, "gpt-4o")
    engine.spent = 0.75
    messages = [{"role": "user", "content": "hi"}]
    engine.check_and_act(0.06, messages)
    assert engine.model == "gpt-4o-mini"
    engine.check_and_act(0.01, messages)
    assert engine.model == "gpt-4o-mini"
