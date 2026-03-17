"""
Tests for KillSwitch — emergency stop for AI agents.
"""

import pytest

from agentfuse.core.kill_switch import KillSwitch, AgentKilled, kill_switch


def test_kill_and_check():
    """Killed agent must raise AgentKilled on check."""
    ks = KillSwitch()
    ks.kill("run_1", reason="test kill")
    with pytest.raises(AgentKilled, match="run_1"):
        ks.check("run_1")


def test_unkilled_agent_passes():
    """Non-killed agent must pass check without error."""
    ks = KillSwitch()
    ks.check("healthy_run")  # should not raise


def test_global_kill():
    """Global kill must block ALL agents."""
    ks = KillSwitch()
    ks.kill_all("emergency")
    with pytest.raises(AgentKilled, match="emergency"):
        ks.check("any_run_id")


def test_revive():
    """Revived agent must pass check."""
    ks = KillSwitch()
    ks.kill("run_1", reason="test")
    ks.revive("run_1")
    ks.check("run_1")  # should not raise


def test_revive_all():
    """revive_all must clear all kills and global kill."""
    ks = KillSwitch()
    ks.kill("run_1")
    ks.kill("run_2")
    ks.kill_all("global")
    ks.revive_all()
    ks.check("run_1")  # should not raise
    ks.check("run_2")  # should not raise


def test_is_killed():
    """is_killed must return correct status without raising."""
    ks = KillSwitch()
    assert ks.is_killed("run_1") is False
    ks.kill("run_1")
    assert ks.is_killed("run_1") is True
    ks.revive("run_1")
    assert ks.is_killed("run_1") is False


def test_auto_kill_cost():
    """Auto-kill must trigger when cost exceeds threshold."""
    ks = KillSwitch()
    ks.set_auto_kill(max_cost_usd=5.0)
    ks.check_auto_kill("run_1", cost_usd=3.0, calls=10)  # under threshold
    ks.check("run_1")  # should pass

    ks.check_auto_kill("run_1", cost_usd=6.0, calls=20)  # over threshold
    with pytest.raises(AgentKilled):
        ks.check("run_1")


def test_auto_kill_calls():
    """Auto-kill must trigger when calls exceed threshold."""
    ks = KillSwitch()
    ks.set_auto_kill(max_calls=100)
    ks.check_auto_kill("run_1", cost_usd=1.0, calls=50)
    ks.check("run_1")  # should pass

    ks.check_auto_kill("run_1", cost_usd=2.0, calls=150)
    with pytest.raises(AgentKilled):
        ks.check("run_1")


def test_on_kill_callback():
    """Kill callbacks must be fired."""
    events = []
    ks = KillSwitch()
    ks.on_kill(lambda run_id, reason: events.append((run_id, reason)))
    ks.kill("run_1", "test callback")
    assert len(events) == 1
    assert events[0] == ("run_1", "test callback")


def test_get_killed_agents():
    """get_killed_agents must return current state."""
    ks = KillSwitch()
    ks.kill("run_1", "reason_1")
    ks.kill("run_2", "reason_2")
    report = ks.get_killed_agents()
    assert "run_1" in report["killed_runs"]
    assert "run_2" in report["killed_runs"]
    assert report["global_kill"] is False


def test_killed_exception_fields():
    """AgentKilled exception must have correct fields."""
    ks = KillSwitch()
    ks.kill("run_x", reason="safety concern")
    try:
        ks.check("run_x")
    except AgentKilled as e:
        assert e.run_id == "run_x"
        assert e.reason == "safety concern"
        assert e.killed_at > 0


def test_gateway_integration():
    """Kill switch must block completion() calls."""
    from agentfuse.gateway import completion
    from agentfuse.core.kill_switch import kill_switch as global_ks

    global_ks.kill("blocked_run", "test gateway block")
    with pytest.raises(AgentKilled):
        completion(model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
                   budget_id="blocked_run")
    global_ks.revive("blocked_run")


def test_auto_kill_duration():
    """Auto-kill must trigger when duration exceeds threshold."""
    import time
    ks = KillSwitch()
    ks.set_auto_kill(max_duration_seconds=0.01)
    started = time.time() - 1.0  # started 1 second ago
    ks.check_auto_kill("run_1", cost_usd=0.0, calls=1, started_at=started)
    with pytest.raises(AgentKilled):
        ks.check("run_1")


def test_callback_error_doesnt_crash():
    """Failing on_kill callback must not crash."""
    ks = KillSwitch()
    ks.on_kill(lambda run_id, reason: 1/0)  # will crash
    ks.kill("run_err", "test")  # should not raise


def test_no_auto_kill_when_not_configured():
    """check_auto_kill with no config must do nothing."""
    ks = KillSwitch()
    ks.check_auto_kill("run_1", cost_usd=1000.0, calls=999999)
    ks.check("run_1")  # should not raise
