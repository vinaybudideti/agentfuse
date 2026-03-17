"""
Tests for CostAlertManager — threshold-based cost alerts.
"""

from agentfuse.core.cost_alert import CostAlertManager, CostAlert
from agentfuse.core.budget import BudgetEngine


def test_alert_fires_at_50_pct():
    """Alert must fire when spend crosses 50% threshold."""
    alerts = []
    manager = CostAlertManager(
        thresholds=[0.50],
        callback=lambda a: alerts.append(a),
    )
    engine = BudgetEngine("run_alert", 10.0, "gpt-4o")
    engine.spent = 5.0  # 50%
    result = manager.check(engine)
    assert result is not None
    assert result.threshold_pct == 0.50
    assert len(alerts) == 1


def test_alert_fires_only_once_per_threshold():
    """Same threshold must not fire twice for same run."""
    alerts = []
    manager = CostAlertManager(
        thresholds=[0.50],
        callback=lambda a: alerts.append(a),
    )
    engine = BudgetEngine("run_dedup", 10.0, "gpt-4o")
    engine.spent = 5.0
    manager.check(engine)
    engine.spent = 6.0
    result = manager.check(engine)
    assert result is None  # already fired
    assert len(alerts) == 1


def test_multiple_thresholds():
    """Multiple thresholds must fire in order."""
    alerts = []
    manager = CostAlertManager(
        thresholds=[0.50, 0.75, 0.90],
        callback=lambda a: alerts.append(a),
    )
    engine = BudgetEngine("run_multi", 10.0, "gpt-4o")

    engine.spent = 5.0
    manager.check(engine)
    assert len(alerts) == 1
    assert alerts[-1].threshold_pct == 0.50

    engine.spent = 7.5
    manager.check(engine)
    assert len(alerts) == 2
    assert alerts[-1].threshold_pct == 0.75

    engine.spent = 9.0
    manager.check(engine)
    assert len(alerts) == 3
    assert alerts[-1].threshold_pct == 0.90


def test_no_alert_below_threshold():
    """No alert must fire below the lowest threshold."""
    alerts = []
    manager = CostAlertManager(
        thresholds=[0.50],
        callback=lambda a: alerts.append(a),
    )
    engine = BudgetEngine("run_below", 10.0, "gpt-4o")
    engine.spent = 4.9
    result = manager.check(engine)
    assert result is None
    assert len(alerts) == 0


def test_alert_payload_correct():
    """Alert payload must contain correct data."""
    manager = CostAlertManager(thresholds=[0.50])
    engine = BudgetEngine("run_payload", 10.0, "gpt-4o")
    engine.spent = 6.0  # 60%

    alert = manager.check(engine)
    assert alert.run_id == "run_payload"
    assert alert.threshold_pct == 0.50
    assert abs(alert.current_pct - 0.60) < 0.01
    assert alert.spent_usd == 6.0
    assert alert.budget_usd == 10.0
    assert alert.model == "gpt-4o"


def test_alert_to_dict():
    """CostAlert.to_dict must return a serializable dict."""
    alert = CostAlert(
        alert_type="threshold",
        tenant_id="t1",
        run_id="r1",
        threshold_pct=0.50,
        current_pct=0.55,
        spent_usd=5.5,
        budget_usd=10.0,
        model="gpt-4o",
    )
    d = alert.to_dict()
    assert d["alert_type"] == "threshold"
    assert d["spent_usd"] == 5.5


def test_alert_to_json():
    """CostAlert.to_json must return valid JSON."""
    import json
    alert = CostAlert(
        alert_type="threshold",
        tenant_id="t1",
        run_id="r1",
        threshold_pct=0.50,
        current_pct=0.55,
        spent_usd=5.5,
        budget_usd=10.0,
        model="gpt-4o",
    )
    j = alert.to_json()
    parsed = json.loads(j)
    assert parsed["run_id"] == "r1"


def test_reset_specific_run():
    """Reset with run_id must only clear that run's fired alerts."""
    alerts = []
    manager = CostAlertManager(
        thresholds=[0.50],
        callback=lambda a: alerts.append(a),
    )

    engine1 = BudgetEngine("run_1", 10.0, "gpt-4o")
    engine2 = BudgetEngine("run_2", 10.0, "gpt-4o")
    engine1.spent = 5.0
    engine2.spent = 5.0

    manager.check(engine1)
    manager.check(engine2)
    assert len(alerts) == 2

    # Reset only run_1
    manager.reset(run_id="run_1")

    # run_1 alert should fire again
    result = manager.check(engine1)
    assert result is not None
    assert len(alerts) == 3

    # run_2 alert should NOT fire again
    result = manager.check(engine2)
    assert result is None


def test_reset_all():
    """Reset without run_id must clear all fired alerts."""
    alerts = []
    manager = CostAlertManager(
        thresholds=[0.50],
        callback=lambda a: alerts.append(a),
    )
    engine = BudgetEngine("run_all", 10.0, "gpt-4o")
    engine.spent = 5.0
    manager.check(engine)
    assert len(alerts) == 1

    manager.reset()
    manager.check(engine)
    assert len(alerts) == 2  # fired again after reset


def test_callback_error_doesnt_crash():
    """Failing callback must not crash the check."""
    def bad_callback(alert):
        raise RuntimeError("callback error")

    manager = CostAlertManager(
        thresholds=[0.50],
        callback=bad_callback,
    )
    engine = BudgetEngine("run_err", 10.0, "gpt-4o")
    engine.spent = 5.0
    # Should not raise
    result = manager.check(engine)
    assert result is not None
