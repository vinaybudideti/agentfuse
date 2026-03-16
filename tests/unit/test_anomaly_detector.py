"""
Tests for CostAnomalyDetector — statistical anomaly detection.
"""

from agentfuse.core.anomaly import CostAnomalyDetector, AnomalyReport


def test_no_anomaly_during_warmup():
    """First min_samples calls must not flag anomalies (warming up)."""
    detector = CostAnomalyDetector(min_samples=5)
    for i in range(5):
        result = detector.record("gpt-4o", 0.01, run_id=f"run_{i}")
        assert result is None


def test_normal_cost_no_anomaly():
    """Costs within normal range must not be flagged."""
    detector = CostAnomalyDetector(min_samples=5, z_threshold=3.0)
    # Warm up with varied costs to build realistic variance
    costs = [0.008, 0.010, 0.012, 0.009, 0.011, 0.010, 0.012, 0.009, 0.010, 0.011,
             0.010, 0.009, 0.011, 0.010, 0.012, 0.010, 0.011, 0.009, 0.010, 0.010]
    for i, c in enumerate(costs):
        detector.record("gpt-4o", c, run_id=f"warmup_{i}")

    # A cost within the normal range should not trigger
    result = detector.record("gpt-4o", 0.011, run_id="normal")
    assert result is None


def test_spike_detected_as_anomaly():
    """A cost 10x above baseline must be detected as anomaly."""
    detector = CostAnomalyDetector(min_samples=5, z_threshold=2.0)
    # Warm up with $0.01 per call
    for i in range(10):
        detector.record("gpt-4o", 0.01, run_id=f"warmup_{i}")

    # Sudden spike to $1.00 (100x normal)
    result = detector.record("gpt-4o", 1.00, run_id="spike")
    assert result is not None
    assert result.severity in ("warning", "critical")
    assert result.z_score > 2.0
    assert result.cost_usd == 1.00


def test_critical_severity():
    """Very large spikes must be classified as critical."""
    detector = CostAnomalyDetector(min_samples=5, z_threshold=2.0, critical_threshold=4.0)
    for i in range(10):
        detector.record("gpt-4o", 0.01, run_id=f"warmup_{i}")

    result = detector.record("gpt-4o", 10.0, run_id="critical_spike")
    assert result is not None
    assert result.severity == "critical"


def test_per_model_isolation():
    """Anomaly detection must be per-model (not global)."""
    detector = CostAnomalyDetector(min_samples=3)
    # gpt-4o baseline: $0.01
    for i in range(5):
        detector.record("gpt-4o", 0.01)
    # claude baseline: $0.10 (10x more expensive)
    for i in range(5):
        detector.record("claude-sonnet-4-6", 0.10)

    # $0.10 is normal for Claude but anomalous for GPT-4o
    result_gpt = detector.record("gpt-4o", 0.10)
    result_claude = detector.record("claude-sonnet-4-6", 0.10)
    assert result_gpt is not None  # anomaly for gpt-4o
    assert result_claude is None  # normal for claude


def test_baseline_accessible():
    """get_baseline must return current statistics."""
    detector = CostAnomalyDetector(min_samples=3)
    for i in range(5):
        detector.record("gpt-4o", 0.01)

    baseline = detector.get_baseline("gpt-4o")
    assert baseline is not None
    assert baseline["samples"] == 5
    assert baseline["mean_cost"] > 0


def test_callback_fired_on_anomaly():
    """Callback must be called when anomaly detected."""
    reports = []
    detector = CostAnomalyDetector(
        min_samples=3, z_threshold=2.0,
        callback=lambda r: reports.append(r),
    )
    for i in range(5):
        detector.record("gpt-4o", 0.01)

    detector.record("gpt-4o", 5.0)
    assert len(reports) == 1
    assert reports[0].severity in ("warning", "critical")


def test_callback_failure_doesnt_crash():
    """Failed callback must not crash anomaly detection."""
    def bad_cb(r):
        raise RuntimeError("webhook down")

    detector = CostAnomalyDetector(min_samples=3, z_threshold=2.0, callback=bad_cb)
    for i in range(5):
        detector.record("gpt-4o", 0.01)

    # Must not raise
    result = detector.record("gpt-4o", 5.0)
    assert result is not None


def test_reset():
    """Reset must clear all state."""
    detector = CostAnomalyDetector(min_samples=3)
    for i in range(5):
        detector.record("gpt-4o", 0.01)

    detector.reset("gpt-4o")
    assert detector.get_baseline("gpt-4o") is None


def test_zero_cost_ignored():
    """Zero or negative costs must be silently ignored."""
    detector = CostAnomalyDetector(min_samples=3)
    result = detector.record("gpt-4o", 0.0)
    assert result is None
    result = detector.record("gpt-4o", -0.5)
    assert result is None
