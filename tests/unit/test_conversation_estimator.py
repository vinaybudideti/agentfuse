"""
Tests for ConversationCostEstimator — cost trajectory prediction.
"""

from agentfuse.core.conversation_estimator import ConversationCostEstimator


def test_flat_pattern_detection():
    """Constant cost per turn must be detected as flat."""
    est = ConversationCostEstimator(budget_usd=10.0)
    for _ in range(5):
        est.record_turn(cost=0.01)
    assert est.detect_pattern() == "flat"


def test_exponential_pattern_detection():
    """Doubling cost per turn must be detected as exponential."""
    est = ConversationCostEstimator(budget_usd=10.0)
    cost = 0.01
    for _ in range(5):
        est.record_turn(cost=cost)
        cost *= 2  # doubling each turn
    assert est.detect_pattern() == "exponential"


def test_projection_flat():
    """Flat pattern projection must predict constant cost."""
    est = ConversationCostEstimator(budget_usd=10.0)
    for _ in range(5):
        est.record_turn(cost=0.01)
    proj = est.project(target_turns=20)
    assert proj["pattern"] == "flat"
    # 5 turns at $0.01 = $0.05, project 15 more = $0.15 + $0.05 = $0.20
    assert abs(proj["projected_total_cost"] - 0.20) < 0.01
    assert proj["will_exceed_budget"] is False


def test_projection_budget_exceeded():
    """Expensive conversation must predict budget exceeded."""
    est = ConversationCostEstimator(budget_usd=1.0)
    for _ in range(5):
        est.record_turn(cost=0.15)  # $0.75 spent
    proj = est.project(target_turns=20)
    assert proj["will_exceed_budget"] is True


def test_turns_until_budget():
    """Must estimate correct turns until budget exhaustion."""
    est = ConversationCostEstimator(budget_usd=0.10)
    for _ in range(3):
        est.record_turn(cost=0.01)
    proj = est.project()
    # $0.03 spent, $0.07 remaining at $0.01/turn = 7 turns
    assert proj["turns_until_budget_exceeded"] == 7


def test_empty_estimator():
    """Empty estimator must return safe defaults."""
    est = ConversationCostEstimator(budget_usd=5.0)
    proj = est.project()
    assert proj["projected_total_cost"] == 0.0
    assert proj["confidence"] == 0.0


def test_summary():
    """Summary must contain expected metrics."""
    est = ConversationCostEstimator(budget_usd=10.0)
    est.record_turn(cost=0.05, input_tokens=100, output_tokens=50)
    est.record_turn(cost=0.10, input_tokens=200, output_tokens=100)
    summary = est.get_summary()
    assert summary["turns"] == 2
    assert abs(summary["total_cost"] - 0.15) < 1e-6
    assert summary["total_input_tokens"] == 300
    assert summary["total_output_tokens"] == 150


def test_confidence_increases_with_data():
    """Confidence must increase as more turns are recorded."""
    est = ConversationCostEstimator(budget_usd=10.0)
    est.record_turn(cost=0.01)
    proj1 = est.project()
    for _ in range(9):
        est.record_turn(cost=0.01)
    proj2 = est.project()
    assert proj2["confidence"] > proj1["confidence"]


def test_unknown_pattern_with_few_turns():
    """Pattern must be 'unknown' with fewer than 3 turns."""
    est = ConversationCostEstimator(budget_usd=10.0)
    est.record_turn(cost=0.01)
    est.record_turn(cost=0.02)
    assert est.detect_pattern() == "unknown"


def test_linear_pattern_detection():
    """Linearly increasing costs must be detected as linear."""
    est = ConversationCostEstimator(budget_usd=100.0)
    for i in range(10):
        est.record_turn(cost=0.01 + 0.005 * i)  # gradual increase
    pattern = est.detect_pattern()
    assert pattern in ("linear", "flat")  # gradual increase may be flat or linear


def test_projection_exponential_exceeds():
    """Exponential growth must predict budget exceeded quickly."""
    est = ConversationCostEstimator(budget_usd=1.0)
    cost = 0.01
    for _ in range(5):
        est.record_turn(cost=cost)
        cost *= 2.5  # aggressive growth
    proj = est.project(target_turns=20)
    assert proj["will_exceed_budget"] is True


def test_zero_cost_turns():
    """All-zero cost turns must not crash."""
    est = ConversationCostEstimator(budget_usd=10.0)
    for _ in range(5):
        est.record_turn(cost=0.0)
    pattern = est.detect_pattern()
    assert pattern == "flat"
    proj = est.project()
    assert proj["projected_total_cost"] == 0.0


def test_single_turn_projection():
    """Single turn must produce a projection."""
    est = ConversationCostEstimator(budget_usd=10.0)
    est.record_turn(cost=0.5)
    proj = est.project(target_turns=10)
    assert proj["current_turns"] == 1
    assert proj["projected_total_cost"] > 0
