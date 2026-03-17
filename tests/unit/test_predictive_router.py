"""
Tests for CostPredictiveRouter — novel predictive model routing.
"""

from agentfuse.core.predictive_router import CostPredictiveRouter


def test_no_routing_below_threshold():
    """Below preemptive threshold, model must not change."""
    router = CostPredictiveRouter(budget_usd=10.0, preemptive_threshold=0.70)
    # Only 30% spent — should not route
    for _ in range(3):
        router.record_cost(1.0)  # $3 of $10 = 30%
    result = router.predict_and_route("gpt-5.4")
    assert result == "gpt-5.4"


def test_routing_when_budget_running_low():
    """When predicted exhaustion is within lookahead, must downgrade."""
    router = CostPredictiveRouter(budget_usd=10.0, lookahead_calls=5, preemptive_threshold=0.70)
    # Spend $8 in 4 calls ($2 each) — only $2 left ≈ 1 call at current rate
    for _ in range(4):
        router.record_cost(2.0)
    result = router.predict_and_route("gpt-5.4")
    assert result != "gpt-5.4"  # should downgrade


def test_weighted_avg_favors_recent():
    """Weighted average must give more weight to recent costs."""
    router = CostPredictiveRouter(budget_usd=100.0)
    # Old costs: cheap
    for _ in range(5):
        router.record_cost(0.01)
    # Recent costs: expensive
    for _ in range(5):
        router.record_cost(1.0)
    avg = router._weighted_avg_cost()
    # Weighted avg should be closer to 1.0 than 0.505 (simple avg)
    assert avg > 0.5


def test_cost_trend_detection():
    """Rising costs must produce positive trend."""
    router = CostPredictiveRouter(budget_usd=100.0)
    # Costs rising
    for i in range(10):
        router.record_cost(0.01 * (i + 1))
    trend = router._cost_trend()
    assert trend > 0


def test_cost_trend_flat():
    """Flat costs must produce near-zero trend."""
    router = CostPredictiveRouter(budget_usd=100.0)
    for _ in range(10):
        router.record_cost(0.05)
    trend = router._cost_trend()
    assert abs(trend) < 0.01


def test_same_family_detection():
    """Same-family models must be detected correctly."""
    assert CostPredictiveRouter._same_family("gpt-5.4", "gpt-4.1-mini") is True
    assert CostPredictiveRouter._same_family("claude-opus-4-6", "claude-haiku-4-5-20251001") is True
    assert CostPredictiveRouter._same_family("gpt-5.4", "claude-sonnet-4-6") is False


def test_get_prediction_structure():
    """get_prediction must return expected keys."""
    router = CostPredictiveRouter(budget_usd=10.0)
    router.record_cost(1.0)
    pred = router.get_prediction()
    assert "total_spent" in pred
    assert "budget_usd" in pred
    assert "pct_spent" in pred
    assert "predicted_remaining_calls" in pred
    assert "cost_trend" in pred
    assert pred["total_spent"] == 1.0


def test_finds_cheaper_same_family():
    """Downgrade must prefer same provider family."""
    router = CostPredictiveRouter(budget_usd=10.0)
    result = router._find_cheaper_model("gpt-5.4")
    # Should pick a GPT model, not Claude
    assert result is not None
    assert result.startswith("gpt") or result.startswith("o")


def test_finds_cheaper_anthropic():
    """Anthropic downgrade must find Haiku."""
    router = CostPredictiveRouter(budget_usd=10.0)
    result = router._find_cheaper_model("claude-sonnet-4-6")
    assert result == "claude-haiku-4-5-20251001"


def test_unknown_model_no_crash():
    """Unknown model must not crash."""
    router = CostPredictiveRouter(budget_usd=10.0)
    for _ in range(5):
        router.record_cost(2.0)
    result = router.predict_and_route("unknown-model")
    assert result == "unknown-model"  # no downgrade available


def test_rising_cost_trend_triggers_downgrade():
    """Rising costs with >60% spent must trigger early downgrade."""
    router = CostPredictiveRouter(budget_usd=10.0, preemptive_threshold=0.60)
    # Simulate rising costs: 0.5, 1.0, 1.5, 2.0, 2.5
    for i in range(5):
        router.record_cost(0.5 * (i + 1))
    # Total spent: 7.5 (75%), trend is positive
    result = router.predict_and_route("gpt-5.4")
    assert result != "gpt-5.4"  # should downgrade due to rising costs


def test_prediction_with_no_data():
    """predict_and_route with no data must return original model."""
    router = CostPredictiveRouter(budget_usd=10.0)
    result = router.predict_and_route("gpt-4o")
    assert result == "gpt-4o"


def test_tier2_downgrade_finds_tier3():
    """Tier 2 model must downgrade to a tier 3 model."""
    router = CostPredictiveRouter(budget_usd=10.0)
    result = router._find_cheaper_model("gpt-4.1")
    assert result is not None
    assert result == "gpt-4.1-mini"  # same family preferred
