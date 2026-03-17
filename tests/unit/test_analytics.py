"""
Tests for UsageAnalytics — spend data insights.
"""

from agentfuse.core.analytics import UsageAnalytics


def _sample_report():
    return {
        "total_usd": 15.50,
        "by_model": {
            "gpt-4o": 8.00,
            "claude-sonnet-4-6": 5.00,
            "gpt-4o-mini": 2.50,
        },
        "by_provider": {
            "openai": 10.50,
            "anthropic": 5.00,
        },
        "by_run": {
            "run_1": 5.00,
            "run_2": 3.00,
            "run_3": 7.50,
        },
    }


def test_insights_structure():
    """Insights must contain expected keys."""
    analytics = UsageAnalytics(_sample_report())
    insights = analytics.get_insights()
    assert "total_spend_usd" in insights
    assert "most_expensive_model" in insights
    assert "provider_breakdown_pct" in insights
    assert "recommendations" in insights


def test_most_expensive_model():
    """Must correctly identify the most expensive model."""
    analytics = UsageAnalytics(_sample_report())
    insights = analytics.get_insights()
    assert insights["most_expensive_model"] == "gpt-4o"


def test_provider_breakdown():
    """Provider percentages must sum to ~100%."""
    analytics = UsageAnalytics(_sample_report())
    insights = analytics.get_insights()
    total_pct = sum(insights["provider_breakdown_pct"].values())
    assert 99 < total_pct < 101


def test_model_efficiency():
    """Model efficiency ranking must be sorted by cost."""
    analytics = UsageAnalytics(_sample_report())
    efficiency = analytics.get_model_efficiency()
    models = list(efficiency.keys())
    assert models[0] == "gpt-4o"  # most expensive first


def test_budget_utilization():
    """Budget utilization must calculate correct stats."""
    analytics = UsageAnalytics(_sample_report())
    util = analytics.get_budget_utilization()
    assert util["runs"] == 3
    assert util["max_run_cost"] == 7.50
    assert util["min_run_cost"] == 3.00


def test_empty_report():
    """Empty report must not crash."""
    analytics = UsageAnalytics({})
    insights = analytics.get_insights()
    assert insights["total_spend_usd"] == 0
    assert insights["model_count"] == 0


def test_recommendations_for_expensive_model():
    """Using expensive models should generate recommendations."""
    report = {
        "total_usd": 100.0,
        "by_model": {"gpt-5.4": 80.0, "gpt-4o-mini": 20.0},
        "by_provider": {"openai": 100.0},
        "by_run": {},
    }
    analytics = UsageAnalytics(report)
    insights = analytics.get_insights()
    assert any("gpt-5.4" in r for r in insights["recommendations"])


def test_single_model_recommendation():
    """Using only 1 model should recommend auto_route."""
    report = {
        "total_usd": 10.0,
        "by_model": {"gpt-4o": 10.0},
        "by_provider": {"openai": 10.0},
        "by_run": {},
    }
    analytics = UsageAnalytics(report)
    insights = analytics.get_insights()
    assert any("auto_route" in r for r in insights["recommendations"])
