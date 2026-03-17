"""Tests for CostForecast."""

from agentfuse.core.cost_forecast import CostForecast


def test_monthly_prediction():
    """Monthly prediction must extrapolate from daily average."""
    report = {"total_usd": 10.0, "by_model": {"gpt-4o": 10.0}, "by_provider": {"openai": 10.0}}
    forecast = CostForecast(report, days_of_data=1.0)
    result = forecast.predict_monthly()
    assert result["monthly_estimate_usd"] == 300.0  # $10/day * 30
    assert result["annual_estimate_usd"] == 3600.0


def test_weekly_prediction():
    """Weekly prediction from 7 days of data."""
    report = {"total_usd": 70.0, "by_model": {"gpt-4o": 70.0}, "by_provider": {}}
    forecast = CostForecast(report, days_of_data=7.0)
    result = forecast.predict_monthly()
    assert result["daily_average_usd"] == 10.0
    assert result["weekly_estimate_usd"] == 70.0


def test_optimization_opportunities():
    """Optimization estimates must be less than monthly estimate."""
    report = {"total_usd": 100.0, "by_model": {"gpt-5": 100.0}, "by_provider": {}}
    forecast = CostForecast(report, days_of_data=1.0)
    result = forecast.predict_monthly()
    monthly = result["monthly_estimate_usd"]
    assert result["optimization_opportunities"]["with_better_caching"] < monthly
    assert result["optimization_opportunities"]["with_intelligent_routing"] < monthly
    assert result["optimization_opportunities"]["with_all_optimizations"] < monthly


def test_budget_duration():
    """Budget duration must predict days remaining."""
    report = {"total_usd": 10.0, "by_model": {}, "by_provider": {}}
    forecast = CostForecast(report, days_of_data=1.0)
    result = forecast.predict_budget_duration(budget_usd=100.0)
    assert result["days_remaining"] == 10.0
    assert result["will_last_month"] is False


def test_empty_report():
    """Empty report must not crash."""
    forecast = CostForecast({}, days_of_data=1.0)
    result = forecast.predict_monthly()
    assert result["monthly_estimate_usd"] == 0.0


def test_zero_days():
    """Zero days must not crash (prevented internally)."""
    forecast = CostForecast({"total_usd": 10.0}, days_of_data=0)
    result = forecast.predict_monthly()
    assert result["monthly_estimate_usd"] > 0
