"""
End-to-end tests — simulate realistic production usage patterns.
"""

from unittest.mock import patch
from types import SimpleNamespace

from agentfuse.gateway import completion, cleanup, get_spend, _cache
from agentfuse.core.session import AgentSession
from agentfuse.core.hierarchical_budget import HierarchicalBudget
from agentfuse.core.tool_cost_tracker import ToolCostTracker
from agentfuse.core.guardrails import ContentGuardrails
from agentfuse.core.model_recommender import ModelRecommender
from agentfuse.core.cost_forecast import CostForecast
from agentfuse.core.report_exporter import ReportExporter


def setup_function():
    cleanup()


def test_e2e_session_with_cache():
    """Realistic session: multiple calls, some cached."""
    msgs1 = [{"role": "user", "content": "e2e test query A unique 111"}]
    msgs2 = [{"role": "user", "content": "e2e test query B unique 222"}]
    _cache.store(model="gpt-4o", messages=msgs1, response="Cached A")
    _cache.store(model="gpt-4o", messages=msgs2, response="Cached B")

    with AgentSession("e2e_session", budget_usd=10.0) as session:
        r1 = session.completion(messages=msgs1)
        r2 = session.completion(messages=msgs2)
        assert r1.choices[0].message.content == "Cached A"
        assert r2.choices[0].message.content == "Cached B"
        assert session._call_count == 2
        assert session._cache_hits >= 2


def test_e2e_hierarchical_multi_agent():
    """Realistic multi-agent: parent allocates to 3 children."""
    parent = HierarchicalBudget("research_project", total_usd=20.0)
    researcher = parent.allocate_child("researcher", budget_usd=8.0, model="gpt-5")
    writer = parent.allocate_child("writer", budget_usd=8.0, model="claude-sonnet-4-6")
    reviewer = parent.allocate_child("reviewer", budget_usd=4.0, model="gpt-4.1-mini")

    # Simulate work
    researcher.engine.record_cost(3.0)
    writer.engine.record_cost(5.0)
    reviewer.engine.record_cost(1.0)

    assert parent.total_spent == 9.0
    report = parent.get_report()
    assert report["children"]["researcher"]["spent_usd"] == 3.0
    assert report["children"]["writer"]["spent_usd"] == 5.0


def test_e2e_tool_tracking_with_budget():
    """Realistic tool usage: web searches + code execution with budget."""
    from agentfuse.core.budget import BudgetEngine
    engine = BudgetEngine("tool_session", 5.0, "gpt-4o")
    tracker = ToolCostTracker(budget_engine=engine)
    tracker.register_tool("web_search")  # auto-detect cost
    tracker.register_tool("code_exec", cost_per_second=0.001)

    # 10 web searches + 30 seconds of code execution
    for _ in range(10):
        tracker.record_tool_call("web_search")
    tracker.record_tool_call("code_exec", duration_seconds=30.0)

    report = tracker.get_report()
    assert report["total_tool_calls"] == 11
    assert report["total_tool_spend"] > 0.1  # 10 × $0.01 + 30 × $0.001


def test_e2e_guardrails_before_cache():
    """Realistic: validate response → only cache if it passes guardrails."""
    guardrails = ContentGuardrails()
    guardrails.add_rule("no_pii", patterns=["email", "phone"])
    guardrails.add_rule("max_length", max_chars=5000)

    good_response = "Python is a programming language."
    bad_response = "Contact john@example.com for details."

    assert guardrails.validate(good_response).passed is True
    assert guardrails.validate(bad_response).passed is False


def test_e2e_model_recommendation_flow():
    """Realistic: recommend model → estimate cost → decide."""
    recommender = ModelRecommender()

    # Low budget: recommend cheap model
    cheap_model = recommender.recommend(budget_remaining=0.50, task_type="factual")
    assert cheap_model in recommender._profiles

    # Estimate cost for the recommended model
    from agentfuse.gateway import estimate_cost
    est = estimate_cost(cheap_model, [{"role": "user", "content": "What is 2+2?"}])
    assert est["estimated_total_cost_usd"] < 0.05  # cheap model should be cheap


def test_e2e_forecast_and_report():
    """Realistic: generate forecast → export report."""
    report = {
        "total_usd": 150.0,
        "by_model": {"gpt-5": 80.0, "gpt-4.1-mini": 50.0, "claude-sonnet-4-6": 20.0},
        "by_provider": {"openai": 130.0, "anthropic": 20.0},
        "by_run": {"agent_1": 60.0, "agent_2": 50.0, "agent_3": 40.0},
    }

    # Forecast
    forecast = CostForecast(report, days_of_data=7.0)
    prediction = forecast.predict_monthly()
    assert prediction["monthly_estimate_usd"] > 0

    # Export
    exporter = ReportExporter(report)
    summary = exporter.to_summary()
    assert "$150.00" in summary
    csv = exporter.to_csv()
    assert "gpt-5" in csv
    json_out = exporter.to_json()
    assert "gpt-5" in json_out


@patch("agentfuse.gateway._call_openai_compatible")
def test_e2e_full_session_lifecycle(mock_call):
    """Full lifecycle: session → calls → tools → receipt."""
    mock_call.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Session response"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=100),
    )

    with AgentSession("lifecycle", budget_usd=10.0) as session:
        session.completion(messages=[{"role": "user", "content": "lifecycle test"}])
        session.record_tool_call("web_search", cost=0.01)

    receipt = session.get_receipt()
    assert receipt["calls"] == 1
    assert receipt["tool_cost_usd"] == 0.01
    assert receipt["duration_seconds"] >= 0


def test_e2e_kill_switch_blocks_session():
    """Kill switch must block session completion calls."""
    from agentfuse.core.kill_switch import kill_switch, AgentKilled
    import pytest

    kill_switch.kill("killed_session", "e2e test")
    with pytest.raises(AgentKilled):
        with AgentSession("killed_session", budget_usd=5.0, run_id="killed_session") as session:
            session.completion(messages=[{"role": "user", "content": "blocked"}])
    kill_switch.revive("killed_session")
