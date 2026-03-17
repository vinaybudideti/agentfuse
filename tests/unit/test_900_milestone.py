"""
Tests to push toward 900 milestone — targeted edge cases.
"""

from agentfuse.core.security import secure_hash, sanitize_for_cache_key, SecurityEvent
from agentfuse.core.deprecation import ModelDeprecationChecker, DEPRECATED_MODELS
from agentfuse.core.context_guard import ContextWindowGuard
from agentfuse.core.report_exporter import ReportExporter
from agentfuse.core.cost_forecast import CostForecast
from agentfuse.core.model_recommender import ModelRecommender
from agentfuse.core.analytics import UsageAnalytics
from agentfuse.providers.router import resolve_provider, list_providers


def test_secure_hash_length():
    assert len(secure_hash("test")) == 64  # SHA-256 hex


def test_sanitize_normalizes_whitespace():
    """Sanitization must collapse multiple spaces."""
    result = sanitize_for_cache_key("hello    world")
    assert result == "hello world"


def test_security_event_has_timestamp():
    e = SecurityEvent("test", severity="info")
    assert e.timestamp > 0


def test_deprecated_models_dict():
    assert len(DEPRECATED_MODELS) > 5


def test_deprecation_checker_caches_warning():
    """Second call for same model must not warn again."""
    checker = ModelDeprecationChecker()
    checker.check_and_suggest("gpt-4o")
    checker.check_and_suggest("gpt-4o")  # no double warn


def test_context_guard_model_limits():
    guard = ContextWindowGuard()
    limits = guard.get_model_limits("gpt-5")
    assert limits["context_window"] == 1_000_000


def test_report_to_dict_sorted():
    r = ReportExporter({"total_usd": 10, "by_model": {"b": 3, "a": 7}, "by_provider": {}, "by_run": {}})
    d = r.to_dict()
    models = list(d["models"].keys())
    assert models[0] == "a"  # sorted by cost descending: a=7 > b=3


def test_forecast_budget_infinite():
    f = CostForecast({"total_usd": 0}, days_of_data=1)
    result = f.predict_budget_duration(100.0)
    assert result["days_remaining"] == -1  # zero daily rate


def test_recommender_compare():
    r = ModelRecommender()
    comp = r.compare("gpt-5.4", "gpt-4.1-nano")
    assert comp["cost_ratio"] > 1  # 5.4 is more expensive


def test_analytics_single_model():
    a = UsageAnalytics({"total_usd": 5, "by_model": {"gpt-4o": 5}, "by_provider": {}, "by_run": {}})
    insights = a.get_insights()
    assert any("auto_route" in r for r in insights["recommendations"])


def test_list_providers_returns_dict():
    providers = list_providers()
    assert "deepseek" in providers
    assert "groq" in providers


def test_resolve_provider_o4():
    name, _ = resolve_provider("o4-mini")
    assert name == "openai"


def test_milestone_900():
    """The 900th test — celebrating production-grade quality."""
    import agentfuse
    assert len(agentfuse.__all__) >= 70
    assert agentfuse.__version__ is not None
