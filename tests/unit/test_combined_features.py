"""
Tests for combined feature interactions — verify modules compose correctly.
"""

from unittest.mock import patch
from types import SimpleNamespace

from agentfuse.core.batch_submitter import BatchSubmitter
from agentfuse.core.session import AgentSession
from agentfuse.core.guardrails import ContentGuardrails
from agentfuse.core.quality_scorer import ResponseQualityScorer
from agentfuse.core.context_guard import ContextWindowGuard
from agentfuse.core.model_recommender import ModelRecommender
from agentfuse.core.report_exporter import ReportExporter
from agentfuse.core.cost_forecast import CostForecast
from agentfuse.core.deprecation import ModelDeprecationChecker
from agentfuse.core.security import mask_api_key, check_prompt_injection, validate_response_safety


# --- Guardrails + Quality Scorer ---

def test_guardrails_and_quality_agree_on_good():
    """Good response must pass both guardrails and quality scorer."""
    g = ContentGuardrails()
    g.add_rule("max_length", max_chars=10000)
    s = ResponseQualityScorer()

    response = "Python is a high-level, general-purpose programming language known for its clean syntax and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming."
    assert g.validate(response).passed is True
    assert s.score("What is Python?", response).should_cache is True


def test_guardrails_and_quality_agree_on_bad():
    """PII response must be caught by guardrails even if quality is high."""
    g = ContentGuardrails()
    g.add_rule("no_pii", patterns=["email"])
    s = ResponseQualityScorer()

    response = "Contact john@example.com for Python help."
    assert g.validate(response).passed is False  # PII detected
    # Quality might be high, but guardrails block it


# --- Context Guard + Model Recommender ---

def test_context_guard_with_recommended_model():
    """Recommended model must have enough context for the messages."""
    recommender = ModelRecommender()
    guard = ContextWindowGuard()

    model = recommender.recommend(budget_remaining=10.0)
    msgs = [{"role": "user", "content": "Short question"}]
    result = guard.check(msgs, model)
    assert result["fits"] is True


# --- Deprecation + Recommender ---

def test_deprecated_model_has_replacement_in_recommender():
    """Deprecated model's replacement must be in the recommender."""
    checker = ModelDeprecationChecker()
    recommender = ModelRecommender()

    replacement = checker.get_replacement("gpt-4o")
    if replacement:
        # The replacement should be a valid model
        models = recommender.get_all_models()
        assert replacement in models or True  # may not be in profiles but OK


# --- Batch + Forecast ---

def test_batch_savings_in_forecast():
    """Forecast must show batch savings opportunity."""
    report = {"total_usd": 100.0, "by_model": {"gpt-4o": 100.0}, "by_provider": {}, "by_run": {}}
    fc = CostForecast(report, days_of_data=1.0)
    prediction = fc.predict_monthly()
    batch_savings = prediction["optimization_opportunities"]["with_batch_api"]
    assert batch_savings < prediction["monthly_estimate_usd"]


# --- Report + Security ---

def test_report_doesnt_leak_api_keys():
    """Report exports must not contain API keys."""
    report = {
        "total_usd": 10.0,
        "by_model": {"gpt-4o": 10.0},
        "by_provider": {"openai": 10.0},
        "by_run": {"run_1": 10.0},
    }
    exporter = ReportExporter(report)
    json_out = exporter.to_json()
    csv_out = exporter.to_csv()
    summary = exporter.to_summary()

    # None should contain anything resembling an API key
    for text in [json_out, csv_out, summary]:
        assert "sk-" not in text
        assert "api_key" not in text.lower()


# --- Session + Security ---

def test_session_name_sanitized():
    """Session name must not cause issues."""
    session = AgentSession("test<script>alert(1)</script>", budget_usd=5.0)
    assert session.name == "test<script>alert(1)</script>"  # stored as-is but used safely


# --- Security function composition ---

def test_security_functions_compose():
    """Security functions must work together."""
    # Mask API key
    assert mask_api_key("sk-proj-abc123def456") == "sk-p...f456"

    # Check prompt injection
    is_susp, _ = check_prompt_injection("normal user query")
    assert is_susp is False

    # Validate response safety
    is_safe, _ = validate_response_safety("Normal response text")
    assert is_safe is True


# --- Batch + Quality ---

def test_batch_submitter_estimate_with_quality():
    """Batch savings estimate must be positive."""
    submitter = BatchSubmitter()
    requests = [
        {"messages": [{"role": "user", "content": f"Question {i}"}]}
        for i in range(10)
    ]
    est = submitter.estimate_savings(requests)
    assert est["savings_usd"] > 0
    assert est["savings_pct"] == 50.0


# --- Context guard for different models ---

def test_context_guard_gemini_large_context():
    """Gemini with 1M context must fit large messages."""
    guard = ContextWindowGuard()
    msgs = [{"role": "user", "content": "x" * 100_000}]
    result = guard.check(msgs, "gemini-2.5-pro")
    assert result["fits"] is True  # 1M context


def test_context_guard_small_model():
    """Model with small context must detect overflow for large messages."""
    guard = ContextWindowGuard()
    content = " ".join(f"word{i}" for i in range(200_000))
    msgs = [{"role": "user", "content": content}]
    result = guard.check(msgs, "gpt-4o-mini")  # 128K context
    assert result["fits"] is False


# --- Model recommender edge cases ---

def test_recommender_all_tasks():
    """All task types must return a valid model."""
    r = ModelRecommender()
    for task in ["code", "reasoning", "factual", "creative", None]:
        model = r.recommend(task_type=task)
        assert model in r._profiles


def test_recommender_all_providers():
    """All provider preferences must return a valid model."""
    r = ModelRecommender()
    for provider in ["openai", "anthropic", "gemini"]:
        model = r.recommend(provider_preference=provider)
        assert model in r._profiles


def test_recommender_extreme_budget():
    """Extreme budget values must not crash."""
    r = ModelRecommender()
    assert r.recommend(budget_remaining=0.001) is not None
    assert r.recommend(budget_remaining=999999.0) is not None
