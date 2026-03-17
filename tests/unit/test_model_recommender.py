"""
Tests for ModelRecommender — optimal model selection.
"""

from agentfuse.core.model_recommender import ModelRecommender


def test_recommend_returns_model():
    """Recommend must return a valid model name."""
    r = ModelRecommender()
    model = r.recommend()
    assert isinstance(model, str)
    assert len(model) > 0


def test_low_budget_favors_cheap():
    """Low budget must favor cheaper models."""
    r = ModelRecommender()
    cheap = r.recommend(budget_remaining=0.10)
    expensive = r.recommend(budget_remaining=100.0)
    # Cheap budget should pick a cheaper model
    cheap_profile = r._profiles.get(cheap, {})
    expensive_profile = r._profiles.get(expensive, {})
    assert cheap_profile.get("cost_per_1k", 1) <= expensive_profile.get("cost_per_1k", 0)


def test_high_latency_favors_fast():
    """High latency priority must favor fast models."""
    r = ModelRecommender()
    fast = r.recommend(latency_priority="high")
    fast_profile = r._profiles.get(fast, {})
    assert fast_profile.get("speed", 0) >= 0.8


def test_code_task_returns_valid():
    """Code task recommendation must return a valid model."""
    r = ModelRecommender()
    model = r.recommend(task_type="code", budget_remaining=50.0)
    assert model in r._profiles


def test_provider_preference():
    """Provider preference must filter to that provider's models."""
    r = ModelRecommender()
    model = r.recommend(provider_preference="anthropic")
    assert model.startswith("claude")


def test_context_requirement_filters():
    """Large context requirement must filter out small-context models."""
    r = ModelRecommender()
    model = r.recommend(required_context=500_000)
    profile = r._profiles.get(model, {})
    assert profile.get("context", 0) >= 500_000


def test_compare_models():
    """Compare must return structured comparison."""
    r = ModelRecommender()
    result = r.compare("gpt-5.4", "gpt-4.1-nano")
    assert result["quality_diff"] > 0  # gpt-5.4 is higher quality
    assert result["cost_ratio"] > 1  # gpt-5.4 is more expensive


def test_get_all_models():
    """Must return all registered models."""
    r = ModelRecommender()
    models = r.get_all_models()
    assert len(models) >= 10
    assert "gpt-5" in models


def test_unknown_provider_returns_any():
    """Unknown provider must not crash."""
    r = ModelRecommender()
    model = r.recommend(provider_preference="unknown_provider")
    assert isinstance(model, str)
