"""
API surface tests — verify the complete public API is stable and accessible.
"""

import agentfuse


def test_version_format():
    """Version must be semver format."""
    parts = agentfuse.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_all_exports_are_not_none():
    """Every export must be a real object, not None."""
    for name in agentfuse.__all__:
        obj = getattr(agentfuse, name, None)
        assert obj is not None, f"{name} is None"


def test_gateway_functions():
    """Gateway functions must be callable."""
    assert callable(agentfuse.completion)
    assert callable(agentfuse.acompletion)
    assert callable(agentfuse.configure)
    assert callable(agentfuse.estimate_cost)
    assert callable(agentfuse.add_api_key)
    assert callable(agentfuse.get_spend_report)


def test_budget_classes():
    """Budget classes must be instantiable."""
    engine = agentfuse.BudgetEngine("api_test", 5.0, "gpt-4o")
    assert engine.budget == 5.0


def test_cache_classes():
    """Cache classes must have lookup/store methods."""
    cache = agentfuse.TwoTierCacheMiddleware()
    assert hasattr(cache, "lookup")
    assert hasattr(cache, "store")


def test_security_functions():
    """Security functions must work."""
    assert agentfuse.mask_api_key("sk-1234567890abcdef") == "sk-1...cdef"
    is_susp, _ = agentfuse.check_prompt_injection("normal text")
    assert is_susp is False
    is_safe, _ = agentfuse.validate_response_safety("normal response")
    assert is_safe is True


def test_session_context_manager():
    """AgentSession must work as context manager."""
    with agentfuse.AgentSession("api_test", budget_usd=5.0) as session:
        assert session.name == "api_test"


def test_kill_switch_singleton():
    """kill_switch must be a singleton with kill/revive methods."""
    ks = agentfuse.kill_switch
    assert hasattr(ks, "kill")
    assert hasattr(ks, "revive")
    assert hasattr(ks, "check")


def test_novel_modules_accessible():
    """All novel modules must be accessible."""
    assert agentfuse.CostPredictiveRouter is not None
    assert agentfuse.PromptCompressor is not None
    assert agentfuse.ToolCostTracker is not None
    assert agentfuse.ConversationCostEstimator is not None
    assert agentfuse.HierarchicalBudget is not None
    assert agentfuse.ContentGuardrails is not None
    assert agentfuse.ContextWindowGuard is not None
    assert agentfuse.UsageAnalytics is not None
    assert agentfuse.ModelRecommender is not None
    assert agentfuse.ResponseQualityScorer is not None
    assert agentfuse.CostForecast is not None
    assert agentfuse.ReportExporter is not None


def test_export_count():
    """Must have at least 70 exports."""
    assert len(agentfuse.__all__) >= 70
