"""
Tests verifying research file 4 findings are correctly applied.
"""

from types import SimpleNamespace


def test_gpt54_base_pricing():
    """GPT-5.4 base must be $2.50/$15.00 (NOT $10/$30)."""
    from agentfuse.providers.registry import ModelRegistry
    reg = ModelRegistry(refresh_hours=0)
    p = reg.get_pricing("gpt-5.4")
    assert p["input"] == 2.50
    assert p["output"] == 15.00


def test_gpt54_pro_pricing():
    """GPT-5.4 Pro must be $30/$180."""
    from agentfuse.providers.registry import ModelRegistry
    reg = ModelRegistry(refresh_hours=0)
    p = reg.get_pricing("gpt-5.4-pro")
    assert p["input"] == 30.00
    assert p["output"] == 180.00


def test_gpt41_cached_10pct():
    """GPT-4.1 cached input must be 10% of base ($0.20)."""
    from agentfuse.providers.registry import ModelRegistry
    reg = ModelRegistry(refresh_hours=0)
    p = reg.get_pricing("gpt-4.1")
    assert p["cached_input"] == 0.20


def test_gpt41_mini_cached():
    """GPT-4.1 Mini cached must be $0.04."""
    from agentfuse.providers.registry import ModelRegistry
    reg = ModelRegistry(refresh_hours=0)
    p = reg.get_pricing("gpt-4.1-mini")
    assert p["cached_input"] == 0.04


def test_gpt5_context_128k():
    """GPT-5 context must be 128K (not 1M)."""
    from agentfuse.providers.registry import ModelRegistry
    reg = ModelRegistry(refresh_hours=0)
    p = reg.get_pricing("gpt-5")
    assert p["context"] == 128_000


def test_gemini_context_2m():
    """Gemini 2.5 Pro context must be 2M."""
    from agentfuse.providers.registry import ModelRegistry
    reg = ModelRegistry(refresh_hours=0)
    p = reg.get_pricing("gemini-2.5-pro")
    assert p["context"] == 2_000_000


def test_budget_safety_margin_20pct():
    """Budget safety margin must be 1.2x (20% buffer)."""
    from agentfuse.core.budget import BudgetEngine
    engine = BudgetEngine("margin_test", 1.0, "gpt-4o")
    engine.spent = 0.79
    # 0.79 + (0.01 * 1.2) = 0.812 — at 81.2%, should trigger 80% downgrade
    msgs, model = engine.check_and_act(0.01, [{"role": "user", "content": "test"}])
    assert model != "gpt-4o"  # should be downgraded


def test_batch_cache_stacking_anthropic():
    """Anthropic batch + cache must show 95% savings."""
    from agentfuse.core.batch_submitter import BatchSubmitter
    submitter = BatchSubmitter()
    requests = [{"messages": [{"role": "user", "content": "test"}]}]
    est = submitter.estimate_savings(requests, model="claude-sonnet-4-6")
    assert est["with_cache_stacking_pct"] == 95.0


def test_batch_cache_stacking_openai():
    """OpenAI batch must show 50% savings (no cache stacking)."""
    from agentfuse.core.batch_submitter import BatchSubmitter
    submitter = BatchSubmitter()
    requests = [{"messages": [{"role": "user", "content": "test"}]}]
    est = submitter.estimate_savings(requests, model="gpt-4o")
    assert est["with_cache_stacking_pct"] == 50.0


def test_stream_usage_tracker_openai():
    """StreamUsageTracker must extract OpenAI final chunk usage."""
    from agentfuse.providers.stream_usage import StreamUsageTracker
    tracker = StreamUsageTracker(provider="openai")
    chunk = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=100, total_tokens=150),
    )
    result = tracker.process_chunk(chunk)
    assert result.input_tokens == 50
    assert result.finalized is True


def test_async_spend_recorder_non_blocking():
    """AsyncSpendRecorder.record() must be non-blocking."""
    from agentfuse.storage.async_recorder import AsyncSpendRecorder
    recorder = AsyncSpendRecorder()
    result = recorder.record(model="gpt-4o", cost_usd=0.05)
    assert result is True


def test_action_loop_detection():
    """Action-hash loop must detect repeated tool calls."""
    from agentfuse.core.loop import LoopDetectionMiddleware, LoopDetected
    import pytest
    detector = LoopDetectionMiddleware(cost_threshold=0.0)
    with pytest.raises(LoopDetected):
        for _ in range(5):
            detector.check_action("search('same query')", step_cost=0.01)
