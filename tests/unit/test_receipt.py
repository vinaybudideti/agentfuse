"""
Loop 7 — CostReceiptEmitter behavioral tests.
"""

import json
from agentfuse.core.receipt import CostReceiptEmitter, ReceiptStep


def test_empty_receipt_has_required_fields():
    """Receipt with no steps must have all 16 required fields."""
    emitter = CostReceiptEmitter("run1", "test_agent", "user1", budget_usd=5.0)
    receipt = emitter.emit()
    required = [
        "receipt_version", "run_id", "agent_type", "user_id",
        "started_at", "completed_at", "status", "total_cost_usd",
        "budget_usd", "budget_utilization_pct", "cache_savings_usd",
        "cache_hit_rate", "retry_cost_usd", "model_downgrades",
        "context_compressions", "steps",
    ]
    for field in required:
        assert field in receipt, f"Missing field: {field}"


def test_add_step_increments_cost():
    """Each step must add to total cost."""
    emitter = CostReceiptEmitter("run1", budget_usd=5.0)
    emitter.add_step("llm_call", "gpt-4o", 100, 50, 0.01, 200)
    emitter.add_step("llm_call", "gpt-4o", 200, 100, 0.02, 300)
    assert abs(emitter.total_cost_usd - 0.03) < 0.001
    assert emitter.total_calls == 2


def test_cache_hit_rate_calculation():
    """Cache hit rate must be correctly calculated."""
    emitter = CostReceiptEmitter("run1", budget_usd=5.0)
    emitter.add_step("llm_call", "gpt-4o", 100, 50, 0.0, 10, cache_tier=1)  # cache hit
    emitter.add_step("llm_call", "gpt-4o", 100, 50, 0.01, 200)  # miss
    emitter.add_step("llm_call", "gpt-4o", 100, 50, 0.0, 5, cache_tier=2)  # cache hit
    receipt = emitter.emit()
    # 2/3 = 0.667
    assert abs(receipt["cache_hit_rate"] - 0.667) < 0.01


def test_budget_utilization_calculation():
    """Budget utilization percentage must be correct."""
    emitter = CostReceiptEmitter("run1", budget_usd=10.0)
    emitter.add_step("llm_call", "gpt-4o", 100, 50, 2.5, 200)
    receipt = emitter.emit()
    assert abs(receipt["budget_utilization_pct"] - 25.0) < 0.01


def test_record_events():
    """record_* methods must increment counters."""
    emitter = CostReceiptEmitter("run1", budget_usd=5.0)
    emitter.record_cache_saving(0.5)
    emitter.record_retry_cost(0.1)
    emitter.record_model_downgrade()
    emitter.record_context_compression()
    receipt = emitter.emit()
    assert receipt["cache_savings_usd"] == 0.5
    assert receipt["retry_cost_usd"] == 0.1
    assert receipt["model_downgrades"] == 1
    assert receipt["context_compressions"] == 1


def test_emit_json_valid():
    """emit_json must produce valid JSON."""
    emitter = CostReceiptEmitter("run1", budget_usd=5.0)
    emitter.add_step("llm_call", "gpt-4o", 100, 50, 0.01, 200)
    json_str = emitter.emit_json()
    parsed = json.loads(json_str)
    assert parsed["run_id"] == "run1"
    assert len(parsed["steps"]) == 1


def test_step_has_all_fields():
    """Each step in the receipt must have all expected fields."""
    emitter = CostReceiptEmitter("run1", budget_usd=5.0)
    emitter.add_step("tool_call", "gpt-4o", 100, 50, 0.01, 200,
                     cache_tier=1, tool_name="search")
    receipt = emitter.emit()
    step = receipt["steps"][0]
    assert step["step"] == 1
    assert step["step_type"] == "tool_call"
    assert step["tool_name"] == "search"
    assert step["model"] == "gpt-4o"
    assert step["cache_tier"] == 1
