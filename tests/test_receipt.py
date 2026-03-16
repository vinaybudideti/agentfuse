from agentfuse.core.receipt import CostReceiptEmitter


def test_receipt_has_all_required_fields():
    r = CostReceiptEmitter("run_abc", "customer_support", "user_1", 0.50)
    r.add_step("llm_call", "gpt-4o", 1240, 380, 0.0089, 1230)
    r.add_step("llm_call", "gpt-4o", 0, 0, 0.0, 12, cache_tier=1)
    receipt = r.emit()

    required = ["receipt_version", "run_id", "agent_type", "user_id",
                "started_at", "completed_at", "status", "total_cost_usd",
                "budget_usd", "budget_utilization_pct", "cache_savings_usd",
                "cache_hit_rate", "retry_cost_usd", "model_downgrades",
                "context_compressions", "steps"]
    for f in required:
        assert f in receipt, f"Missing field: {f}"


def test_receipt_step_fields():
    r = CostReceiptEmitter("run_1", "test", "user_1", 1.00)
    r.add_step("tool_call", "gpt-4o", 100, 50, 0.001, 350, tool_name="search")
    receipt = r.emit()
    step = receipt["steps"][0]
    for f in ["step", "step_type", "model", "input_tokens",
              "output_tokens", "cost_usd", "cache_tier", "latency_ms"]:
        assert f in step, f"Missing step field: {f}"


def test_receipt_cache_hit_rate():
    r = CostReceiptEmitter("run_1", "test", "user_1", 1.00)
    r.add_step("llm_call", "gpt-4o", 100, 50, 0.001, 100)              # miss
    r.add_step("llm_call", "gpt-4o", 0, 0, 0.0, 5, cache_tier=1)      # hit
    r.add_step("llm_call", "gpt-4o", 0, 0, 0.0, 5, cache_tier=1)      # hit
    receipt = r.emit()
    assert abs(receipt["cache_hit_rate"] - 0.667) < 0.01


def test_receipt_emit_json():
    import json
    r = CostReceiptEmitter("run_1", "test", "user_1", 1.00)
    r.add_step("llm_call", "gpt-4o", 100, 50, 0.001, 100)
    json_str = r.emit_json()
    parsed = json.loads(json_str)
    assert parsed["run_id"] == "run_1"
