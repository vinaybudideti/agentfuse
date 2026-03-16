"""
Tests for RequestOptimizer — pre-send request optimization.
"""

from agentfuse.core.request_optimizer import RequestOptimizer


def test_remove_empty_messages():
    opt = RequestOptimizer()
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "world"},
    ]
    result, report = opt.optimize(msgs, "gpt-4o")
    assert len(result) == 2
    assert report.messages_removed == 1
    assert "empty" in report.optimizations_applied[0]


def test_remove_consecutive_duplicates():
    opt = RequestOptimizer()
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "hello"},  # duplicate
        {"role": "assistant", "content": "hi"},
    ]
    result, report = opt.optimize(msgs, "gpt-4o")
    assert len(result) == 2
    assert report.messages_removed == 1


def test_dedup_system_prompts():
    opt = RequestOptimizer()
    msgs = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "You are helpful"},  # duplicate system
    ]
    result, report = opt.optimize(msgs, "gpt-4o")
    system_msgs = [m for m in result if m["role"] == "system"]
    assert len(system_msgs) == 1


def test_no_optimization_needed():
    opt = RequestOptimizer()
    msgs = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]
    result, report = opt.optimize(msgs, "gpt-4o")
    assert len(result) == 3
    assert report.messages_removed == 0
    assert len(report.optimizations_applied) == 0


def test_tokens_saved_calculated():
    opt = RequestOptimizer()
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "hello"},  # duplicate
        {"role": "user", "content": "hello"},  # duplicate
    ]
    _, report = opt.optimize(msgs, "gpt-4o")
    assert report.estimated_tokens_saved > 0
    assert report.estimated_cost_saving_usd > 0


def test_estimate_cost():
    opt = RequestOptimizer()
    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    estimate = opt.estimate_cost(msgs, "gpt-4o")
    assert estimate["estimated_input_tokens"] > 0
    assert estimate["estimated_total_cost_usd"] > 0
    assert estimate["model"] == "gpt-4o"


def test_check_context_window():
    opt = RequestOptimizer()
    msgs = [{"role": "user", "content": "short message"}]
    check = opt.check_context_window(msgs, "gpt-4o")
    assert check["fits"] is True
    assert check["input_tokens"] < check["context_limit"]
    assert check["utilization_pct"] < 1.0


def test_pct_saved():
    opt = RequestOptimizer()
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": ""},
        {"role": "user", "content": ""},
    ]
    _, report = opt.optimize(msgs, "gpt-4o")
    assert report.pct_saved > 0.5  # removed 2 of 3
