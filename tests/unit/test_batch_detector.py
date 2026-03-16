"""
Tests for BatchEligibilityDetector.
"""

from agentfuse.core.batch_detector import BatchEligibilityDetector


def test_no_batch_below_threshold():
    detector = BatchEligibilityDetector(min_batch_size=5)
    msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "hi"}]
    for _ in range(4):
        result = detector.observe(msgs, "gpt-4o", estimated_cost=0.01)
    assert result is None  # only 4, need 5


def test_batch_detected_at_threshold():
    detector = BatchEligibilityDetector(min_batch_size=3)
    msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "query"}]
    results = []
    for _ in range(5):
        result = detector.observe(msgs, "gpt-4o", estimated_cost=0.01)
        if result:
            results.append(result)

    assert len(results) > 0
    assert results[0].request_count >= 3
    assert results[0].estimated_savings_usd > 0


def test_different_system_prompts_not_batched():
    detector = BatchEligibilityDetector(min_batch_size=3)
    for i in range(5):
        msgs = [{"role": "system", "content": f"Unique prompt {i}"}, {"role": "user", "content": "hi"}]
        result = detector.observe(msgs, "gpt-4o")
    assert result is None  # all different prompts


def test_different_models_not_batched():
    detector = BatchEligibilityDetector(min_batch_size=3)
    msgs = [{"role": "system", "content": "Same prompt"}, {"role": "user", "content": "hi"}]
    detector.observe(msgs, "gpt-4o")
    detector.observe(msgs, "gpt-4o")
    result = detector.observe(msgs, "claude-sonnet-4-6")  # different model
    assert result is None


def test_stats():
    detector = BatchEligibilityDetector(min_batch_size=2)
    msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "hi"}]
    for _ in range(5):
        detector.observe(msgs, "gpt-4o")

    stats = detector.get_stats()
    assert stats["total_observed"] == 5
    assert stats["batch_opportunities"] > 0


def test_no_system_prompt_still_works():
    detector = BatchEligibilityDetector(min_batch_size=2)
    msgs = [{"role": "user", "content": "no system prompt"}]
    detector.observe(msgs, "gpt-4o")
    result = detector.observe(msgs, "gpt-4o")
    assert result is not None
