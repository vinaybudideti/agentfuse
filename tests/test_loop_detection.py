import pytest
from agentfuse.core.loop import LoopDetectionMiddleware, LoopDetected


def test_no_loop_on_different_prompts():
    l = LoopDetectionMiddleware()
    if l.embedder is None:
        pytest.skip("Embedder not available")
    l.check("What is Python?", 0.01)
    l.check("What is JavaScript?", 0.01)
    l.check("What is Rust?", 0.01)
    # No exception = no false positive


def test_loop_detected_on_identical_prompts():
    l = LoopDetectionMiddleware(cost_threshold=0.05)
    if l.embedder is None:
        pytest.skip("Embedder not available")
    prompt = "Search for the latest news about AI"
    with pytest.raises(LoopDetected):
        for _ in range(10):
            l.check(prompt, step_cost=0.02)


def test_reset_clears_window():
    l = LoopDetectionMiddleware()
    l.loop_cost = 99.0
    l.window = [("fake", 0.0)]
    l.reset()
    assert l.loop_cost == 0.0
    assert l.window == []
