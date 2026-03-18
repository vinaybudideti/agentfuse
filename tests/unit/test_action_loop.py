"""Tests for action-based loop detection."""

import pytest
from agentfuse.core.loop import LoopDetectionMiddleware, LoopDetected


def test_action_loop_detected():
    """Repeating same action 3+ times must be detected."""
    detector = LoopDetectionMiddleware(cost_threshold=0.0)
    with pytest.raises(LoopDetected):
        for _ in range(5):
            detector.check_action("search(query='weather')", step_cost=0.01)


def test_different_actions_no_loop():
    """Different actions must not trigger loop detection."""
    detector = LoopDetectionMiddleware(cost_threshold=1.0)
    detector.check_action("search(query='weather')")
    detector.check_action("calculate(expr='2+2')")
    detector.check_action("fetch(url='example.com')")
    # No loop — all different actions


def test_action_loop_cost_threshold():
    """Loop must only trigger when cost exceeds threshold."""
    detector = LoopDetectionMiddleware(cost_threshold=0.10)
    # 3 repeats at $0.01 each = $0.03 < $0.10 threshold
    for _ in range(3):
        detector.check_action("search(query='same')", step_cost=0.01)
    # Should not raise yet


def test_action_reset():
    """Reset must clear action window."""
    detector = LoopDetectionMiddleware(cost_threshold=0.0)
    detector.check_action("action_a")
    detector.check_action("action_a")
    detector.reset()
    # After reset, same action should be fine again
    detector.check_action("action_a")
