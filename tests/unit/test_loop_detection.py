"""
Loop 8 — LoopDetectionMiddleware unit tests (with mocked embedder).
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from agentfuse.core.loop import LoopDetectionMiddleware, LoopDetected


def _make_loop_detector(sim_threshold=0.92, cost_threshold=0.50):
    """Create a loop detector with mock embedder."""
    detector = LoopDetectionMiddleware.__new__(LoopDetectionMiddleware)
    detector.window = []
    detector.sim_threshold = sim_threshold
    detector.cost_threshold = cost_threshold
    detector.loop_cost = 0.0
    detector.max_window = 10

    # Mock embedder that returns deterministic vectors
    mock = MagicMock()
    detector.embedder = mock
    detector._lock = __import__("threading").Lock()
    return detector, mock


def _unit_vec(dim=768, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def test_no_loop_on_different_prompts():
    """Different prompts must not trigger loop detection."""
    detector, mock = _make_loop_detector()
    # Return different embeddings for each call
    mock.encode.side_effect = [
        np.array([_unit_vec(seed=i)]) for i in range(5)
    ]
    for i in range(5):
        detector.check(f"prompt {i}", step_cost=0.10)
    # Should not raise


def test_loop_detected_on_identical_prompts():
    """Identical embeddings with enough cost must trigger LoopDetected."""
    detector, mock = _make_loop_detector(cost_threshold=0.20)
    same_vec = _unit_vec(seed=1)
    mock.encode.return_value = np.array([same_vec])

    detector.check("same prompt", step_cost=0.05)
    detector.check("same prompt", step_cost=0.05)
    detector.check("same prompt", step_cost=0.05)

    with pytest.raises(LoopDetected) as exc_info:
        detector.check("same prompt", step_cost=0.10)

    assert exc_info.value.loop_cost > 0.20
    assert exc_info.value.similarity > 0.92


def test_loop_cost_accumulates():
    """Loop cost must accumulate across similar prompts."""
    detector, mock = _make_loop_detector(cost_threshold=1.0)
    same_vec = _unit_vec(seed=1)
    mock.encode.return_value = np.array([same_vec])

    for i in range(5):
        detector.check("repeated", step_cost=0.10)
    # First call: no matches (window empty). Calls 2-5 each match call 1 (and earlier).
    # Each match adds step_cost to loop_cost.
    assert detector.loop_cost > 0


def test_reset_clears_state():
    """reset() must clear window and loop cost."""
    detector, mock = _make_loop_detector()
    same_vec = _unit_vec(seed=1)
    mock.encode.return_value = np.array([same_vec])
    detector.check("prompt", step_cost=0.10)
    detector.check("prompt", step_cost=0.10)

    detector.reset()
    assert detector.window == []
    assert detector.loop_cost == 0.0


def test_window_size_limited():
    """Window must not exceed max_window entries."""
    detector, mock = _make_loop_detector()
    mock.encode.side_effect = [
        np.array([_unit_vec(seed=i)]) for i in range(15)
    ]
    for i in range(15):
        detector.check(f"unique {i}", step_cost=0.01)
    assert len(detector.window) <= 10


def test_no_embedder_skips_check():
    """When embedder is None, check must return None (no crash)."""
    detector = LoopDetectionMiddleware.__new__(LoopDetectionMiddleware)
    detector.embedder = None
    detector.window = []
    detector.loop_cost = 0.0
    result = detector.check("anything", step_cost=0.10)
    assert result is None


def test_cosine_sim_identical_vectors():
    """Cosine similarity of identical vectors must be ~1.0."""
    detector, _ = _make_loop_detector()
    vec = _unit_vec(seed=1)
    sim = detector._cosine_sim(vec, vec)
    assert abs(sim - 1.0) < 0.001


def test_cosine_sim_zero_vector():
    """Cosine similarity with zero vector must return 0.0."""
    detector, _ = _make_loop_detector()
    vec = _unit_vec(seed=1)
    zero = [0.0] * len(vec)
    sim = detector._cosine_sim(vec, zero)
    assert sim == 0.0


def test_cosine_sim_orthogonal_vectors():
    """Orthogonal vectors must have ~0.0 similarity."""
    detector, _ = _make_loop_detector()
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    sim = detector._cosine_sim(a, b)
    assert abs(sim) < 0.001
