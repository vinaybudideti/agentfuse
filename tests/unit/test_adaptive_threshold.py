"""
Tests for AdaptiveSimilarityThreshold.
"""

from agentfuse.core.adaptive_threshold import AdaptiveSimilarityThreshold


def test_initial_threshold():
    t = AdaptiveSimilarityThreshold(initial=0.92)
    assert t.get() == 0.92


def test_bad_hit_tightens():
    t = AdaptiveSimilarityThreshold(initial=0.92, step=0.01)
    t.report_bad_hit()
    assert t.get() == 0.93


def test_good_hit_loosens():
    t = AdaptiveSimilarityThreshold(initial=0.92, step=0.01)
    t.report_good_hit()
    assert t.get() == 0.915  # loosens at half rate


def test_asymmetric_adjustment():
    """Tightening must be 2x faster than loosening."""
    t = AdaptiveSimilarityThreshold(initial=0.92, step=0.02)
    t.report_bad_hit()   # +0.02 = 0.94
    t.report_good_hit()  # -0.01 = 0.93
    assert abs(t.get() - 0.93) < 0.001


def test_max_threshold_bounded():
    t = AdaptiveSimilarityThreshold(initial=0.97, max_threshold=0.98, step=0.02)
    t.report_bad_hit()  # 0.97 + 0.02 = 0.99, but max is 0.98
    assert t.get() == 0.98


def test_min_threshold_bounded():
    t = AdaptiveSimilarityThreshold(initial=0.86, min_threshold=0.85, step=0.02)
    t.report_good_hit()  # 0.86 - 0.01 = 0.85 (at min)
    t.report_good_hit()  # can't go below 0.85
    assert t.get() == 0.85


def test_stats():
    t = AdaptiveSimilarityThreshold()
    t.report_good_hit()
    t.report_good_hit()
    t.report_bad_hit()
    stats = t.get_stats()
    assert stats["good_hits"] == 2
    assert stats["bad_hits"] == 1
    assert stats["accuracy"] > 0.6


def test_reset():
    t = AdaptiveSimilarityThreshold(initial=0.92)
    t.report_bad_hit()
    t.report_bad_hit()
    assert t.get() > 0.92
    t.reset()
    assert t.get() == 0.92
