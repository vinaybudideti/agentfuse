"""Tests for CacheMonitor."""

from agentfuse.core.cache_monitor import CacheMonitor


def test_hit_rate_100pct():
    m = CacheMonitor()
    for _ in range(10):
        m.record_hit()
    assert m.hit_rate == 1.0


def test_hit_rate_0pct():
    m = CacheMonitor()
    for _ in range(10):
        m.record_miss()
    assert m.hit_rate == 0.0


def test_hit_rate_mixed():
    m = CacheMonitor()
    for _ in range(7):
        m.record_hit()
    for _ in range(3):
        m.record_miss()
    assert abs(m.hit_rate - 0.7) < 0.01


def test_is_degraded():
    m = CacheMonitor(alert_threshold=0.5)
    for _ in range(10):
        m.record_miss()
    assert m.is_degraded() is True


def test_not_degraded():
    m = CacheMonitor(alert_threshold=0.5)
    for _ in range(10):
        m.record_hit()
    assert m.is_degraded() is False


def test_alert_callback():
    alerts = []
    m = CacheMonitor(alert_threshold=0.5, alert_callback=lambda rate, reason: alerts.append(rate))
    for _ in range(15):
        m.record_miss()
    assert len(alerts) >= 1


def test_stats():
    m = CacheMonitor()
    m.record_hit()
    m.record_miss()
    stats = m.get_stats()
    assert stats["total_hits"] == 1
    assert stats["total_misses"] == 1
    assert stats["total_lookups"] == 2


def test_reset():
    m = CacheMonitor()
    m.record_hit()
    m.record_miss()
    m.reset()
    assert m.get_stats()["total_lookups"] == 0


def test_sliding_window():
    m = CacheMonitor(window_size=5)
    for _ in range(5):
        m.record_miss()
    for _ in range(5):
        m.record_hit()
    # Window only has last 5 (all hits)
    assert m.hit_rate == 1.0
