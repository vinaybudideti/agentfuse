"""Tests for AsyncSpendRecorder."""

import pytest
import asyncio
from agentfuse.storage.async_recorder import AsyncSpendRecorder, SpendEvent


def test_spend_event_defaults():
    """SpendEvent must have sensible defaults."""
    e = SpendEvent(model="gpt-4o", cost_usd=0.05)
    assert e.model == "gpt-4o"
    assert e.timestamp > 0
    assert e.cached is False


def test_record_non_blocking():
    """record() must not block."""
    recorder = AsyncSpendRecorder()
    result = recorder.record(model="gpt-4o", cost_usd=0.05, run_id="test")
    assert result is True
    assert recorder._total_recorded == 1


def test_record_tracks_count():
    """Multiple records must be counted."""
    recorder = AsyncSpendRecorder()
    for i in range(10):
        recorder.record(model="gpt-4o", cost_usd=0.01)
    assert recorder._total_recorded == 10


def test_queue_full_drops():
    """Full queue must drop events (not block)."""
    recorder = AsyncSpendRecorder(max_queue_size=5)
    for i in range(10):
        recorder.record(model="gpt-4o", cost_usd=0.01)
    assert recorder._total_dropped > 0


def test_stats():
    """get_stats must return expected keys."""
    recorder = AsyncSpendRecorder()
    recorder.record(model="gpt-4o", cost_usd=0.05)
    stats = recorder.get_stats()
    assert stats["total_recorded"] == 1
    assert stats["running"] is False
    assert "queue_size" in stats


@pytest.mark.asyncio
async def test_start_and_stop():
    """start/stop lifecycle must not crash."""
    recorder = AsyncSpendRecorder(flush_interval=0.1)
    await recorder.start()
    assert recorder._running is True
    recorder.record(model="gpt-4o", cost_usd=0.05, run_id="async_test")
    await asyncio.sleep(0.3)  # let flush happen
    await recorder.stop()
    assert recorder._running is False


@pytest.mark.asyncio
async def test_flush_writes_events():
    """Flushed events must be written."""
    recorder = AsyncSpendRecorder(flush_interval=0.1)
    await recorder.start()
    for i in range(5):
        recorder.record(model="gpt-4o", cost_usd=0.01, run_id=f"flush_{i}")
    await asyncio.sleep(0.5)
    await recorder.stop()
    assert recorder._total_flushed >= 5
