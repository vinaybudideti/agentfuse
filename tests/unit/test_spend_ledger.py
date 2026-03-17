"""
Tests for SpendLedger — persistent cost tracking.
"""

import os
import tempfile
from agentfuse.storage.spend_ledger import SpendLedger


def _make_ledger():
    """Create a ledger with a temp file."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_spend.jsonl")
    return SpendLedger(path), path


def test_record_and_query():
    ledger, _ = _make_ledger()
    ledger.record("run_1", "gpt-4o", 0.05, provider="openai", input_tokens=100)
    assert ledger.get_total_spend() == 0.05
    assert ledger.get_run_spend("run_1") == 0.05


def test_multiple_records():
    ledger, _ = _make_ledger()
    ledger.record("run_1", "gpt-4o", 0.10, provider="openai")
    ledger.record("run_1", "gpt-4o", 0.20, provider="openai")
    ledger.record("run_2", "claude-sonnet-4-6", 0.50, provider="anthropic")

    assert abs(ledger.get_total_spend() - 0.80) < 0.001
    assert abs(ledger.get_run_spend("run_1") - 0.30) < 0.001
    assert abs(ledger.get_run_spend("run_2") - 0.50) < 0.001


def test_spend_by_model():
    ledger, _ = _make_ledger()
    ledger.record("run_1", "gpt-4o", 0.10)
    ledger.record("run_2", "gpt-4o", 0.20)
    ledger.record("run_3", "claude-sonnet-4-6", 0.50)

    by_model = ledger.get_spend_by_model()
    assert abs(by_model["gpt-4o"] - 0.30) < 0.001
    assert abs(by_model["claude-sonnet-4-6"] - 0.50) < 0.001


def test_spend_by_provider():
    ledger, _ = _make_ledger()
    ledger.record("run_1", "gpt-4o", 0.10, provider="openai")
    ledger.record("run_2", "claude-sonnet-4-6", 0.50, provider="anthropic")

    by_provider = ledger.get_spend_by_provider()
    assert by_provider["openai"] == 0.10
    assert by_provider["anthropic"] == 0.50


def test_persistence_survives_restart():
    """Ledger must survive process restart by rebuilding from file."""
    _, path = _make_ledger()
    ledger1 = SpendLedger(path)
    ledger1.record("run_1", "gpt-4o", 0.10)
    ledger1.record("run_1", "gpt-4o", 0.20)

    # Simulate restart — create new ledger from same file
    ledger2 = SpendLedger(path)
    assert abs(ledger2.get_total_spend() - 0.30) < 0.001
    assert abs(ledger2.get_run_spend("run_1") - 0.30) < 0.001


def test_get_entries():
    ledger, _ = _make_ledger()
    ledger.record("run_1", "gpt-4o", 0.10)
    ledger.record("run_2", "claude-sonnet-4-6", 0.50)

    entries = ledger.get_entries()
    assert len(entries) == 2

    run_1_entries = ledger.get_entries(run_id="run_1")
    assert len(run_1_entries) == 1
    assert run_1_entries[0]["model"] == "gpt-4o"


def test_empty_ledger():
    ledger, _ = _make_ledger()
    assert ledger.get_total_spend() == 0.0
    assert ledger.get_spend_by_model() == {}
    assert ledger.get_run_spend("nonexistent") == 0.0


def test_concurrent_writes():
    """Multiple threads writing must not corrupt the ledger."""
    import threading
    ledger, _ = _make_ledger()
    errors = []

    def write_many(thread_id):
        try:
            for i in range(50):
                ledger.record(f"run_{thread_id}", "gpt-4o", 0.001)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write_many, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert ledger.get_total_spend() > 0
    # 5 threads * 50 writes * 0.001 = 0.25
    assert abs(ledger.get_total_spend() - 0.25) < 0.001


def test_corrupted_lines_ignored():
    """Corrupted JSONL lines must be silently ignored on rebuild."""
    _, path = _make_ledger()
    # Write valid + corrupted + valid lines
    with open(path, "w") as f:
        f.write('{"run_id":"r1","model":"gpt-4o","cost_usd":0.05,"provider":"openai"}\n')
        f.write('THIS IS NOT JSON\n')
        f.write('{"run_id":"r2","model":"gpt-4o","cost_usd":0.10,"provider":"openai"}\n')

    ledger = SpendLedger(path)
    assert abs(ledger.get_total_spend() - 0.15) < 1e-6


def test_cached_flag_recorded():
    """cached flag must be stored and retrievable in entries."""
    ledger, _ = _make_ledger()
    ledger.record("r1", "gpt-4o", 0.0, cached=True)
    entries = ledger.get_entries()
    assert entries[0]["cached"] is True


def test_entries_limit():
    """get_entries with limit must return only last N entries."""
    ledger, _ = _make_ledger()
    for i in range(10):
        ledger.record(f"run_{i}", "gpt-4o", 0.01)
    entries = ledger.get_entries(limit=3)
    assert len(entries) == 3
    # Should be the last 3 entries
    assert entries[0]["run_id"] == "run_7"
