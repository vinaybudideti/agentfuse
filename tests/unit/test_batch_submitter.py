"""
Tests for BatchSubmitter — actual batch API execution.
"""

import pytest
from unittest.mock import patch, MagicMock
from agentfuse.core.batch_submitter import BatchSubmitter, BatchJob


def test_batch_job_dataclass():
    """BatchJob must have correct default values."""
    job = BatchJob(batch_id="test-123", provider="openai", request_count=10)
    assert job.status == "pending"
    assert job.results is None
    assert job.submitted_at > 0


def test_unsupported_provider_raises():
    """Unsupported provider must raise ValueError."""
    submitter = BatchSubmitter(provider="unknown")
    with pytest.raises(ValueError, match="Unsupported"):
        submitter.submit([{"messages": [{"role": "user", "content": "hi"}]}])


def test_estimate_savings():
    """Savings estimate must show 50% discount."""
    submitter = BatchSubmitter(provider="openai")
    requests = [
        {"messages": [{"role": "user", "content": "Q1"}]},
        {"messages": [{"role": "user", "content": "Q2"}]},
    ]
    est = submitter.estimate_savings(requests, model="gpt-4o")
    assert est["savings_pct"] == 50.0
    assert est["batch_cost_usd"] < est["realtime_cost_usd"]
    assert est["request_count"] == 2


def test_list_jobs_empty():
    """list_jobs must return empty list initially."""
    submitter = BatchSubmitter()
    assert submitter.list_jobs() == []


def test_get_job_unknown():
    """get_job for unknown ID must return None."""
    submitter = BatchSubmitter()
    assert submitter.get_job("nonexistent") is None


def test_check_status_unknown():
    """check_status for unknown ID must return 'unknown'."""
    submitter = BatchSubmitter()
    assert submitter.check_status("nonexistent") == "unknown"


def test_get_results_incomplete():
    """get_results for non-completed job must return None."""
    submitter = BatchSubmitter()
    submitter._jobs["test-1"] = BatchJob(
        batch_id="test-1", provider="openai", request_count=5, status="in_progress"
    )
    assert submitter.get_results("test-1") is None


def test_openai_import_error():
    """Missing openai package must raise ImportError."""
    submitter = BatchSubmitter(provider="openai")
    with patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(ImportError, match="openai"):
            submitter.submit([{"messages": [{"role": "user", "content": "hi"}]}])


def test_anthropic_import_error():
    """Missing anthropic package must raise ImportError."""
    submitter = BatchSubmitter(provider="anthropic")
    with patch.dict("sys.modules", {"anthropic": None}):
        with pytest.raises(ImportError, match="anthropic"):
            submitter.submit([{"messages": [{"role": "user", "content": "hi"}]}])


def test_estimate_multiple_models():
    """Savings estimate must work with mixed models."""
    submitter = BatchSubmitter()
    requests = [
        {"messages": [{"role": "user", "content": "Q1"}], "model": "gpt-4o"},
        {"messages": [{"role": "user", "content": "Q2"}], "model": "gpt-5"},
    ]
    est = submitter.estimate_savings(requests)
    assert est["request_count"] == 2
    assert est["realtime_cost_usd"] > 0
