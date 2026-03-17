"""
Tests for ReportExporter — multi-format cost report generation.
"""

import json
import os
import tempfile

from agentfuse.core.report_exporter import ReportExporter


def _sample_report():
    return {
        "total_usd": 25.50,
        "by_model": {"gpt-4o": 15.00, "claude-sonnet-4-6": 8.00, "gpt-4o-mini": 2.50},
        "by_provider": {"openai": 17.50, "anthropic": 8.00},
        "by_run": {"run_1": 10.00, "run_2": 8.00, "run_3": 7.50},
    }


def test_to_json_valid():
    """JSON export must produce valid JSON."""
    exporter = ReportExporter(_sample_report())
    output = exporter.to_json()
    parsed = json.loads(output)
    assert parsed["total_spend_usd"] == 25.50
    assert "gpt-4o" in parsed["by_model"]


def test_to_json_file():
    """JSON export to file must write valid JSON."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        exporter = ReportExporter(_sample_report())
        exporter.to_json(path=path)
        with open(path) as f:
            data = json.load(f)
        assert data["total_spend_usd"] == 25.50
    finally:
        os.unlink(path)


def test_to_csv_valid():
    """CSV export must have header and data rows."""
    exporter = ReportExporter(_sample_report())
    csv_text = exporter.to_csv()
    lines = csv_text.strip().split("\n")
    assert "model" in lines[0] and "cost_usd" in lines[0]
    assert any("TOTAL" in line for line in lines)
    assert len(lines) >= 4  # header + 3 models + total (may have extra newlines)


def test_to_csv_file():
    """CSV export to file must work."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        exporter = ReportExporter(_sample_report())
        exporter.to_csv(path=path)
        with open(path) as f:
            content = f.read()
        assert "model,cost_usd" in content
    finally:
        os.unlink(path)


def test_to_summary_readable():
    """Summary must be human-readable."""
    exporter = ReportExporter(_sample_report())
    summary = exporter.to_summary()
    assert "Total Spend" in summary
    assert "$25.50" in summary
    assert "gpt-4o" in summary


def test_to_dict():
    """Dict export must have expected structure."""
    exporter = ReportExporter(_sample_report())
    d = exporter.to_dict()
    assert d["total_spend_usd"] == 25.50
    assert "gpt-4o" in d["models"]
    assert d["run_count"] == 3


def test_empty_report():
    """Empty report must not crash."""
    exporter = ReportExporter({})
    assert exporter.to_json() is not None
    assert exporter.to_csv() is not None
    assert exporter.to_summary() is not None
    assert exporter.to_dict()["total_spend_usd"] == 0
