"""
ReportExporter — generates cost reports in multiple formats.

Companies need cost reports for:
- Management dashboards (HTML)
- Accounting systems (CSV)
- Alert channels (JSON)
- Compliance audits (structured data)

Usage:
    from agentfuse import get_spend_report
    from agentfuse.core.report_exporter import ReportExporter

    exporter = ReportExporter(get_spend_report())
    exporter.to_csv("cost_report.csv")
    exporter.to_json("cost_report.json")
    summary = exporter.to_summary()
"""

import csv
import json
import io
import time
from typing import Optional


class ReportExporter:
    """
    Exports spend data in multiple formats for reporting and compliance.
    """

    def __init__(self, spend_report: Optional[dict] = None):
        self._report = spend_report or {}

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Export as formatted JSON. Optionally writes to file."""
        data = {
            "report_generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_spend_usd": self._report.get("total_usd", 0.0),
            "by_model": self._report.get("by_model", {}),
            "by_provider": self._report.get("by_provider", {}),
            "by_run": self._report.get("by_run", {}),
        }
        output = json.dumps(data, indent=indent, sort_keys=True)
        if path:
            with open(path, "w") as f:
                f.write(output)
        return output

    def to_csv(self, path: Optional[str] = None) -> str:
        """Export model breakdown as CSV. Optionally writes to file."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["model", "cost_usd", "percentage"])

        by_model = self._report.get("by_model", {})
        total = sum(by_model.values()) or 1.0

        for model, cost in sorted(by_model.items(), key=lambda x: -x[1]):
            writer.writerow([model, f"{cost:.6f}", f"{cost/total*100:.1f}%"])

        writer.writerow(["TOTAL", f"{sum(by_model.values()):.6f}", "100.0%"])

        csv_text = output.getvalue()
        if path:
            with open(path, "w") as f:
                f.write(csv_text)
        return csv_text

    def to_summary(self) -> str:
        """Generate a human-readable text summary."""
        total = self._report.get("total_usd", 0.0)
        by_model = self._report.get("by_model", {})
        by_provider = self._report.get("by_provider", {})
        by_run = self._report.get("by_run", {})

        lines = [
            "=" * 50,
            "  AgentFuse Cost Report",
            "=" * 50,
            f"  Total Spend: ${total:.4f}",
            f"  Models Used: {len(by_model)}",
            f"  Providers: {len(by_provider)}",
            f"  Runs: {len(by_run)}",
            "",
        ]

        if by_model:
            lines.append("  Cost by Model:")
            for model, cost in sorted(by_model.items(), key=lambda x: -x[1]):
                pct = cost / total * 100 if total > 0 else 0
                lines.append(f"    {model:<30} ${cost:.4f} ({pct:.0f}%)")

        if by_provider:
            lines.append("")
            lines.append("  Cost by Provider:")
            for provider, cost in sorted(by_provider.items(), key=lambda x: -x[1]):
                pct = cost / total * 100 if total > 0 else 0
                lines.append(f"    {provider:<30} ${cost:.4f} ({pct:.0f}%)")

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export as a clean dictionary."""
        return {
            "total_spend_usd": round(self._report.get("total_usd", 0.0), 6),
            "models": {
                model: round(cost, 6)
                for model, cost in sorted(
                    self._report.get("by_model", {}).items(),
                    key=lambda x: -x[1]
                )
            },
            "providers": dict(self._report.get("by_provider", {})),
            "run_count": len(self._report.get("by_run", {})),
        }
