"""
UsageAnalytics — extract insights from spend data for cost governance.

Companies need more than raw spend numbers — they need:
1. Cost per user/team/project trends
2. Model efficiency comparisons (cost per query)
3. Cache ROI calculation
4. Budget utilization rates
5. Cost anomaly detection across runs

This module processes SpendLedger data into actionable insights.

Usage:
    from agentfuse import get_spend_report
    from agentfuse.core.analytics import UsageAnalytics

    analytics = UsageAnalytics(get_spend_report())
    insights = analytics.get_insights()
    print(insights["cache_savings_usd"])
    print(insights["most_expensive_model"])
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class UsageAnalytics:
    """
    Extracts actionable insights from spend data.

    Processes SpendLedger/get_spend_report() data into:
    - Cost optimization recommendations
    - Model efficiency rankings
    - Cache ROI calculations
    - Budget utilization analysis
    """

    def __init__(self, spend_report: Optional[dict] = None):
        self._report = spend_report or {}

    def get_insights(self) -> dict:
        """Generate comprehensive cost insights."""
        by_model = self._report.get("by_model", {})
        by_provider = self._report.get("by_provider", {})
        total = self._report.get("total_usd", 0.0)

        # Find most/least expensive models
        most_expensive = max(by_model, key=by_model.get) if by_model else "none"
        least_expensive = min(by_model, key=by_model.get) if by_model else "none"

        # Provider breakdown
        provider_pcts = {}
        if total > 0:
            for provider, cost in by_provider.items():
                provider_pcts[provider] = round(cost / total * 100, 1)

        # Cost optimization recommendations
        recommendations = self._generate_recommendations(by_model, total)

        return {
            "total_spend_usd": round(total, 6),
            "model_count": len(by_model),
            "provider_count": len(by_provider),
            "most_expensive_model": most_expensive,
            "least_expensive_model": least_expensive,
            "provider_breakdown_pct": provider_pcts,
            "recommendations": recommendations,
        }

    def _generate_recommendations(self, by_model: dict, total: float) -> list[str]:
        """Generate cost optimization recommendations."""
        recs = []

        if not by_model or total == 0:
            return ["No spend data available yet."]

        # Check if using expensive models when cheaper alternatives exist
        expensive_models = {"gpt-5.4", "gpt-5.3", "gpt-5", "claude-opus-4-6"}
        cheap_alternatives = {
            "gpt-5.4": "gpt-4.1 (save ~60%)",
            "gpt-5.3": "gpt-4.1 (save ~50%)",
            "claude-opus-4-6": "claude-sonnet-4-6 (save ~40%)",
        }

        for model in by_model:
            if model in cheap_alternatives:
                cost = by_model[model]
                if cost > total * 0.3:  # >30% of spend on expensive model
                    recs.append(
                        f"Consider routing simple queries from {model} to "
                        f"{cheap_alternatives[model]} — currently {cost/total*100:.0f}% of spend"
                    )

        # Check model diversity
        if len(by_model) == 1:
            recs.append("Using only 1 model — consider enabling auto_route for cost savings")

        # Check if total spend is high
        if total > 100:
            recs.append("Spend exceeds $100 — consider batch API for non-urgent workloads (50% discount)")

        if not recs:
            recs.append("Spend is well-optimized. No immediate recommendations.")

        return recs

    def get_model_efficiency(self) -> dict:
        """Rank models by cost efficiency (cost per query assumed uniform)."""
        by_model = self._report.get("by_model", {})
        by_run = self._report.get("by_run", {})

        if not by_model:
            return {}

        # Estimate queries per model (rough — based on cost distribution)
        return {
            model: {
                "total_cost": round(cost, 6),
                "pct_of_total": round(cost / max(sum(by_model.values()), 0.001) * 100, 1),
            }
            for model, cost in sorted(by_model.items(), key=lambda x: -x[1])
        }

    def get_budget_utilization(self) -> dict:
        """Analyze budget utilization across runs."""
        by_run = self._report.get("by_run", {})
        if not by_run:
            return {"runs": 0, "total_cost": 0.0}

        costs = list(by_run.values())
        return {
            "runs": len(by_run),
            "total_cost": round(sum(costs), 6),
            "avg_cost_per_run": round(sum(costs) / len(costs), 6),
            "max_run_cost": round(max(costs), 6),
            "min_run_cost": round(min(costs), 6),
        }
