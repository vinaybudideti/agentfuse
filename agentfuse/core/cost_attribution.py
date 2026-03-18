"""
CostAttribution — granular cost attribution by user, team, feature, and session.

Production companies need to know WHO is spending money and WHERE:
- Per-user cost: "User X spent $45 this month on GPT-5"
- Per-team cost: "Engineering team used 80% of the budget"
- Per-feature cost: "Chat feature costs 3x more than search"
- Per-session cost: "Average session costs $0.52"

This module aggregates spend data with metadata tags for attribution.

Usage:
    attr = CostAttribution()
    attr.record(cost=0.05, user_id="user_123", team="engineering", feature="chat")
    report = attr.get_by_user()
    # {"user_123": {"total_cost": 0.05, "calls": 1}}
"""

import logging
import threading
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class CostAttribution:
    """
    Granular cost attribution for multi-tenant LLM applications.
    Thread-safe, O(1) per-record, O(n) per-query.
    """

    def __init__(self):
        self._by_user: dict[str, dict] = defaultdict(lambda: {"total_cost": 0.0, "calls": 0})
        self._by_team: dict[str, dict] = defaultdict(lambda: {"total_cost": 0.0, "calls": 0})
        self._by_feature: dict[str, dict] = defaultdict(lambda: {"total_cost": 0.0, "calls": 0})
        self._by_session: dict[str, dict] = defaultdict(lambda: {"total_cost": 0.0, "calls": 0})
        self._lock = threading.Lock()

    def record(
        self,
        cost: float,
        user_id: Optional[str] = None,
        team: Optional[str] = None,
        feature: Optional[str] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Record a cost event with attribution metadata."""
        with self._lock:
            if user_id:
                self._by_user[user_id]["total_cost"] += cost
                self._by_user[user_id]["calls"] += 1
            if team:
                self._by_team[team]["total_cost"] += cost
                self._by_team[team]["calls"] += 1
            if feature:
                self._by_feature[feature]["total_cost"] += cost
                self._by_feature[feature]["calls"] += 1
            if session_id:
                self._by_session[session_id]["total_cost"] += cost
                self._by_session[session_id]["calls"] += 1

    def get_by_user(self) -> dict:
        """Get cost breakdown by user."""
        with self._lock:
            return {k: {**v, "total_cost": round(v["total_cost"], 6)} for k, v in self._by_user.items()}

    def get_by_team(self) -> dict:
        """Get cost breakdown by team."""
        with self._lock:
            return {k: {**v, "total_cost": round(v["total_cost"], 6)} for k, v in self._by_team.items()}

    def get_by_feature(self) -> dict:
        """Get cost breakdown by feature."""
        with self._lock:
            return {k: {**v, "total_cost": round(v["total_cost"], 6)} for k, v in self._by_feature.items()}

    def get_by_session(self) -> dict:
        """Get cost breakdown by session."""
        with self._lock:
            return {k: {**v, "total_cost": round(v["total_cost"], 6)} for k, v in self._by_session.items()}

    def get_top_users(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N users by cost."""
        with self._lock:
            sorted_users = sorted(self._by_user.items(), key=lambda x: -x[1]["total_cost"])
            return [(k, round(v["total_cost"], 6)) for k, v in sorted_users[:n]]

    def get_top_teams(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N teams by cost."""
        with self._lock:
            sorted_teams = sorted(self._by_team.items(), key=lambda x: -x[1]["total_cost"])
            return [(k, round(v["total_cost"], 6)) for k, v in sorted_teams[:n]]

    def get_summary(self) -> dict:
        """Get overall attribution summary."""
        with self._lock:
            total_cost = sum(v["total_cost"] for v in self._by_user.values()) if self._by_user else 0
            return {
                "total_cost_usd": round(total_cost, 6),
                "unique_users": len(self._by_user),
                "unique_teams": len(self._by_team),
                "unique_features": len(self._by_feature),
                "unique_sessions": len(self._by_session),
            }

    def reset(self):
        """Reset all attribution data."""
        with self._lock:
            self._by_user.clear()
            self._by_team.clear()
            self._by_feature.clear()
            self._by_session.clear()
