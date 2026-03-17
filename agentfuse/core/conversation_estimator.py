"""
ConversationCostEstimator — predicts total conversation cost from early turns.

NOVEL APPROACH: Uses exponential growth detection to warn when a conversation
is on track to exceed its budget. Most conversations follow predictable patterns:
- FAQ: flat cost (each turn costs about the same)
- Research: linear growth (context grows, each turn costs slightly more)
- Agent loops: exponential growth (context explodes, cost doubles per turn)

By detecting the growth pattern early (turns 3-5), we can warn users and
automatically apply optimization strategies.

Usage:
    estimator = ConversationCostEstimator(budget_usd=5.00)
    estimator.record_turn(cost=0.01, input_tokens=50, output_tokens=100)
    estimator.record_turn(cost=0.02, input_tokens=120, output_tokens=200)

    projection = estimator.project(target_turns=20)
    # {'projected_total_cost': 2.50, 'pattern': 'linear', 'will_exceed_budget': False}
"""

import math
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    turn: int
    cost: float
    input_tokens: int
    output_tokens: int
    cumulative_cost: float


class ConversationCostEstimator:
    """
    Predicts total conversation cost based on early turn patterns.

    Detects three growth patterns:
    - FLAT: cost per turn is constant (FAQ-style)
    - LINEAR: cost grows linearly (growing context)
    - EXPONENTIAL: cost doubles per N turns (agent loops, exploding context)
    """

    def __init__(self, budget_usd: float = 10.0):
        self._budget = budget_usd
        self._turns: list[TurnRecord] = []
        self._lock = threading.Lock()

    def record_turn(self, cost: float, input_tokens: int = 0, output_tokens: int = 0):
        """Record the cost of a conversation turn."""
        with self._lock:
            cumulative = sum(t.cost for t in self._turns) + cost
            self._turns.append(TurnRecord(
                turn=len(self._turns) + 1,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cumulative_cost=cumulative,
            ))

    def detect_pattern(self) -> str:
        """Detect the cost growth pattern: 'flat', 'linear', or 'exponential'."""
        with self._lock:
            if len(self._turns) < 3:
                return "unknown"

            costs = [t.cost for t in self._turns]
            return self._classify_growth(costs)

    def project(self, target_turns: int = 20) -> dict:
        """Project total cost for the conversation.

        Args:
            target_turns: Number of total turns to project

        Returns dict with projected cost, pattern, and budget warning.
        """
        with self._lock:
            if not self._turns:
                return {
                    "projected_total_cost": 0.0,
                    "pattern": "unknown",
                    "will_exceed_budget": False,
                    "turns_until_budget_exceeded": -1,
                    "confidence": 0.0,
                }

            costs = [t.cost for t in self._turns]
            current_turns = len(self._turns)
            pattern = self._classify_growth(costs)
            remaining = target_turns - current_turns

            if remaining <= 0:
                projected = sum(costs)
            elif pattern == "flat":
                avg_cost = sum(costs) / len(costs)
                projected = sum(costs) + avg_cost * remaining
            elif pattern == "linear":
                # Linear regression to project future costs
                slope, intercept = self._linear_fit(costs)
                projected = sum(costs) + sum(
                    slope * (current_turns + i) + intercept
                    for i in range(1, remaining + 1)
                )
            else:  # exponential
                # Use ratio of last few costs to project
                if len(costs) >= 2 and costs[-2] > 0:
                    growth_rate = costs[-1] / costs[-2]
                    growth_rate = min(growth_rate, 3.0)  # cap at 3x per turn
                    projected = sum(costs)
                    last_cost = costs[-1]
                    for _ in range(remaining):
                        last_cost *= growth_rate
                        projected += last_cost
                else:
                    projected = sum(costs) * (target_turns / max(current_turns, 1))

            # Calculate turns until budget exceeded
            turns_until = self._turns_until_budget(costs, pattern)

            # Confidence based on data points
            confidence = min(1.0, len(costs) / 10.0)

            return {
                "projected_total_cost": round(projected, 6),
                "pattern": pattern,
                "will_exceed_budget": projected > self._budget,
                "turns_until_budget_exceeded": turns_until,
                "current_spent": round(sum(costs), 6),
                "current_turns": current_turns,
                "confidence": round(confidence, 2),
            }

    def _classify_growth(self, costs: list[float]) -> str:
        """Classify cost growth as flat, linear, or exponential."""
        if len(costs) < 3:
            return "unknown"

        # Calculate coefficient of variation
        avg = sum(costs) / len(costs)
        if avg == 0:
            return "flat"

        variance = sum((c - avg) ** 2 for c in costs) / len(costs)
        cv = math.sqrt(variance) / avg if avg > 0 else 0

        # Check for exponential growth: ratio of consecutive costs > 1.5
        ratios = []
        for i in range(1, len(costs)):
            if costs[i - 1] > 0:
                ratios.append(costs[i] / costs[i - 1])
        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0

        if avg_ratio > 1.5:
            return "exponential"
        elif cv < 0.3:
            return "flat"
        else:
            return "linear"

    def _linear_fit(self, costs: list[float]) -> tuple[float, float]:
        """Simple linear regression: y = slope * x + intercept."""
        n = len(costs)
        if n < 2:
            return 0.0, costs[0] if costs else 0.0

        x_mean = (n - 1) / 2
        y_mean = sum(costs) / n

        num = sum((i - x_mean) * (costs[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))

        slope = num / den if den > 0 else 0.0
        intercept = y_mean - slope * x_mean
        return slope, intercept

    def _turns_until_budget(self, costs: list[float], pattern: str) -> int:
        """Estimate turns until budget is exceeded."""
        current_spent = sum(costs)
        remaining_budget = self._budget - current_spent
        if remaining_budget <= 0:
            return 0

        if not costs:
            return -1

        avg_cost = sum(costs) / len(costs)
        if avg_cost <= 0:
            return -1

        if pattern == "flat":
            return int(remaining_budget / avg_cost)
        elif pattern == "linear":
            slope, intercept = self._linear_fit(costs)
            next_cost = slope * len(costs) + intercept
            if next_cost <= 0:
                return -1
            return int(remaining_budget / max(next_cost, avg_cost))
        else:  # exponential
            if len(costs) >= 2 and costs[-2] > 0:
                growth = costs[-1] / costs[-2]
                if growth <= 1.0:
                    return -1
                # Geometric series: budget = cost * (growth^n - 1) / (growth - 1)
                cost = costs[-1]
                turns = 0
                spent = 0
                while spent < remaining_budget and turns < 1000:
                    cost *= growth
                    spent += cost
                    turns += 1
                return turns
            return int(remaining_budget / avg_cost)

    def get_summary(self) -> dict:
        """Get summary of recorded turns."""
        with self._lock:
            if not self._turns:
                return {"turns": 0, "total_cost": 0.0}
            return {
                "turns": len(self._turns),
                "total_cost": round(sum(t.cost for t in self._turns), 6),
                "avg_cost_per_turn": round(sum(t.cost for t in self._turns) / len(self._turns), 6),
                "total_input_tokens": sum(t.input_tokens for t in self._turns),
                "total_output_tokens": sum(t.output_tokens for t in self._turns),
            }
