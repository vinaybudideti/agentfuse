"""
HierarchicalBudget — parent-child budget allocation for multi-agent systems.

Production AI systems use multiple agents (researcher, writer, reviewer, etc.)
that share a global budget. Without hierarchy, runaway sub-agents can consume
the entire budget before other agents get their turn.

This module provides:
1. Parent budget with allocatable sub-budgets
2. Automatic rebalancing when child agents finish early
3. Budget inheritance for nested agent workflows
4. Real-time visibility into per-agent spend

Usage:
    parent = HierarchicalBudget("project_1", total_usd=10.0)
    researcher = parent.allocate_child("researcher", budget_usd=4.0)
    writer = parent.allocate_child("writer", budget_usd=4.0)
    reviewer = parent.allocate_child("reviewer", budget_usd=2.0)

    # Each child has its own BudgetEngine
    researcher.engine.check_and_act(est_cost, messages)

    # When researcher finishes early, unused budget returns to parent
    parent.release_child("researcher")  # unused budget goes back to pool

    # Reallocate to writer who needs more
    parent.reallocate("writer", additional_usd=1.5)
"""

import logging
import threading
from typing import Optional
from dataclasses import dataclass, field

from agentfuse.core.budget import BudgetEngine

logger = logging.getLogger(__name__)


@dataclass
class ChildBudget:
    """A child budget allocation."""
    name: str
    allocated_usd: float
    engine: BudgetEngine
    released: bool = False


class HierarchicalBudget:
    """
    Parent budget that allocates sub-budgets to child agents.

    Prevents runaway sub-agents from consuming the entire budget.
    Supports rebalancing when children finish early or need more.
    """

    def __init__(self, name: str, total_usd: float, model: str = "gpt-4o"):
        self.name = name
        self.total_usd = total_usd
        self.model = model
        self._children: dict[str, ChildBudget] = {}
        self._unallocated: float = total_usd
        self._lock = threading.Lock()

    def allocate_child(
        self,
        child_name: str,
        budget_usd: float,
        model: Optional[str] = None,
    ) -> ChildBudget:
        """
        Allocate a sub-budget to a child agent.

        Raises ValueError if insufficient unallocated budget.
        """
        with self._lock:
            if child_name in self._children:
                raise ValueError(f"Child '{child_name}' already allocated")

            if budget_usd > self._unallocated:
                raise ValueError(
                    f"Insufficient budget: requested ${budget_usd:.2f}, "
                    f"available ${self._unallocated:.2f}"
                )

            engine = BudgetEngine(
                run_id=f"{self.name}/{child_name}",
                budget_usd=budget_usd,
                model=model or self.model,
            )

            child = ChildBudget(
                name=child_name,
                allocated_usd=budget_usd,
                engine=engine,
            )
            self._children[child_name] = child
            self._unallocated -= budget_usd

            logger.info("Budget allocated: %s → $%.2f (remaining: $%.2f)",
                        child_name, budget_usd, self._unallocated)
            return child

    def release_child(self, child_name: str) -> float:
        """
        Release a child's budget. Unused portion returns to parent.

        Returns the amount returned to the parent pool.
        """
        with self._lock:
            if child_name not in self._children:
                raise ValueError(f"Unknown child: {child_name}")

            child = self._children[child_name]
            if child.released:
                return 0.0

            unused = child.allocated_usd - child.engine.spent
            unused = max(0.0, unused)

            child.released = True
            self._unallocated += unused

            logger.info("Budget released: %s → $%.4f returned (spent $%.4f of $%.2f)",
                        child_name, unused, child.engine.spent, child.allocated_usd)
            return unused

    def reallocate(self, child_name: str, additional_usd: float):
        """
        Add more budget to a child from the unallocated pool.

        Raises ValueError if insufficient unallocated budget.
        """
        with self._lock:
            if child_name not in self._children:
                raise ValueError(f"Unknown child: {child_name}")

            if additional_usd > self._unallocated:
                raise ValueError(
                    f"Insufficient budget: requested ${additional_usd:.2f}, "
                    f"available ${self._unallocated:.2f}"
                )

            child = self._children[child_name]
            child.engine.budget += additional_usd
            child.allocated_usd += additional_usd
            self._unallocated -= additional_usd

            logger.info("Budget reallocated: %s += $%.2f (new total: $%.2f)",
                        child_name, additional_usd, child.allocated_usd)

    def get_child(self, child_name: str) -> Optional[ChildBudget]:
        """Get a child budget by name."""
        return self._children.get(child_name)

    def get_report(self) -> dict:
        """Get detailed budget report for all children."""
        with self._lock:
            children_report = {}
            total_spent = 0.0

            for name, child in self._children.items():
                spent = child.engine.spent
                total_spent += spent
                children_report[name] = {
                    "allocated_usd": round(child.allocated_usd, 6),
                    "spent_usd": round(spent, 6),
                    "remaining_usd": round(child.allocated_usd - spent, 6),
                    "utilization_pct": round(spent / child.allocated_usd * 100, 1) if child.allocated_usd > 0 else 0,
                    "model": child.engine.model,
                    "state": child.engine.state.value,
                    "released": child.released,
                }

            return {
                "name": self.name,
                "total_budget_usd": self.total_usd,
                "total_spent_usd": round(total_spent, 6),
                "unallocated_usd": round(self._unallocated, 6),
                "children": children_report,
            }

    @property
    def total_spent(self) -> float:
        """Total spend across all children."""
        return sum(c.engine.spent for c in self._children.values())

    @property
    def unallocated(self) -> float:
        """Unallocated budget available for new children."""
        return self._unallocated
