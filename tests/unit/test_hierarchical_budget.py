"""
Tests for HierarchicalBudget — parent-child budget allocation.
"""

import pytest

from agentfuse.core.hierarchical_budget import HierarchicalBudget


def test_allocate_child():
    """Allocating a child must create a BudgetEngine with correct budget."""
    parent = HierarchicalBudget("project", total_usd=10.0)
    child = parent.allocate_child("researcher", budget_usd=4.0)
    assert child.engine.budget == 4.0
    assert child.engine.run_id == "project/researcher"
    assert parent.unallocated == 6.0


def test_multiple_children():
    """Multiple children must share the parent budget."""
    parent = HierarchicalBudget("proj", total_usd=10.0)
    parent.allocate_child("a", budget_usd=3.0)
    parent.allocate_child("b", budget_usd=3.0)
    parent.allocate_child("c", budget_usd=2.0)
    assert parent.unallocated == 2.0


def test_overallocation_raises():
    """Allocating more than available must raise ValueError."""
    parent = HierarchicalBudget("proj", total_usd=5.0)
    parent.allocate_child("a", budget_usd=3.0)
    with pytest.raises(ValueError, match="Insufficient budget"):
        parent.allocate_child("b", budget_usd=3.0)  # only 2.0 left


def test_duplicate_child_raises():
    """Allocating same name twice must raise ValueError."""
    parent = HierarchicalBudget("proj", total_usd=10.0)
    parent.allocate_child("a", budget_usd=3.0)
    with pytest.raises(ValueError, match="already allocated"):
        parent.allocate_child("a", budget_usd=3.0)


def test_release_returns_unused():
    """Releasing a child must return unused budget to parent."""
    parent = HierarchicalBudget("proj", total_usd=10.0)
    child = parent.allocate_child("worker", budget_usd=5.0)
    child.engine.record_cost(2.0)  # spent $2 of $5
    returned = parent.release_child("worker")
    assert returned == 3.0  # $3 unused
    assert parent.unallocated == 8.0  # 5 original + 3 returned


def test_reallocate_adds_budget():
    """Reallocating must add budget to a child."""
    parent = HierarchicalBudget("proj", total_usd=10.0)
    child = parent.allocate_child("worker", budget_usd=3.0)
    parent.reallocate("worker", additional_usd=2.0)
    assert child.engine.budget == 5.0
    assert parent.unallocated == 5.0


def test_reallocate_insufficient_raises():
    """Reallocating more than unallocated must raise ValueError."""
    parent = HierarchicalBudget("proj", total_usd=5.0)
    parent.allocate_child("worker", budget_usd=4.0)
    with pytest.raises(ValueError, match="Insufficient budget"):
        parent.reallocate("worker", additional_usd=2.0)  # only 1.0 left


def test_total_spent():
    """total_spent must sum all children's spend."""
    parent = HierarchicalBudget("proj", total_usd=10.0)
    c1 = parent.allocate_child("a", budget_usd=5.0)
    c2 = parent.allocate_child("b", budget_usd=5.0)
    c1.engine.record_cost(1.0)
    c2.engine.record_cost(2.0)
    assert parent.total_spent == 3.0


def test_get_report():
    """Report must contain expected structure."""
    parent = HierarchicalBudget("proj", total_usd=10.0)
    child = parent.allocate_child("worker", budget_usd=5.0)
    child.engine.record_cost(1.5)

    report = parent.get_report()
    assert report["name"] == "proj"
    assert report["total_budget_usd"] == 10.0
    assert report["total_spent_usd"] == 1.5
    assert "worker" in report["children"]
    assert report["children"]["worker"]["spent_usd"] == 1.5
    assert report["children"]["worker"]["utilization_pct"] == 30.0


def test_get_child():
    """get_child must return the child or None."""
    parent = HierarchicalBudget("proj", total_usd=10.0)
    parent.allocate_child("worker", budget_usd=5.0)
    assert parent.get_child("worker") is not None
    assert parent.get_child("nonexistent") is None


def test_child_budget_enforcement():
    """Child budget engine must enforce its own limits."""
    from agentfuse.core.budget import BudgetExhaustedGracefully
    parent = HierarchicalBudget("proj", total_usd=10.0)
    child = parent.allocate_child("worker", budget_usd=0.01)
    child.engine.record_cost(0.01)
    with pytest.raises(BudgetExhaustedGracefully):
        child.engine.check_and_act(0.01, [{"role": "user", "content": "hi"}])
