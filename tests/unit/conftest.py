"""
Shared fixtures for unit tests.
"""

import pytest
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.providers.tokenizer import TokenCounterAdapter
from agentfuse.core.budget import BudgetEngine


@pytest.fixture
def pricing():
    return ModelPricingEngine()


@pytest.fixture
def tokenizer():
    return TokenCounterAdapter()


@pytest.fixture
def budget_engine():
    return BudgetEngine("test_run", 10.0, "gpt-4o")
