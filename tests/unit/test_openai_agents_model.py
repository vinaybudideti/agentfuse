"""
Loop 29 — AgentFuseModel and AgentFuseModelProvider tests.
"""

import asyncio
import pytest
from agentfuse.integrations.openai_agents import AgentFuseModel, AgentFuseModelProvider


def test_model_init():
    model = AgentFuseModel(budget=5.0, model="gpt-4o")
    assert model.engine.budget == 5.0
    assert model.run_id is not None


def test_model_no_inner_raises():
    """get_response without inner model must raise RuntimeError."""
    model = AgentFuseModel(budget=5.0, model="gpt-4o")

    async def run():
        with pytest.raises(RuntimeError, match="No inner model"):
            await model.get_response(input=[{"role": "user", "content": "hi"}])

    asyncio.run(run())


def test_provider_creates_model():
    provider = AgentFuseModelProvider(budget=5.0)
    model = provider.get_model("gpt-4o")
    assert isinstance(model, AgentFuseModel)
    assert model.engine.budget == 5.0


def test_provider_default_model():
    """get_model with no name should default to gpt-4o."""
    provider = AgentFuseModelProvider(budget=5.0)
    model = provider.get_model()
    assert model.engine.model == "gpt-4o"


def test_cache_hit_exception():
    """CacheHitException must carry the response."""
    from agentfuse.integrations.openai_agents import CacheHitException
    exc = CacheHitException("cached response")
    assert exc.response == "cached response"
