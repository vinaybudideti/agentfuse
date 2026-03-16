"""
OpenAI Agents SDK integration — Model interface wrapper.

The SDK provides RunHooks (on_llm_start/on_llm_end) for observability,
but these cannot block or replace calls. The proper integration point
is the abstract Model class.
"""

import asyncio
import logging
from uuid import uuid4
from typing import Optional

from agentfuse.core.budget import BudgetEngine, BudgetExhaustedGracefully
from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit

logger = logging.getLogger(__name__)


class CacheHitException(Exception):
    def __init__(self, response):
        self.response = response


class AgentFuseModel:
    """
    Model wrapper for OpenAI Agents SDK.

    Usage:
        from agentfuse.integrations.openai_agents import AgentFuseModelProvider
        provider = AgentFuseModelProvider(inner=OpenAIChatCompletionsModel(...))
        Runner.run(agent, run_config=RunConfig(model_provider=provider))
    """

    def __init__(self, inner=None, budget: float = 10.0,
                 run_id: Optional[str] = None, model: str = "gpt-4o",
                 alert_cb=None):
        self.inner = inner
        self.run_id = run_id or str(uuid4())
        self.engine = BudgetEngine(self.run_id, budget, model, alert_cb)
        self.cache = TwoTierCacheMiddleware()

    async def get_response(self, system_instructions=None, input=None,
                           model_settings=None, tools=None, output_schema=None,
                           handoffs=None, tracing=None, **kwargs):
        """Async model response with cache check → budget check → delegate."""
        from agentfuse.core.keys import build_cache_key

        messages = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]
        cache_key = build_cache_key(messages, self.engine.model)

        cache_result = self.cache.check(cache_key, self.engine.model)
        if isinstance(cache_result, CacheHit):
            return cache_result.response

        # Budget check
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.providers.tokenizer import TokenCounterAdapter

        pricing = ModelPricingEngine()
        tokenizer = TokenCounterAdapter()
        token_count = tokenizer.count_messages_tokens(messages, self.engine.model)
        est_cost = pricing.input_cost(self.engine.model, token_count)
        new_messages, active_model = self.engine.check_and_act(est_cost, messages)

        if self.inner is None:
            raise RuntimeError("No inner model configured")

        response = await self.inner.get_response(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=tracing,
            **kwargs,
        )

        # Cache store (non-blocking)
        try:
            response_text = str(response)
            if response_text:
                asyncio.create_task(self._async_cache_store(cache_key, response_text))
        except Exception:
            pass

        return response

    async def _async_cache_store(self, cache_key, response_text):
        """Non-blocking cache store."""
        try:
            self.cache.store(cache_key, response_text, self.engine.model)
        except Exception as e:
            logger.warning("Async cache store failed: %s", e)


class AgentFuseModelProvider:
    """
    ModelProvider wrapper for OpenAI Agents SDK.
    """

    def __init__(self, inner=None, budget: float = 10.0, **kwargs):
        self.inner = inner
        self._budget = budget
        self._kwargs = kwargs

    def get_model(self, model_name: str = None) -> AgentFuseModel:
        inner_model = None
        if self.inner and hasattr(self.inner, "get_model"):
            inner_model = self.inner.get_model(model_name)
        return AgentFuseModel(
            inner=inner_model,
            budget=self._budget,
            model=model_name or "gpt-4o",
            **self._kwargs,
        )


# Backward compat
class AgentFuseRunHooks:
    """Legacy RunHooks — kept for backward compat but AgentFuseModel is preferred."""

    def __init__(self, budget: float, run_id: str = None,
                 model: str = "gpt-4o", alert_cb=None):
        self.run_id = run_id or str(uuid4())
        self.engine = BudgetEngine(self.run_id, budget, model, alert_cb)
        self.cache = TwoTierCacheMiddleware()

    def on_llm_start(self, context, messages):
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.providers.tokenizer import TokenCounterAdapter
        from agentfuse.core.keys import build_cache_key

        pricing = ModelPricingEngine()
        tokenizer = TokenCounterAdapter()
        cache_key = build_cache_key(messages, self.engine.model)
        cache_result = self.cache.check(cache_key, self.engine.model)
        if isinstance(cache_result, CacheHit):
            raise CacheHitException(response=cache_result.response)

        token_count = tokenizer.count_messages_tokens(messages, self.engine.model)
        est_cost = pricing.input_cost(self.engine.model, token_count)
        self.engine.check_and_act(est_cost, messages)

    def on_llm_end(self, context, response):
        pass

    def get_receipt(self):
        return {
            "run_id": self.run_id,
            "spent_usd": self.engine.spent,
            "budget_usd": self.engine.budget,
            "model": self.engine.model,
        }
