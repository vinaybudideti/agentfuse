"""
OpenAI Agents SDK integration — Model interface wrapper.

The SDK provides RunHooks (on_llm_start/on_llm_end) for observability,
but these cannot block or replace calls. The proper integration point
is the abstract Model class.

PRODUCTION FIX: Records actual cost after inner model response.
PRODUCTION FIX: Uses new two-tier cache lookup() API.
"""

import asyncio
import logging
from uuid import uuid4
from typing import Optional

from agentfuse.core.budget import BudgetEngine, BudgetExhaustedGracefully
from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.providers.tokenizer import TokenCounterAdapter

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
        self._pricing = ModelPricingEngine()
        self._tokenizer = TokenCounterAdapter()

    async def get_response(self, system_instructions=None, input=None,
                           model_settings=None, tools=None, output_schema=None,
                           handoffs=None, tracing=None, **kwargs):
        """Async model response with cache check → budget check → delegate → record cost."""
        messages = input if isinstance(input, list) else [{"role": "user", "content": str(input or "")}]

        # Step 1: Check cache
        cache_result = self.cache.lookup(
            model=self.engine.model,
            messages=messages,
        )
        if isinstance(cache_result, CacheHit):
            return cache_result.response

        # Step 2: Budget check
        token_count = self._tokenizer.count_messages_tokens(messages, self.engine.model)
        est_cost = self._pricing.input_cost(self.engine.model, token_count)
        new_messages, active_model = self.engine.check_and_act(est_cost, messages)

        if self.inner is None:
            raise RuntimeError("No inner model configured")

        # Step 3: Make real call
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

        # Step 4: Record cost
        try:
            response_text = str(response) if response else ""
            if response_text:
                output_tokens = self._tokenizer.count_tokens(response_text, active_model)
                actual_cost = self._pricing.total_cost(active_model, token_count, output_tokens)
                self.engine.record_cost(actual_cost)
        except Exception:
            pass

        # Step 5: Cache store (non-blocking)
        try:
            if response_text and response_text.strip():
                asyncio.create_task(self._async_cache_store(messages, response_text, active_model))
        except Exception:
            pass

        return response

    async def _async_cache_store(self, messages, response_text, model):
        """Non-blocking cache store."""
        try:
            self.cache.store(model=model, messages=messages, response=response_text)
        except Exception as e:
            logger.warning("Async cache store failed: %s", e)


class AgentFuseModelProvider:
    """ModelProvider wrapper for OpenAI Agents SDK."""

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
        from agentfuse.core.keys import build_cache_key
        cache_key = build_cache_key(messages, self.engine.model)
        cache_result = self.cache.check(cache_key, self.engine.model)
        if isinstance(cache_result, CacheHit):
            raise CacheHitException(response=cache_result.response)

        tokenizer = TokenCounterAdapter()
        pricing = ModelPricingEngine()
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
