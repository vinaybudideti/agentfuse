"""
CrewAI integration — hook functions for budget + cache.

CrewAI provides @before_llm_call / @after_llm_call hooks.
Before hooks can return False to BLOCK execution, but cannot directly
return cached responses — use a side-channel dict workaround.
"""

import logging
from uuid import uuid4
from typing import Optional

from agentfuse.core.budget import BudgetEngine
from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit

logger = logging.getLogger(__name__)


def create_agentfuse_hooks(
    budget: float,
    run_id: Optional[str] = None,
    model: str = "gpt-4o",
    alert_cb=None,
):
    """
    Returns (before_hook, after_hook) functions for CrewAI integration.

    Usage:
        from agentfuse.integrations.crewai import create_agentfuse_hooks
        before, after = create_agentfuse_hooks(budget=5.00)
    """
    run_id = run_id or str(uuid4())
    engine = BudgetEngine(run_id, budget, model, alert_cb)
    cache = TwoTierCacheMiddleware()
    _cached_responses: dict = {}

    def before_hook(context) -> Optional[bool]:
        """
        Return False to BLOCK the LLM call.
        Return None to allow execution.
        """
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.providers.tokenizer import TokenCounterAdapter
        from agentfuse.core.keys import build_cache_key

        pricing = ModelPricingEngine()
        tokenizer = TokenCounterAdapter()

        messages = getattr(context, "messages", [])
        cache_key = build_cache_key(messages, engine.model)

        cache_result = cache.check(cache_key, engine.model)
        if isinstance(cache_result, CacheHit):
            _cached_responses[id(context)] = cache_result.response
            return False  # Block the real call

        token_count = tokenizer.count_messages_tokens(messages, engine.model)
        est_cost = pricing.input_cost(engine.model, token_count)
        new_messages, active_model = engine.check_and_act(est_cost, messages)

        if hasattr(context, "messages"):
            context.messages = new_messages
        if hasattr(context, "model"):
            context.model = active_model

        return None  # Allow

    def after_hook(context) -> Optional[str]:
        """
        Return modified string to change response, None to keep original.
        """
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.core.keys import build_cache_key

        pricing = ModelPricingEngine()

        # Check side-channel for cached response
        ctx_id = id(context)
        if ctx_id in _cached_responses:
            cached = _cached_responses.pop(ctx_id)
            return cached

        # Record actual cost
        try:
            response = getattr(context, "response", None)
            usage = getattr(response, "usage", None) if response else None
            if usage:
                cost = pricing.total_cost(
                    engine.model,
                    getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)),
                    getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)),
                )
                engine.record_cost(cost)

            messages = getattr(context, "messages", [])
            cache_key = build_cache_key(messages, engine.model)

            response_text = ""
            if hasattr(response, "choices") and response.choices:
                response_text = response.choices[0].message.content
            elif hasattr(response, "content") and isinstance(response.content, list):
                for block in response.content:
                    if hasattr(block, "text"):
                        response_text = block.text
                        break

            if response_text:
                cache.store(cache_key, response_text, engine.model)
        except (AttributeError, KeyError, TypeError, IndexError) as e:
            logger.warning("CrewAI after_hook failed: %s", e)

        return None  # Keep original

    return before_hook, after_hook


# Backward compat alias
agentfuse_hooks = create_agentfuse_hooks
