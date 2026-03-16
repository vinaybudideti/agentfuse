"""
CrewAI integration — hook functions for budget + cache.

CrewAI provides @before_llm_call / @after_llm_call hooks.
Before hooks can return False to BLOCK execution, but cannot directly
return cached responses — use a side-channel dict workaround.

PRODUCTION FIX: Uses new two-tier cache lookup() API.
PRODUCTION FIX: Records actual cost in after_hook using extract_usage.
PRODUCTION FIX: Side-channel uses unique keys instead of id() to prevent GC reuse.
"""

import logging
from uuid import uuid4
from typing import Optional

from agentfuse.core.budget import BudgetEngine
from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.providers.tokenizer import TokenCounterAdapter

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
    pricing = ModelPricingEngine()
    tokenizer = TokenCounterAdapter()

    # Use UUID keys instead of id(context) to prevent GC reuse issues
    _cached_responses: dict[str, str] = {}
    _context_keys: dict[int, str] = {}

    def _get_context_key(context) -> str:
        """Get or create a unique key for this context object."""
        ctx_id = id(context)
        if ctx_id not in _context_keys:
            _context_keys[ctx_id] = str(uuid4())
        return _context_keys[ctx_id]

    def before_hook(context) -> Optional[bool]:
        """Return False to BLOCK the LLM call. Return None to allow."""
        messages = getattr(context, "messages", [])
        temperature = getattr(context, "temperature", 0.0)
        tools = getattr(context, "tools", None)

        # Check cache using two-tier API
        cache_result = cache.lookup(
            model=engine.model,
            messages=messages,
            temperature=temperature,
            tools=tools,
        )
        if isinstance(cache_result, CacheHit):
            ctx_key = _get_context_key(context)
            _cached_responses[ctx_key] = cache_result.response
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
        """Return modified string to change response, None to keep original."""
        # Check side-channel for cached response
        ctx_key = _get_context_key(context)
        if ctx_key in _cached_responses:
            cached = _cached_responses.pop(ctx_key)
            _context_keys.pop(id(context), None)
            return cached

        # Record actual cost
        try:
            messages = getattr(context, "messages", [])
            response = getattr(context, "response", None)
            temperature = getattr(context, "temperature", 0.0)
            tools = getattr(context, "tools", None)

            # Extract usage and calculate cost
            usage = getattr(response, "usage", None) if response else None
            if usage:
                from agentfuse.providers.response import extract_usage
                provider = "anthropic" if engine.model.startswith("claude") else "openai"
                normalized = extract_usage(provider, usage)
                cost = pricing.total_cost_normalized(engine.model, normalized)
                engine.record_cost(cost)

            # Extract response text and cache it
            response_text = ""
            if response:
                if hasattr(response, "choices") and response.choices:
                    response_text = getattr(response.choices[0].message, "content", "")
                elif hasattr(response, "content") and isinstance(response.content, list):
                    for block in response.content:
                        if hasattr(block, "text") and block.text:
                            response_text = block.text
                            break

            if response_text and response_text.strip():
                cache.store(
                    model=engine.model,
                    messages=messages,
                    response=response_text,
                    temperature=temperature,
                    tools=tools,
                )
        except (AttributeError, KeyError, TypeError, IndexError) as e:
            logger.warning("CrewAI after_hook failed: %s", e)

        _context_keys.pop(id(context), None)
        return None  # Keep original

    return before_hook, after_hook


# Backward compat alias
agentfuse_hooks = create_agentfuse_hooks
