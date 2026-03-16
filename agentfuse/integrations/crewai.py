import logging
from uuid import uuid4
from agentfuse.core.budget import BudgetEngine
from agentfuse.core.cache import CacheMiddleware, CacheHit

logger = logging.getLogger(__name__)


def agentfuse_hooks(budget: float, run_id: str = None,
                    model: str = "gpt-4o", alert_cb=None):
    """
    Returns (before_llm_call, after_llm_call) hook functions for CrewAI.

    Usage (2 lines):
        from agentfuse.integrations.crewai import agentfuse_hooks
        before, after = agentfuse_hooks(budget=5.00)
    """
    run_id = run_id or str(uuid4())
    engine = BudgetEngine(run_id, budget, model, alert_cb)
    cache = CacheMiddleware()

    def before_llm_call(context):
        """
        Called before each LLM call.
        Return False to block the call (cache hit).
        Return True to proceed with the call.
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
            if hasattr(context, "inject_response"):
                context.inject_response(cache_result.response)
            return False

        token_count = tokenizer.count_messages_tokens(messages, engine.model)
        est_cost = pricing.input_cost(engine.model, token_count)
        new_messages, active_model = engine.check_and_act(est_cost, messages)

        if hasattr(context, "messages"):
            context.messages = new_messages
        if hasattr(context, "model"):
            context.model = active_model

        return True

    def after_llm_call(context, result):
        """Called after each LLM call. Record cost and cache response."""
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.core.keys import build_cache_key

        pricing = ModelPricingEngine()

        try:
            usage = getattr(result, "usage", None)
            if usage:
                cost = pricing.total_cost(
                    engine.model,
                    getattr(usage, "input_tokens",
                            getattr(usage, "prompt_tokens", 0)),
                    getattr(usage, "output_tokens",
                            getattr(usage, "completion_tokens", 0))
                )
                engine.record_cost(cost)

            messages = getattr(context, "messages", [])
            cache_key = build_cache_key(messages, engine.model)

            response_text = ""
            if hasattr(result, "choices") and result.choices:
                response_text = result.choices[0].message.content
            elif hasattr(result, "content") and result.content:
                if hasattr(result.content[0], "text"):
                    response_text = result.content[0].text

            if response_text:
                cache.store(cache_key, response_text, engine.model)
        except (AttributeError, KeyError, TypeError, IndexError) as e:
            logger.warning("CrewAI after_llm_call failed: %s", e)

    return before_llm_call, after_llm_call
