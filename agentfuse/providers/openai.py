"""
OpenAI monkey-patch — intercepts openai.chat.completions.create for
budget enforcement and semantic caching.

Usage (2 lines):
    from agentfuse import wrap_openai
    wrap_openai(budget_usd=5.00, run_id="run_123")
"""

from agentfuse.providers.mock_responses import MockOpenAIResponse

# Module-level state
_budget_engines = {}  # run_id -> BudgetEngine
_cache_middleware = None


def wrap_openai(budget_usd: float, run_id: str = None,
                model: str = "gpt-4o", alert_cb=None,
                store=None):
    """
    Monkey-patches openai.chat.completions.create to intercept all calls.
    Call once at the start of an agent run.
    """
    import openai
    from agentfuse.core.budget import BudgetEngine
    from agentfuse.core.cache import CacheMiddleware, CacheHit
    from agentfuse.providers.pricing import ModelPricingEngine
    from agentfuse.providers.tokenizer import TokenCounterAdapter
    from agentfuse.storage.memory import InMemoryStore
    import uuid

    if run_id is None:
        run_id = str(uuid.uuid4())

    store = store or InMemoryStore()
    engine = BudgetEngine(run_id, budget_usd, model, alert_cb)
    cache = CacheMiddleware()
    pricing = ModelPricingEngine()
    tokenizer = TokenCounterAdapter()

    _budget_engines[run_id] = engine

    original_create = openai.chat.completions.create

    def intercepted_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        call_model = kwargs.get("model", engine.model)

        # Step 1: Check cache
        from agentfuse.core.keys import build_cache_key
        cache_key = build_cache_key(messages, call_model)
        cache_result = cache.check(cache_key, call_model, engine)
        if isinstance(cache_result, CacheHit):
            return _mock_openai_response(cache_result.response, call_model)

        # Step 2: Check budget
        token_count = tokenizer.count_messages_tokens(messages, call_model)
        est_cost = pricing.input_cost(call_model, token_count)
        messages, active_model = engine.check_and_act(est_cost, messages)
        kwargs["messages"] = messages
        kwargs["model"] = active_model

        # Step 3: Make real call
        result = original_create(*args, **kwargs)

        # Step 4: Record cost
        if hasattr(result, "usage") and result.usage:
            actual_cost = pricing.total_cost(
                active_model,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
            )
            engine.record_cost(actual_cost)

        # Step 5: Store in cache
        if result.choices and result.choices[0].message.content:
            response_text = result.choices[0].message.content
            engine.add_partial_result(response_text)
            cache.store(cache_key, response_text, active_model)

        return result

    openai.chat.completions.create = intercepted_create
    return run_id


def _mock_openai_response(content: str, model: str):
    """Creates an OpenAI-compatible response object for cache hits."""
    return MockOpenAIResponse(content, model)
