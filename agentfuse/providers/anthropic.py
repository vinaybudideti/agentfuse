"""
Anthropic monkey-patch — intercepts anthropic.messages.create for
budget enforcement and semantic caching.

Usage (2 lines):
    from agentfuse import wrap_anthropic
    wrap_anthropic(budget_usd=5.00, run_id="run_123")
"""

from types import SimpleNamespace

# Module-level state
_budget_engines = {}  # run_id -> BudgetEngine
_cache_middleware = None


def wrap_anthropic(budget_usd: float, run_id: str = None,
                   model: str = "claude-sonnet-4-6", alert_cb=None,
                   store=None):
    """
    Monkey-patches anthropic.messages.create to intercept all calls.
    Call once at the start of an agent run.

    Returns (run_id, client) — the patched Anthropic client.
    """
    import anthropic as anthropic_sdk
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

    client = anthropic_sdk.Anthropic()
    original_create = client.messages.create

    def intercepted_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        call_model = kwargs.get("model", engine.model)

        # Step 1: Check cache
        prompt = " ".join(
            m.get("content", "") for m in messages
            if isinstance(m.get("content"), str)
        )
        cache_result = cache.check(prompt, call_model, engine)
        if isinstance(cache_result, CacheHit):
            return _mock_anthropic_response(cache_result.response, call_model)

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
                result.usage.input_tokens,
                result.usage.output_tokens,
            )
            engine.record_cost(actual_cost)

        # Step 5: Store in cache
        if result.content:
            response_text = result.content[0].text
            engine.add_partial_result(response_text)
            cache.store(prompt, response_text, active_model)

        return result

    client.messages.create = intercepted_create
    return run_id, client


def _mock_anthropic_response(content: str, model: str):
    """Creates a minimal Anthropic-compatible response object for cache hits."""
    return SimpleNamespace(
        id="cache_hit",
        model=model,
        type="message",
        role="assistant",
        content=[SimpleNamespace(type="text", text=content)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=0, output_tokens=0),
    )
