"""
Anthropic monkey-patch — intercepts anthropic.messages.create for
budget enforcement and semantic caching.

Usage (2 lines):
    from agentfuse import wrap_anthropic
    wrap_anthropic(budget_usd=5.00, run_id="run_123")

Handles both streaming and non-streaming responses.
Uses extract_usage() for accurate cache billing.
"""

from agentfuse.providers.mock_responses import MockAnthropicResponse

# Module-level state
_budget_engines = {}  # run_id -> BudgetEngine
_cache_middleware = None
_original_anthropic_create = None  # store original to prevent double-wrapping


def wrap_anthropic(budget_usd: float, run_id: str = None,
                   model: str = "claude-sonnet-4-6", alert_cb=None,
                   store=None):
    """
    Monkey-patches anthropic.messages.create to intercept all calls.
    Call once at the start of an agent run.

    Returns (run_id, client) — the patched Anthropic client.
    """
    try:
        import anthropic as anthropic_sdk
    except ImportError:
        raise ImportError(
            "anthropic package is required for wrap_anthropic(). "
            "Install with: pip install agentfuse-runtime[anthropic]"
        )
    from agentfuse.core.budget import BudgetEngine
    from agentfuse.core.cache import CacheMiddleware, CacheHit
    from agentfuse.providers.pricing import ModelPricingEngine
    from agentfuse.providers.tokenizer import TokenCounterAdapter
    from agentfuse.providers.response import extract_usage
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

    global _original_anthropic_create
    client = anthropic_sdk.Anthropic()
    if _original_anthropic_create is None:
        _original_anthropic_create = client.messages.create
    original_create = _original_anthropic_create

    def intercepted_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        call_model = kwargs.get("model", engine.model)
        temperature = kwargs.get("temperature", 0.0)
        tools = kwargs.get("tools", None)
        is_streaming = kwargs.get("stream", False)

        # Step 1: Check cache
        cache_result = cache.lookup(
            model=call_model,
            messages=messages,
            temperature=temperature,
            tools=tools,
        )
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

        # Step 4 + 5: Handle response
        if is_streaming:
            return _wrap_anthropic_stream(result, active_model, engine, pricing,
                                          cache, messages, temperature, tools)
        else:
            _record_and_cache_anthropic(result, active_model, engine, pricing,
                                         cache, messages, temperature, tools)
            return result

    client.messages.create = intercepted_create
    return run_id, client


def _record_and_cache_anthropic(result, model, engine, pricing, cache,
                                  messages, temperature, tools):
    """Record cost and cache response for non-streaming Anthropic calls."""
    from agentfuse.providers.response import extract_usage
    try:
        if hasattr(result, "usage") and result.usage:
            normalized = extract_usage("anthropic", result.usage)
            actual_cost = pricing.total_cost_normalized(model, normalized)
            engine.record_cost(actual_cost)

        response_text = _extract_response_text(result)
        if response_text:
            engine.add_partial_result(response_text)
            cache.store(
                model=model, messages=messages, response=response_text,
                temperature=temperature, tools=tools,
            )
    except (AttributeError, IndexError, TypeError):
        pass


def _wrap_anthropic_stream(stream, model, engine, pricing,
                            cache, messages, temperature, tools):
    """Wrap a streaming Anthropic response to track cost and collect content."""
    collected_content = []
    token_count = 0

    def stream_wrapper():
        nonlocal token_count
        for event in stream:
            try:
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    text = event.delta.text
                    if text:
                        collected_content.append(text)
                        token_count += max(1, int(len(text) / 3.5))
            except (AttributeError, IndexError):
                pass
            yield event

        # After stream ends: record cost and cache
        try:
            output_cost = pricing.output_cost(model, token_count)
            input_token_est = sum(
                len(m.get("content", "")) // 4 + 3
                for m in messages if isinstance(m, dict)
            )
            input_cost = pricing.input_cost(model, input_token_est)
            engine.record_cost(input_cost + output_cost)

            full_response = "".join(collected_content)
            if full_response.strip():
                engine.add_partial_result(full_response)
                cache.store(
                    model=model, messages=messages, response=full_response,
                    temperature=temperature, tools=tools,
                )
        except Exception:
            pass

    return stream_wrapper()


def cleanup_anthropic(run_id: str = None):
    """Release resources for a completed run. Call when agent run is done."""
    if run_id and run_id in _budget_engines:
        del _budget_engines[run_id]
    elif run_id is None:
        _budget_engines.clear()


def _extract_response_text(result) -> str:
    """Extract text from Anthropic response, handling tool_use blocks."""
    if not hasattr(result, "content") or not result.content:
        return ""
    for block in result.content:
        if hasattr(block, "text") and block.text:
            return block.text
    return ""


def _mock_anthropic_response(content: str, model: str):
    """Creates an Anthropic-compatible response object for cache hits."""
    return MockAnthropicResponse(content, model)
