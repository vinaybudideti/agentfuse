"""
Anthropic monkey-patch — intercepts anthropic.messages.create for
budget enforcement and semantic caching.

Usage (2 lines):
    from agentfuse import wrap_anthropic
    wrap_anthropic(budget_usd=5.00, run_id="run_123")

ARCHITECTURE FIX: Uses single global interceptor with ContextVar routing.
Multiple wrap_anthropic() calls coexist without overwriting each other.

CORRECTNESS FIX: Passes stop_reason to response validator so truncated
responses are never cached.
"""

import threading
from contextvars import ContextVar

from agentfuse.providers.mock_responses import MockAnthropicResponse

# Module-level state
_wrap_lock = threading.Lock()
_run_contexts = {}  # run_id -> {engine, cache, pricing, tokenizer}
_original_anthropic_create = None
_patched_client = None

_active_anthropic_run = ContextVar("_active_anthropic_run", default=None)


def wrap_anthropic(budget_usd: float, run_id: str = None,
                   model: str = "claude-sonnet-4-6", alert_cb=None,
                   store=None):
    """
    Monkey-patches anthropic.messages.create to intercept all calls.
    Returns (run_id, client).
    """
    try:
        import anthropic as anthropic_sdk
    except ImportError:
        raise ImportError(
            "anthropic package is required for wrap_anthropic(). "
            "Install with: pip install agentfuse-runtime[anthropic]"
        )
    from agentfuse.core.budget import BudgetEngine
    from agentfuse.core.cache import CacheMiddleware
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

    with _wrap_lock:
        _run_contexts[run_id] = {
            "engine": engine, "cache": cache,
            "pricing": pricing, "tokenizer": tokenizer,
        }
        _active_anthropic_run.set(run_id)

        global _original_anthropic_create, _patched_client
        client = anthropic_sdk.Anthropic()
        if _original_anthropic_create is None:
            _original_anthropic_create = client.messages.create
            client.messages.create = _global_intercepted_create
            _patched_client = client

    return run_id, _patched_client or client


def _get_run_context():
    """Get the active run context."""
    run_id = _active_anthropic_run.get()
    if run_id and run_id in _run_contexts:
        return run_id, _run_contexts[run_id]
    if _run_contexts:
        last_id = list(_run_contexts.keys())[-1]
        return last_id, _run_contexts[last_id]
    return None, None


def _global_intercepted_create(*args, **kwargs):
    """Single global interceptor routing to correct run's engine."""
    from agentfuse.core.cache import CacheHit
    from agentfuse.providers.response import extract_usage

    run_id, ctx = _get_run_context()
    if ctx is None:
        return _original_anthropic_create(*args, **kwargs)

    engine = ctx["engine"]
    cache = ctx["cache"]
    pricing = ctx["pricing"]
    tokenizer = ctx["tokenizer"]

    messages = kwargs.get("messages", [])
    call_model = kwargs.get("model", engine.model)
    temperature = kwargs.get("temperature", 0.0)
    tools = kwargs.get("tools", None)
    is_streaming = kwargs.get("stream", False)

    # Step 1: Check cache
    cache_result = cache.lookup(
        model=call_model, messages=messages,
        temperature=temperature, tools=tools,
    )
    if isinstance(cache_result, CacheHit):
        return _mock_anthropic_response(cache_result.response, call_model)

    # Step 2: Budget check
    token_count = tokenizer.count_messages_tokens(messages, call_model)
    est_cost = pricing.input_cost(call_model, token_count)
    messages, active_model = engine.check_and_act(est_cost, messages)
    kwargs["messages"] = messages
    kwargs["model"] = active_model

    # Step 3: Make real call
    result = _original_anthropic_create(*args, **kwargs)

    # Step 4 + 5
    if is_streaming:
        return _wrap_anthropic_stream(result, active_model, engine, pricing,
                                      cache, messages, temperature, tools)
    else:
        _record_and_cache_anthropic(result, active_model, engine, pricing,
                                     cache, messages, temperature, tools)
        return result


def _record_and_cache_anthropic(result, model, engine, pricing, cache,
                                  messages, temperature, tools):
    """Record cost and cache response for non-streaming Anthropic calls.
    Uses auto-discovery pattern adapter as fallback."""
    from agentfuse.providers.response import extract_usage
    from agentfuse.providers.token_pattern import extract_with_pattern
    from agentfuse.core.response_validator import validate_for_cache
    try:
        if hasattr(result, "usage") and result.usage:
            try:
                normalized = extract_usage("anthropic", result.usage)
            except Exception:
                normalized = extract_with_pattern(result.usage, "anthropic")
            actual_cost = pricing.total_cost_normalized(model, normalized)
            engine.record_cost(actual_cost)

        response_text = _extract_response_text(result)
        stop_reason = getattr(result, "stop_reason", None)

        if response_text:
            engine.add_partial_result(response_text)
            # Map Anthropic stop_reason to OpenAI-style finish_reason
            finish_reason = "stop" if stop_reason == "end_turn" else stop_reason
            if validate_for_cache(response_text, finish_reason=finish_reason):
                cache.store(
                    model=model, messages=messages, response=response_text,
                    temperature=temperature, tools=tools,
                )
    except (AttributeError, IndexError, TypeError):
        pass


def _wrap_anthropic_stream(stream, model, engine, pricing,
                            cache, messages, temperature, tools):
    """Wrap streaming Anthropic response."""
    MAX_COLLECTED_CHARS = 500_000
    collected_content = []
    collected_chars = 0
    token_count = 0

    def stream_wrapper():
        nonlocal token_count, collected_chars
        for event in stream:
            try:
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    text = event.delta.text
                    if text:
                        token_count += max(1, int(len(text) / 3.5))
                        if collected_chars < MAX_COLLECTED_CHARS:
                            collected_content.append(text)
                            collected_chars += len(text)
            except (AttributeError, IndexError):
                pass
            yield event

        try:
            output_cost = pricing.output_cost(model, token_count)
            input_token_est = sum(
                len(str(m.get("content", ""))) // 4 + 3
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


def _extract_response_text(result) -> str:
    """Extract text from Anthropic response, handling tool_use blocks."""
    if not hasattr(result, "content") or not result.content:
        return ""
    for block in result.content:
        if hasattr(block, "text") and block.text:
            return block.text
    return ""


def set_active_run(run_id: str):
    """Set the active Anthropic run for the current thread/async task."""
    _active_anthropic_run.set(run_id)


def cleanup_anthropic(run_id: str = None):
    """Release resources for a completed run."""
    with _wrap_lock:
        if run_id and run_id in _run_contexts:
            del _run_contexts[run_id]
        elif run_id is None:
            _run_contexts.clear()


def _mock_anthropic_response(content: str, model: str):
    return MockAnthropicResponse(content, model)
