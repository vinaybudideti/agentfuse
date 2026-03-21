"""
OpenAI monkey-patch — intercepts openai.chat.completions.create for
budget enforcement and semantic caching.

Usage (2 lines):
    from agentfuse import wrap_openai
    wrap_openai(budget_usd=5.00, run_id="run_123")

Handles both streaming and non-streaming responses.

ARCHITECTURE FIX: Uses a single global interceptor that routes to the
correct BudgetEngine per-run via ContextVar. This allows multiple
concurrent wrap_openai() calls to coexist without overwriting each other.

CORRECTNESS FIX: Passes finish_reason to response validator so truncated
responses (finish_reason="length") are never cached.
"""

import threading
from contextvars import ContextVar

from agentfuse.providers.mock_responses import MockOpenAIResponse

# Module-level state — protected by _wrap_lock
_wrap_lock = threading.Lock()
_run_contexts = {}  # run_id -> {engine, cache, pricing, tokenizer}
_original_openai_create = None
_patched = False

# ContextVar to identify the active run in the current thread/task
_active_openai_run = ContextVar("_active_openai_run", default=None)


def wrap_openai(budget_usd: float, run_id: str = None,
                model: str = "gpt-4o", alert_cb=None,
                store=None):
    """
    Monkey-patches openai.chat.completions.create to intercept all calls.

    Multiple calls with different run_ids are safe — each gets its own
    BudgetEngine and cache. The active run is identified by ContextVar.
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package is required for wrap_openai(). "
            "Install with: pip install agentfuse-runtime[openai]"
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

    with _wrap_lock:
        _run_contexts[run_id] = {
            "engine": engine,
            "cache": cache,
            "pricing": pricing,
            "tokenizer": tokenizer,
        }

        # Set this run as active in current thread/task
        _active_openai_run.set(run_id)

        # Only patch once — the interceptor routes to the correct run
        global _original_openai_create, _patched
        if not _patched:
            _original_openai_create = openai.chat.completions.create
            openai.chat.completions.create = _global_intercepted_create
            _patched = True

    return run_id


def _get_run_context():
    """Get the active run context for this thread/task."""
    run_id = _active_openai_run.get()
    if run_id and run_id in _run_contexts:
        return run_id, _run_contexts[run_id]

    # Fallback: use the most recently registered run
    if _run_contexts:
        try:
            last_id = next(reversed(_run_contexts))  # O(1), no list conversion
            return last_id, _run_contexts[last_id]
        except (StopIteration, KeyError):
            pass

    return None, None


def _global_intercepted_create(*args, **kwargs):
    """Single global interceptor that routes to the correct run's engine."""
    from agentfuse.core.cache import CacheHit
    from agentfuse.providers.response import extract_usage

    run_id, ctx = _get_run_context()
    if ctx is None:
        # No active run — call original directly
        return _original_openai_create(*args, **kwargs)

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
        return _mock_openai_response(cache_result.response, call_model)

    # Step 2: Check budget
    token_count = tokenizer.count_messages_tokens(messages, call_model)
    est_cost = pricing.input_cost(call_model, token_count)
    messages, active_model = engine.check_and_act(est_cost, messages)
    kwargs["messages"] = messages
    kwargs["model"] = active_model

    # Step 3: Make real call
    result = _original_openai_create(*args, **kwargs)

    # Step 4 + 5: Handle streaming vs non-streaming
    if is_streaming:
        return _wrap_openai_stream(result, active_model, engine, pricing,
                                   cache, messages, temperature, tools)
    else:
        _record_and_cache_openai(result, active_model, engine, pricing,
                                  cache, messages, temperature, tools)
        return result


def _record_and_cache_openai(result, model, engine, pricing, cache,
                               messages, temperature, tools):
    """Record cost and cache response for non-streaming OpenAI calls.
    Uses auto-discovery pattern adapter as fallback for unknown providers."""
    from agentfuse.providers.response import extract_usage
    from agentfuse.providers.token_pattern import extract_with_pattern
    from agentfuse.core.response_validator import validate_for_cache
    try:
        if hasattr(result, "usage") and result.usage:
            # Try standard extraction first, fall back to auto-discovery
            try:
                normalized = extract_usage("openai", result.usage)
            except Exception:
                normalized = extract_with_pattern(result.usage, "openai")
            actual_cost = pricing.total_cost_normalized(model, normalized)
            engine.record_cost(actual_cost)

        if result.choices and result.choices[0].message.content:
            response_text = result.choices[0].message.content
            finish_reason = getattr(result.choices[0], "finish_reason", None)

            engine.add_partial_result(response_text)

            # Only cache if response is complete and valid
            if validate_for_cache(response_text, finish_reason=finish_reason):
                cache.store(
                    model=model, messages=messages, response=response_text,
                    temperature=temperature, tools=tools,
                )
    except (AttributeError, IndexError):
        pass


def _wrap_openai_stream(stream, model, engine, pricing,
                         cache, messages, temperature, tools):
    """Wrap a streaming OpenAI response to track cost and collect content."""
    MAX_COLLECTED_CHARS = 500_000
    collected_content = []
    collected_chars = 0
    token_count = 0

    def stream_wrapper():
        nonlocal token_count, collected_chars
        for chunk in stream:
            try:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        token_count += max(1, int(len(delta.content) / 3.5))
                        if collected_chars < MAX_COLLECTED_CHARS:
                            collected_content.append(delta.content)
                            collected_chars += len(delta.content)
            except (AttributeError, IndexError):
                pass
            yield chunk

        # After stream ends: record cost
        try:
            output_cost = pricing.output_cost(model, token_count)
            input_token_est = sum(
                len(str(m.get("content", ""))) // 4 + 4
                for m in messages if isinstance(m, dict)
            )
            input_cost = pricing.input_cost(model, input_token_est)
            engine.record_cost(input_cost + output_cost)

            full_response = "".join(collected_content)
            if full_response.strip():
                engine.add_partial_result(full_response)
                from agentfuse.core.response_validator import validate_for_cache
                if validate_for_cache(full_response):
                    cache.store(
                        model=model, messages=messages, response=full_response,
                        temperature=temperature, tools=tools,
                    )
        except Exception:
            pass

    return stream_wrapper()


def set_active_run(run_id: str):
    """Set the active run for the current thread/async task."""
    _active_openai_run.set(run_id)


def cleanup_openai(run_id: str = None):
    """Release resources for a completed run."""
    with _wrap_lock:
        if run_id and run_id in _run_contexts:
            del _run_contexts[run_id]
        elif run_id is None:
            _run_contexts.clear()


def _mock_openai_response(content: str, model: str):
    """Creates an OpenAI-compatible response object for cache hits."""
    return MockOpenAIResponse(content, model)
