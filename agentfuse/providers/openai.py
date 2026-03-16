"""
OpenAI monkey-patch — intercepts openai.chat.completions.create for
budget enforcement and semantic caching.

Usage (2 lines):
    from agentfuse import wrap_openai
    wrap_openai(budget_usd=5.00, run_id="run_123")

Handles both streaming and non-streaming responses.
"""

import threading

from agentfuse.providers.mock_responses import MockOpenAIResponse

# Module-level state — protected by _wrap_lock
_wrap_lock = threading.Lock()
_budget_engines = {}  # run_id -> BudgetEngine
_cache_middleware = None
_original_openai_create = None  # store original to prevent double-wrapping


def wrap_openai(budget_usd: float, run_id: str = None,
                model: str = "gpt-4o", alert_cb=None,
                store=None):
    """
    Monkey-patches openai.chat.completions.create to intercept all calls.
    Call once at the start of an agent run.
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
        _budget_engines[run_id] = engine

        global _original_openai_create
        if _original_openai_create is None:
            _original_openai_create = openai.chat.completions.create
        original_create = _original_openai_create

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
            return _mock_openai_response(cache_result.response, call_model)

        # Step 2: Check budget
        token_count = tokenizer.count_messages_tokens(messages, call_model)
        est_cost = pricing.input_cost(call_model, token_count)
        messages, active_model = engine.check_and_act(est_cost, messages)
        kwargs["messages"] = messages
        kwargs["model"] = active_model

        # Step 3: Make real call
        result = original_create(*args, **kwargs)

        # Step 4 + 5: Handle streaming vs non-streaming
        if is_streaming:
            return _wrap_openai_stream(result, active_model, engine, pricing,
                                       cache, messages, temperature, tools)
        else:
            _record_and_cache_openai(result, active_model, engine, pricing,
                                      cache, messages, temperature, tools)
            return result

    openai.chat.completions.create = intercepted_create
    return run_id


def _record_and_cache_openai(result, model, engine, pricing, cache,
                               messages, temperature, tools):
    """Record cost and cache response for non-streaming OpenAI calls."""
    from agentfuse.providers.response import extract_usage
    try:
        if hasattr(result, "usage") and result.usage:
            normalized = extract_usage("openai", result.usage)
            actual_cost = pricing.total_cost_normalized(model, normalized)
            engine.record_cost(actual_cost)

        if result.choices and result.choices[0].message.content:
            response_text = result.choices[0].message.content
            engine.add_partial_result(response_text)
            cache.store(
                model=model, messages=messages, response=response_text,
                temperature=temperature, tools=tools,
            )
    except (AttributeError, IndexError):
        pass


def _wrap_openai_stream(stream, model, engine, pricing,
                         cache, messages, temperature, tools):
    """Wrap a streaming OpenAI response to track cost and collect content."""
    MAX_COLLECTED_CHARS = 500_000  # 500KB max to prevent memory spike
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

        # After stream ends: record cost using correct model
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
                cache.store(
                    model=model, messages=messages, response=full_response,
                    temperature=temperature, tools=tools,
                )
        except Exception:
            pass

    return stream_wrapper()


def cleanup_openai(run_id: str = None):
    """Release resources for a completed run. Call when agent run is done."""
    if run_id and run_id in _budget_engines:
        del _budget_engines[run_id]
    elif run_id is None:
        _budget_engines.clear()


def _mock_openai_response(content: str, model: str):
    """Creates an OpenAI-compatible response object for cache hits."""
    return MockOpenAIResponse(content, model)
