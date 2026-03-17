"""
AgentFuse Gateway — unified LLM completion with automatic cost optimization.

This is the production-grade entry point inspired by LiteLLM's completion()
and Portkey's gateway architecture. Instead of monkey-patching individual
SDKs, users call a single function that handles:

1. Provider routing (OpenAI, Anthropic, Gemini, DeepSeek, etc.)
2. Cache lookup (L1 Redis + L2 FAISS semantic)
3. Budget enforcement (graduated policies)
4. Cost tracking (normalized across providers)
5. Response validation (prevents caching garbage)
6. Error classification + retry with fallback
7. Observability (OTel spans, metrics, logging)

Usage:
    from agentfuse.gateway import completion

    response = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        budget_id="my_run",
        budget_usd=5.00,
    )

    # Works with ANY provider:
    response = completion(model="claude-sonnet-4-6", messages=[...], budget_id="run_2", budget_usd=3.00)
    response = completion(model="gemini-2.5-pro", messages=[...], budget_id="run_3", budget_usd=1.00)
    response = completion(model="deepseek/deepseek-chat", messages=[...])

This is what LiteLLM does — one function for all providers.
This is what Portkey does — a gateway that intercepts and optimizes.
AgentFuse adds what neither has: per-run budgets + semantic caching + anomaly detection.
"""

import logging
import os
import time
import threading
from typing import Optional, Any
from contextvars import ContextVar

from agentfuse.core.budget import BudgetEngine, BudgetExhaustedGracefully
from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit, CacheMiss
from agentfuse.core.error_classifier import classify_error
from agentfuse.core.response_validator import validate_for_cache
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.providers.tokenizer import TokenCounterAdapter
from agentfuse.providers.router import resolve_provider
from agentfuse.providers.response import extract_usage
from agentfuse.providers.token_pattern import extract_with_pattern
from agentfuse.providers.mock_responses import MockOpenAIResponse, MockAnthropicResponse
from agentfuse.core.request_optimizer import RequestOptimizer
from agentfuse.core.model_router import IntelligentModelRouter
from agentfuse.core.dedup import RequestDeduplicator
from agentfuse.core.fallback_chain import DEFAULT_CHAINS
from agentfuse.core.security import validate_response_safety, strip_invisible_chars

# Observability imports — all optional, never crash if unavailable
try:
    from agentfuse.observability.metrics import (
        record_cache_lookup, record_cost as record_cost_metric,
        record_error as record_error_metric, record_tokens,
    )
    _METRICS = True
except ImportError:
    _METRICS = False

logger = logging.getLogger(__name__)

# Shared instances (thread-safe, singleton per process)
_lock = threading.Lock()
_engines: dict[str, BudgetEngine] = {}
_cache = TwoTierCacheMiddleware()
_pricing = ModelPricingEngine()
_tokenizer = TokenCounterAdapter()
_optimizer = RequestOptimizer(_pricing, _tokenizer)
_router = IntelligentModelRouter()
_dedup = RequestDeduplicator()

# Auto-configure from environment variables (12-factor app pattern)
# AGENTFUSE_RATE_LIMIT_RPS — enable rate limiting (e.g., "10.0")
# AGENTFUSE_REDIS_URL — enable Redis L1 cache (e.g., "redis://localhost:6379")
_env_rate_limit = os.environ.get("AGENTFUSE_RATE_LIMIT_RPS")
if _env_rate_limit:
    try:
        from agentfuse.core.gcra_limiter import GCRARateLimiter
        _rate_limiter = GCRARateLimiter(rate=float(_env_rate_limit))
        logger.info("Rate limiting enabled: %s RPS", _env_rate_limit)
    except Exception:
        pass

# Anomaly detector for cost spike detection
try:
    from agentfuse.core.anomaly import CostAnomalyDetector
    _anomaly_detector = CostAnomalyDetector()
except ImportError:
    _anomaly_detector = None
_alert_manager = None  # lazily initialized via configure()
_rate_limiter = None  # lazily initialized via configure()


def configure(
    alert_callback=None,
    alert_webhook_url: Optional[str] = None,
    alert_thresholds: Optional[list[float]] = None,
    rate_limit_rps: Optional[float] = None,
    rate_limit_burst: int = 5,
):
    """Configure gateway-level settings.

    Args:
        alert_callback: Function called when cost threshold is crossed
        alert_webhook_url: URL to POST cost alerts to (e.g., Slack webhook)
        alert_thresholds: List of budget percentages to alert at (default: [0.50, 0.75, 0.90])
        rate_limit_rps: Max requests per second per tenant (None = no limit)
        rate_limit_burst: Burst tolerance for rate limiting
    """
    global _alert_manager, _rate_limiter
    if alert_callback or alert_webhook_url:
        from agentfuse.core.cost_alert import CostAlertManager
        _alert_manager = CostAlertManager(
            thresholds=alert_thresholds,
            callback=alert_callback,
            webhook_url=alert_webhook_url,
        )
    if rate_limit_rps is not None:
        from agentfuse.core.gcra_limiter import GCRARateLimiter
        _rate_limiter = GCRARateLimiter(rate=rate_limit_rps, burst_tolerance=rate_limit_burst)
_spend_ledger = None  # lazily initialized


def _get_ledger():
    """Lazily initialize the SpendLedger."""
    global _spend_ledger
    if _spend_ledger is None:
        try:
            from agentfuse.storage.spend_ledger import SpendLedger
            _spend_ledger = SpendLedger()
        except Exception:
            pass
    return _spend_ledger


def completion(
    model: str,
    messages: list[dict],
    budget_id: Optional[str] = None,
    budget_usd: Optional[float] = None,
    temperature: float = 0.0,
    tools: Optional[list] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
    auto_route: bool = False,
    **kwargs,
) -> Any:
    """
    Unified LLM completion with automatic cost optimization.

    Works with ANY provider — routes based on model name.
    Applies cache, budget, validation, and cost tracking automatically.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-6", "gemini-2.5-pro")
        messages: Chat messages in OpenAI format
        budget_id: Run/session identifier for budget tracking
        budget_usd: Budget limit in USD (creates BudgetEngine if not exists)
        temperature: Sampling temperature
        tools: Tool/function definitions
        max_tokens: Maximum output tokens
        stream: Enable streaming response
        api_key: Provider API key (uses env var if not set)
        tenant_id: Tenant identifier for cache isolation
        **kwargs: Additional provider-specific parameters

    Returns:
        Provider response object (OpenAI-format for compatible providers,
        native format for Anthropic)
    """
    start_time = time.monotonic()

    # Input validation — fail fast on obviously bad inputs
    if not model or not isinstance(model, str):
        raise ValueError("model must be a non-empty string")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list of dicts")
    if budget_usd is not None and budget_usd <= 0:
        raise ValueError("budget_usd must be > 0")
    if temperature < 0 or temperature > 2:
        raise ValueError("temperature must be between 0 and 2")

    # Rate limiting (per-tenant if configured)
    if _rate_limiter:
        rate_key = tenant_id or budget_id or "global"
        if not _rate_limiter.check(rate_key):
            from agentfuse.core.rate_limiter import RateLimitExceeded
            raise RateLimitExceeded(f"Rate limit exceeded for tenant {rate_key}")

    # Resolve provider
    provider, base_url = resolve_provider(model)

    # Step 0a: Optimize request (remove empty/duplicate messages)
    messages, opt_report = _optimizer.optimize(messages, model)
    if opt_report.messages_removed > 0:
        logger.info("Request optimized: saved %d tokens ($%.6f)",
                     opt_report.estimated_tokens_saved, opt_report.estimated_cost_saving_usd)

    # Step 0b: Intelligent model routing (RouteLLM-inspired)
    # Routes simple queries to cheaper models for 70%+ cost savings
    if auto_route:
        model = _router.route(model, messages)
        provider, base_url = resolve_provider(model)  # re-resolve after routing

    # Get or create budget engine
    engine = _get_engine(budget_id, budget_usd, model) if budget_id else None

    # Step 1: Cache lookup
    cache_result = _cache.lookup(
        model=model, messages=messages,
        temperature=temperature, tools=tools, tenant_id=tenant_id,
    )
    if isinstance(cache_result, CacheHit):
        logger.debug("Cache hit (tier %d) for %s", cache_result.tier, model)
        if _METRICS:
            try:
                record_cache_lookup(model, hit=True, tier=cache_result.tier)
            except Exception:
                pass

        # Record cache hit to ledger (zero cost, cached=True)
        ledger = _get_ledger()
        if ledger:
            try:
                ledger.record(
                    run_id=budget_id or "untracked",
                    model=model,
                    cost_usd=0.0,
                    provider=provider,
                    cached=True,
                )
            except Exception:
                pass

        if provider == "anthropic":
            return MockAnthropicResponse(cache_result.response, model)
        return MockOpenAIResponse(cache_result.response, model)

    # Record cache miss
    if _METRICS:
        try:
            record_cache_lookup(model, hit=False)
        except Exception:
            pass

    # Step 2: Budget check
    active_model = model
    if engine:
        token_count = _tokenizer.count_messages(messages, model)
        est_cost = _pricing.input_cost(model, token_count)
        messages, active_model = engine.check_and_act(est_cost, messages)

    # Step 3: Route to provider and make the call
    # Non-streaming requests are deduplicated — identical in-flight requests
    # share a single API call (saves money on duplicate requests)
    try:
        def _make_call():
            if provider == "anthropic":
                return _call_anthropic(active_model, messages, temperature, tools,
                                          max_tokens, stream, api_key, **kwargs)
            else:
                return _call_openai_compatible(active_model, messages, temperature,
                                                  tools, max_tokens, stream, api_key,
                                                  base_url, **kwargs)

        if stream:
            result = _make_call()  # streaming can't be deduplicated
        else:
            dedup_key = _dedup.make_key(active_model, messages, temperature)
            result = _dedup.execute(dedup_key, _make_call)
    except Exception as exc:
        classified = classify_error(exc, provider)
        logger.warning("LLM call failed: %s (%s)", classified.error_type, provider)
        if _METRICS:
            try:
                record_error_metric(classified.error_type, provider)
            except Exception:
                pass

        # Automatic fallback: if error is retryable and we have fallback models
        if classified.retryable and active_model in DEFAULT_CHAINS:
            for fallback_model in DEFAULT_CHAINS[active_model]:
                try:
                    fb_provider, fb_base_url = resolve_provider(fallback_model)
                    logger.info("Falling back: %s → %s", active_model, fallback_model)
                    if fb_provider == "anthropic":
                        result = _call_anthropic(fallback_model, messages, temperature,
                                                  tools, max_tokens, stream, api_key, **kwargs)
                    else:
                        result = _call_openai_compatible(fallback_model, messages, temperature,
                                                          tools, max_tokens, stream, api_key,
                                                          fb_base_url, **kwargs)

                    # Record cost and cache for fallback
                    if not stream:
                        _record_cost(result, fallback_model, fb_provider, engine)
                        _validate_and_cache(result, fallback_model, fb_provider,
                                             messages, temperature, tools, tenant_id)

                    elapsed = time.monotonic() - start_time
                    logger.debug("completion(%s, fallback from %s) took %.3fs",
                                 fallback_model, active_model, elapsed)
                    return result
                except Exception:
                    continue  # try next fallback

        raise

    # Step 4: Record cost
    if not stream:
        _record_cost(result, active_model, provider, engine)

    # Step 5: Validate and cache
    if not stream:
        _validate_and_cache(result, active_model, provider, messages,
                             temperature, tools, tenant_id)

    elapsed = time.monotonic() - start_time
    logger.debug("completion(%s) took %.3fs", active_model, elapsed)

    return result


def _get_engine(budget_id: str, budget_usd: Optional[float], model: str) -> BudgetEngine:
    """Get or create a BudgetEngine for this budget_id."""
    if budget_id in _engines:
        return _engines[budget_id]

    if budget_usd is None:
        budget_usd = 10.0  # default $10 budget

    with _lock:
        if budget_id not in _engines:
            _engines[budget_id] = BudgetEngine(budget_id, budget_usd, model)
    return _engines[budget_id]


def _call_openai_compatible(model, messages, temperature, tools,
                             max_tokens, stream, api_key, base_url, **kwargs):
    """Call any OpenAI-compatible provider."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required. Install: pip install openai")

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = openai.OpenAI(**client_kwargs)

    call_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        call_kwargs["tools"] = tools
    if max_tokens:
        call_kwargs["max_tokens"] = max_tokens
    if stream:
        call_kwargs["stream"] = True
    call_kwargs.update(kwargs)

    return client.chat.completions.create(**call_kwargs)


def _call_anthropic(model, messages, temperature, tools,
                     max_tokens, stream, api_key, **kwargs):
    """Call Anthropic's native API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install: pip install anthropic")

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key

    client = anthropic.Anthropic(**client_kwargs)

    # Extract system message (Anthropic handles it separately)
    system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
    chat_msgs = [m for m in messages if m.get("role") != "system"]

    call_kwargs = {
        "model": model,
        "messages": chat_msgs,
        "max_tokens": max_tokens or 4096,
    }
    if system_msgs:
        call_kwargs["system"] = "\n".join(system_msgs)
    if temperature > 0:
        call_kwargs["temperature"] = temperature
    if tools:
        call_kwargs["tools"] = tools
    if stream:
        call_kwargs["stream"] = True
    call_kwargs.update(kwargs)

    return client.messages.create(**call_kwargs)


def _record_cost(result, model, provider, engine):
    """Extract usage and record actual cost."""
    try:
        usage_obj = getattr(result, "usage", None)
        if usage_obj:
            try:
                normalized = extract_usage(provider, usage_obj)
            except Exception:
                normalized = extract_with_pattern(usage_obj, provider)

            actual_cost = _pricing.total_cost_normalized(model, normalized)
            if engine:
                engine.record_cost(actual_cost)
                # Check cost alerts
                if _alert_manager:
                    try:
                        _alert_manager.check(engine)
                    except Exception:
                        pass

            # Check for cost anomalies (EMA + z-score spike detection)
            if _anomaly_detector:
                try:
                    _anomaly_detector.record(model, actual_cost)
                except Exception:
                    pass

            # Record to persistent ledger (out of hot path — append-only file)
            ledger = _get_ledger()
            if ledger:
                try:
                    ledger.record(
                        run_id=engine.run_id if engine else "untracked",
                        model=model,
                        cost_usd=actual_cost,
                        provider=provider,
                        input_tokens=normalized.total_input_tokens,
                        output_tokens=normalized.total_output_tokens,
                    )
                except Exception:
                    pass

            # Record metrics
            if _METRICS:
                try:
                    record_cost_metric(model, provider, actual_cost)
                    record_tokens(model,
                                  input_tokens=normalized.total_input_tokens,
                                  output_tokens=normalized.total_output_tokens)
                except Exception:
                    pass
    except Exception as e:
        logger.warning("Cost recording failed: %s", e)


def _validate_and_cache(result, model, provider, messages,
                         temperature, tools, tenant_id):
    """Validate response and store in cache if valid."""
    try:
        # Extract text content
        response_text = ""
        finish_reason = None

        if provider == "anthropic":
            if hasattr(result, "content") and result.content:
                for block in result.content:
                    # Skip thinking blocks — they contain internal reasoning, not cacheable content
                    block_type = getattr(block, "type", None)
                    if block_type == "thinking":
                        continue
                    if hasattr(block, "text") and block.text:
                        response_text = block.text
                        break
            # Map Anthropic stop_reason to finish_reason for validation
            # Anthropic values: end_turn, tool_use, max_tokens, stop_sequence, pause_turn, refusal
            stop_reason = getattr(result, "stop_reason", None)
            finish_reason = "stop" if stop_reason == "end_turn" else stop_reason
        else:
            if hasattr(result, "choices") and result.choices:
                choice = result.choices[0]
                response_text = getattr(choice.message, "content", "") or ""
                finish_reason = getattr(choice, "finish_reason", None)

        if response_text and validate_for_cache(response_text, finish_reason=finish_reason):
            # Security: check response for malicious content before caching
            is_safe, _ = validate_response_safety(response_text)
            if is_safe:
                _cache.store(
                    model=model, messages=messages, response=response_text,
                    temperature=temperature, tools=tools, tenant_id=tenant_id,
                )
    except Exception as e:
        logger.warning("Cache storage failed: %s", e)


def get_engine(budget_id: str) -> Optional[BudgetEngine]:
    """Get an existing BudgetEngine by budget_id."""
    return _engines.get(budget_id)


def get_spend(budget_id: str) -> float:
    """Get total spend for a budget_id."""
    engine = _engines.get(budget_id)
    return engine.spent if engine else 0.0


def cleanup(budget_id: Optional[str] = None):
    """Release resources for a budget_id or all."""
    with _lock:
        if budget_id:
            _engines.pop(budget_id, None)
        else:
            _engines.clear()


def get_spend_report() -> dict:
    """Get a spend report from the persistent ledger.

    Returns dict with total_usd, by_model, by_provider, by_run.
    This data survives process restarts (persisted to JSONL).
    """
    ledger = _get_ledger()
    if ledger is None:
        return {"total_usd": 0.0, "by_model": {}, "by_provider": {}, "by_run": {}}
    return {
        "total_usd": ledger.get_total_spend(),
        "by_model": ledger.get_spend_by_model(),
        "by_provider": ledger.get_spend_by_provider(),
        "by_run": ledger.get_spend_by_run(),
    }
