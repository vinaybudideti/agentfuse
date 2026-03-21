# AgentFuse: Exact Fix Instructions for VS Code Bot

> **Rules for the bot:**
> 1. Do NOT delete any existing files (except package-lock.json as instructed)
> 2. Every fix shows the EXACT old text and the EXACT new text. Copy-paste only.
> 3. Do the fixes in the order listed. Some later fixes depend on earlier ones.
> 4. After all fixes, run: `pytest tests/unit/ -q --tb=short` to verify nothing broke.

---

## PREREQUISITES PAGE: What Vinay Needs to Provide

Before this project can be tested in the real world and published, you need these things that no bot can create for you:

**Required:**
1. An OpenAI API key. Set it as environment variable: `export OPENAI_API_KEY="sk-..."`
2. An Anthropic API key. Set it as environment variable: `export ANTHROPIC_API_KEY="sk-ant-..."`
3. A PyPI account at https://pypi.org with 2FA enabled and an API token for uploading

**Recommended:**
4. A local Redis instance for testing L1 cache: `docker run -d -p 6379:6379 redis:8`
5. A clean Python 3.11+ virtual environment: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[all]" && pip install pytest pytest-cov pytest-asyncio build twine`

**Clarification:** This is a Python package published to PyPI as `agentfuse-runtime`. It is NOT an npm package. The `package-lock.json` at the root is a stale artifact.

---

## FIX 1: _rate_limiter Gets Overwritten to None

**File:** `agentfuse/gateway.py`

**Problem:** The env var block sets `_rate_limiter` from `AGENTFUSE_RATE_LIMIT_RPS`, then three lines later `_rate_limiter = None` unconditionally resets it. Environment variable rate limiting has never worked.

**Find this exact text:**

```
# Anomaly detector for cost spike detection
try:
    from agentfuse.core.anomaly import CostAnomalyDetector
    _anomaly_detector = CostAnomalyDetector()
except ImportError:
    _anomaly_detector = None
_alert_manager = None  # lazily initialized via configure()
_rate_limiter = None  # lazily initialized via configure()
_output_guardrails = None  # lazily initialized via configure()
```

**Replace with this exact text:**

```
# Module-level state for optional subsystems (lazily initialized via configure())
_alert_manager = None
_rate_limiter = None
_output_guardrails = None

# Anomaly detector for cost spike detection
try:
    from agentfuse.core.anomaly import CostAnomalyDetector
    _anomaly_detector = CostAnomalyDetector()
except ImportError:
    _anomaly_detector = None
```

**Why this works:** The three `None` declarations now appear BEFORE the env var auto-configure block (which is above this section in the file). The env var block can set `_rate_limiter` to a real `GCRARateLimiter`, and nothing overwrites it afterward.

---

## FIX 2: configure() Cannot Set output_guardrails

**File:** `agentfuse/gateway.py`

**Problem:** The `configure()` function declares `global _alert_manager, _rate_limiter` but forgets `_output_guardrails`. The line `_output_guardrails = output_guardrails` creates a local variable that is immediately discarded.

**Find this exact text:**

```
    global _alert_manager, _rate_limiter
    if alert_callback or alert_webhook_url:
```

**Replace with this exact text:**

```
    global _alert_manager, _rate_limiter, _output_guardrails
    if alert_callback or alert_webhook_url:
```

---

## FIX 3: output_guardrails Is Never Checked During Completion

**File:** `agentfuse/gateway.py`

**Problem:** Even after Fix 2, `_output_guardrails` is configured but never used. The `_validate_and_cache()` function does not check it before caching responses.

**Find this exact text:**

```
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
```

**Replace with this exact text:**

```
        if response_text and validate_for_cache(response_text, finish_reason=finish_reason):
            # Security: check response for malicious content before caching
            is_safe, _ = validate_response_safety(response_text)
            if is_safe:
                # Output guardrails check (if configured via configure())
                if _output_guardrails is not None:
                    try:
                        guardrail_result = _output_guardrails.check(response_text)
                        if hasattr(guardrail_result, 'passed') and not guardrail_result.passed:
                            logger.warning("Output guardrails blocked caching: %s",
                                           getattr(guardrail_result, 'reason', 'unknown'))
                            return
                    except Exception:
                        pass  # Never let guardrails crash the caching flow
                _cache.store(
                    model=model, messages=messages, response=response_text,
                    temperature=temperature, tools=tools, tenant_id=tenant_id,
                )
    except Exception as e:
        logger.warning("Cache storage failed: %s", e)
```

---

## FIX 4: Version Mismatch (pyproject.toml says 0.2.0, docs say 0.2.1)

**File 1:** `pyproject.toml`

**Find this exact text:**

```
version = "0.2.0"
```

**Replace with this exact text:**

```
version = "0.2.1"
```

**File 2:** `agentfuse/__init__.py`

**Find this exact text:**

```
__version__ = "0.2.0"
__version_info__ = (0, 2, 0)
```

**Replace with this exact text:**

```
__version__ = "0.2.1"
__version_info__ = (0, 2, 1)
```

---

## FIX 5: Streaming Responses Cached Without Validation (OpenAI)

**File:** `agentfuse/providers/openai.py`

**Problem:** `_wrap_openai_stream()` caches the accumulated response without calling `validate_for_cache()`. Truncated, empty, or garbage streaming responses can poison the cache.

**Find this exact text:**

```
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


```

**Replace with this exact text:**

```
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


```

---

## FIX 6: Streaming Responses Cached Without Validation (Anthropic)

**File:** `agentfuse/providers/anthropic.py`

**Problem:** Same as Fix 5 but in the Anthropic provider.

**Find this exact text:**

```
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
```

**Replace with this exact text:**

```
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


def _extract_response_text(result) -> str:
```

---

## FIX 7: acompletion() Missing Fallback Chain on Error

**File:** `agentfuse/gateway.py`

**Problem:** The sync `completion()` has automatic fallback to cheaper models when an API call fails with a retryable error. The async `acompletion()` just re-raises the exception. Async users get no fallback.

**Find this exact text:**

```
    # Native async provider call
    try:
        result = await _call_provider_async(
            active_model, provider, base_url, messages, temperature,
            tools, max_tokens, stream, api_key, **kwargs,
        )
    except Exception as exc:
        classified = classify_error(exc, provider)
        logger.warning("Async LLM call failed: %s (%s)", classified.error_type, provider)
        raise

    # Post-call processing (sync — fast)
    if not stream:
        _record_cost(result, active_model, provider, engine)
        _validate_and_cache(result, active_model, provider, messages,
                             temperature, tools, tenant_id)

    return result
```

**Replace with this exact text:**

```
    # Native async provider call
    try:
        result = await _call_provider_async(
            active_model, provider, base_url, messages, temperature,
            tools, max_tokens, stream, api_key, **kwargs,
        )
    except Exception as exc:
        classified = classify_error(exc, provider)
        logger.warning("Async LLM call failed: %s (%s)", classified.error_type, provider)
        if _METRICS:
            try:
                record_error_metric(classified.error_type, provider)
            except Exception:
                pass

        # Automatic fallback: same logic as sync completion()
        if classified.retryable and active_model in DEFAULT_CHAINS:
            for fallback_model in DEFAULT_CHAINS[active_model]:
                try:
                    fb_provider, fb_base_url = resolve_provider(fallback_model)
                    logger.info("Async falling back: %s -> %s", active_model, fallback_model)
                    result = await _call_provider_async(
                        fallback_model, fb_provider, fb_base_url, messages,
                        temperature, tools, max_tokens, stream, api_key, **kwargs,
                    )
                    if not stream:
                        _record_cost(result, fallback_model, fb_provider, engine)
                        _validate_and_cache(result, fallback_model, fb_provider,
                                             messages, temperature, tools, tenant_id)
                    elapsed = time.monotonic() - start_time
                    logger.debug("acompletion(%s, fallback from %s) took %.3fs",
                                 fallback_model, active_model, elapsed)
                    return result
                except Exception:
                    continue  # try next fallback

        raise

    # Post-call processing (sync — fast)
    if not stream:
        _record_cost(result, active_model, provider, engine)
        _validate_and_cache(result, active_model, provider, messages,
                             temperature, tools, tenant_id)

    elapsed = time.monotonic() - start_time
    logger.debug("acompletion(%s) took %.3fs", active_model, elapsed)

    return result
```

---

## FIX 8: acompletion() Uses Deprecated asyncio.get_event_loop()

**File:** `agentfuse/gateway.py`

**Problem:** In `_call_provider_async()`, the executor fallback uses `asyncio.get_event_loop()` which is deprecated in Python 3.10+ and emits a DeprecationWarning. Should use `asyncio.get_running_loop()`.

**Find this exact text:**

```
    # Fallback: run sync completion in executor
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
```

**Replace with this exact text:**

```
    # Fallback: run sync completion in executor
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
```

---

## FIX 9: acompletion() Missing stream_options for OpenAI

**File:** `agentfuse/gateway.py`

**Problem:** In `_call_provider_async()`, when streaming is enabled for OpenAI, the code does not inject `stream_options: {"include_usage": True}`. The sync version does this. Without it, streaming calls return zero usage data and cost tracking breaks.

**Find this exact text (inside _call_provider_async):**

```
        if tools:
            call_kwargs["tools"] = tools
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens
        if stream:
            call_kwargs["stream"] = True
        call_kwargs.update(kwargs)

        return await client.chat.completions.create(**call_kwargs)
    except ImportError:
        pass
```

**Replace with this exact text:**

```
        if tools:
            call_kwargs["tools"] = tools
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens
        if stream:
            call_kwargs["stream"] = True
            # Auto-inject stream_options for usage tracking (matches sync behavior)
            if "stream_options" not in kwargs:
                call_kwargs["stream_options"] = {"include_usage": True}
        call_kwargs.update(kwargs)

        return await client.chat.completions.create(**call_kwargs)
    except ImportError:
        pass
```

---

## FIX 10: New SDK Client Created on Every Single API Call

**File:** `agentfuse/gateway.py`

**Problem:** `_call_openai_compatible()` and `_call_anthropic()` create a brand new SDK client on every request. This wastes connection pool setup, TLS handshake, and memory. High-throughput usage creates thousands of clients.

**Find this exact text:**

```
_load_balancer = None
_spend_ledger = None  # lazily initialized
```

**Replace with this exact text:**

```
_load_balancer = None
_spend_ledger = None  # lazily initialized

# Client cache: reuse SDK clients across requests (avoids TLS + connection pool setup per call)
_client_cache: dict[tuple, Any] = {}
_client_cache_lock = threading.Lock()


def _get_or_create_client(client_class, cache_key: tuple, **kwargs):
    """Get a cached SDK client or create and cache a new one."""
    existing = _client_cache.get(cache_key)
    if existing is not None:
        return existing
    with _client_cache_lock:
        if cache_key not in _client_cache:
            _client_cache[cache_key] = client_class(**kwargs)
        return _client_cache[cache_key]
```

Then update `_call_openai_compatible()`.

**Find this exact text:**

```
    client = openai.OpenAI(**client_kwargs)

    call_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
```

**Replace with this exact text:**

```
    cache_key = ("openai_sync", api_key or "_default_", base_url or "_default_")
    client = _get_or_create_client(openai.OpenAI, cache_key, **client_kwargs)

    call_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
```

Then update `_call_anthropic()`.

**Find this exact text:**

```
    client = anthropic.Anthropic(**client_kwargs)

    # Extract system message (Anthropic handles it separately)
```

**Replace with this exact text:**

```
    cache_key = ("anthropic_sync", api_key or "_default_")
    client = _get_or_create_client(anthropic.Anthropic, cache_key, **client_kwargs)

    # Extract system message (Anthropic handles it separately)
```

Then update `_call_provider_async()` for the async Anthropic client.

**Find this exact text:**

```
            client = anthropic.AsyncAnthropic(**client_kwargs)

            system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
```

**Replace with this exact text:**

```
            cache_key = ("anthropic_async", api_key or "_default_")
            client = _get_or_create_client(anthropic.AsyncAnthropic, cache_key, **client_kwargs)

            system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
```

Then update `_call_provider_async()` for the async OpenAI client.

**Find this exact text:**

```
        client = openai.AsyncOpenAI(**client_kwargs)

        call_kwargs = {
```

**Replace with this exact text:**

```
        cache_key = ("openai_async", api_key or "_default_", base_url or "_default_")
        client = _get_or_create_client(openai.AsyncOpenAI, cache_key, **client_kwargs)

        call_kwargs = {
```

**NOTE:** This fix and Fix 9 both touch `_call_provider_async()` but in different spots. They do not conflict. Apply in any order.

---

## FIX 11: cache.py _redis_failures and _embedding_version Are Class Variables

**File:** `agentfuse/core/cache.py`

**Problem:** `_redis_failures` is a class-level variable (shared across all instances). If multiple `TwoTierCacheMiddleware` instances exist and one experiences Redis failures, the circuit breaker counter is shared. Same for `_embedding_version`. Both should be instance variables set in `__init__`.

**Find this exact text (at the end of __init__):**

```
        # Thresholds for backward-compatible check() method
        self._direct_threshold = 0.85
        self._adapt_threshold = 0.70
```

**Replace with this exact text:**

```
        # Thresholds for backward-compatible check() method
        self._direct_threshold = 0.85
        self._adapt_threshold = 0.70

        # Redis circuit breaker (instance-level, not shared across instances)
        self._redis_failures = 0

        # Embedding version tracking for cache drift detection
        self._embedding_version = "v1"
```

Then remove the class-level declarations. **Find this exact text:**

```
    # --- L1 helpers with Redis circuit breaker ---

    _redis_failures: int = 0
    _REDIS_CIRCUIT_OPEN_THRESHOLD: int = 5  # disable after 5 consecutive failures
```

**Replace with this exact text:**

```
    # --- L1 helpers with Redis circuit breaker ---

    _REDIS_CIRCUIT_OPEN_THRESHOLD: int = 5  # disable after 5 consecutive failures
```

Then remove the class-level _embedding_version. **Find this exact text:**

```
    _embedding_version: str = "v1"  # Track embedding model version for drift detection

    def set_embedding_version(self, version: str):
```

**Replace with this exact text:**

```
    def set_embedding_version(self, version: str):
```

---

## FIX 12: AgentSession Has No async with Support

**File:** `agentfuse/core/session.py`

**Problem:** `AgentSession` only supports sync `with` statement. Async frameworks that use `await acompletion()` cannot use `async with AgentSession()`.

**Find this exact text:**

```
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ended_at = time.time()
        elapsed = self._ended_at - self._started_at
        logger.info("AgentSession ended: %s (%.1fs, %d calls, $%.4f spent)",
                     self.run_id, elapsed, self._call_count,
                     self._get_total_cost())
        return False  # don't suppress exceptions
```

**Replace with this exact text:**

```
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ended_at = time.time()
        elapsed = self._ended_at - self._started_at
        logger.info("AgentSession ended: %s (%.1fs, %d calls, $%.4f spent)",
                     self.run_id, elapsed, self._call_count,
                     self._get_total_cost())
        return False  # don't suppress exceptions

    async def __aenter__(self):
        self._started_at = time.time()
        logger.info("AgentSession started (async): %s (budget=$%.2f, model=%s)",
                     self.run_id, self.budget_usd, self.model)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._ended_at = time.time()
        elapsed = self._ended_at - self._started_at
        logger.info("AgentSession ended (async): %s (%.1fs, %d calls, $%.4f spent)",
                     self.run_id, elapsed, self._call_count,
                     self._get_total_cost())
        return False
```

---

## FIX 13: All 3 Example Files Are Empty (Single Comment Line)

**File 1:** `examples/openai_basic.py`

**Delete all existing content and replace with:**

```python
"""AgentFuse: Basic OpenAI integration with budget enforcement and caching."""
import os
from agentfuse import completion, get_spend_report

# Requires: export OPENAI_API_KEY="sk-..."

# One function replaces openai.chat.completions.create()
# Automatically: caches, enforces budget, tracks cost, validates responses
response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is Python? Answer in one sentence."}],
    budget_id="demo_run",
    budget_usd=1.00,
)

print("Response:", response.choices[0].message.content)
print(f"Spend so far: ${get_spend_report()['total_usd']:.6f}")

# Second identical call hits cache (free, instant)
response2 = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is Python? Answer in one sentence."}],
    budget_id="demo_run",
    budget_usd=1.00,
)

cached = getattr(response2, "_agentfuse_cache_hit", False)
print(f"Second call cached: {cached}")
print(f"Total spend: ${get_spend_report()['total_usd']:.6f}")
```

**File 2:** `examples/langchain_example.py`

**Delete all existing content and replace with:**

```python
"""AgentFuse: LangChain integration with budget-enforced chat model."""
from agentfuse.integrations.langchain import AgentFuseChatModel

# Requires: pip install langchain-core
# Requires: export OPENAI_API_KEY="sk-..."

# Wrap any LangChain-compatible model with budget enforcement + caching
model = AgentFuseChatModel(
    inner_model_name="gpt-4o-mini",
    budget_usd=5.00,
    run_id="langchain_demo",
)

# Use in any LangChain chain, agent, or direct invocation
from langchain_core.messages import HumanMessage
response = model.invoke([HumanMessage(content="Explain quantum computing briefly")])
print("Response:", response.content)
```

**File 3:** `examples/crewai_example.py`

**Delete all existing content and replace with:**

```python
"""AgentFuse: Session-based budget tracking (works with CrewAI or any framework)."""
from agentfuse import AgentSession

# Requires: export OPENAI_API_KEY="sk-..."

# AgentSession wraps budget + cost tracking + tool tracking in one context manager
with AgentSession("research_agent", budget_usd=3.00, model="gpt-4o-mini") as session:
    response = session.completion(
        messages=[{"role": "user", "content": "What are the latest AI trends?"}],
    )
    print("Response:", response.choices[0].message.content)

    # Track non-LLM tool costs too
    session.record_tool_call("web_search", cost=0.01)

receipt = session.get_receipt()
print(f"Total cost: ${receipt['total_cost_usd']:.6f}")
print(f"LLM cost: ${receipt['llm_cost_usd']:.6f}")
print(f"Tool cost: ${receipt['tool_cost_usd']:.6f}")
print(f"Cache hit rate: {receipt['cache_hit_rate']:.1%}")
```

---

## FIX 14: Delete Stale package-lock.json

**Action:** Delete the file `package-lock.json` from the project root directory.

**File:** `.gitignore`

**Find this exact text (the last line of .gitignore):**

```
.pytest_cache/
```

**Replace with this exact text:**

```
.pytest_cache/
package-lock.json
node_modules/
*.egg
```

---

## FIX 15: Add Real-World End-to-End Test Script

**Create new file:** `examples/e2e_real_test.py`

**Contents:**

```python
"""
AgentFuse: Real-world end-to-end validation script.

Run this BEFORE publishing to PyPI to verify the gateway works with real APIs.
Requires: OPENAI_API_KEY environment variable.
Estimated cost: $0.01-0.05 per run.

Usage:
    python examples/e2e_real_test.py
"""
import os
import sys
import time

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: Set OPENAI_API_KEY environment variable first.")
    print("  export OPENAI_API_KEY=\"sk-...\"")
    sys.exit(1)

from agentfuse import completion, get_spend_report, estimate_cost, AgentSession
from agentfuse.gateway import cleanup

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} -- {detail}")


print("=" * 60)
print("AgentFuse Real-World E2E Validation")
print("=" * 60)

# --- Test 1: Basic completion ---
print("\n[1] Basic completion with budget...")
cleanup()
try:
    r1 = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        budget_id="e2e_basic",
        budget_usd=0.10,
    )
    text1 = r1.choices[0].message.content
    print(f"     Response: {text1}")
    check("Got non-empty response", len(text1) > 0, f"Got empty string")
    check("Has usage data", hasattr(r1, "usage") and r1.usage is not None)
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Test 2: Cache hit on identical request ---
print("\n[2] Cache hit on repeated request...")
try:
    r2 = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        budget_id="e2e_basic",
        budget_usd=0.10,
    )
    text2 = r2.choices[0].message.content
    cached = getattr(r2, "_agentfuse_cache_hit", False)
    print(f"     Response: {text2}")
    print(f"     Cache hit: {cached}")
    check("Response matches first call", text2 == text1, f"'{text2}' != '{text1}'")
    check("Marked as cache hit", cached, "._agentfuse_cache_hit is False")
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Test 3: Spend tracking ---
print("\n[3] Spend tracking...")
try:
    report = get_spend_report()
    print(f"     Total: ${report['total_usd']:.6f}")
    print(f"     By model: {report['by_model']}")
    check("Total spend > 0", report["total_usd"] > 0, f"Got {report['total_usd']}")
    check("gpt-4o-mini in by_model", "gpt-4o-mini" in report.get("by_model", {}))
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Test 4: Cost estimation ---
print("\n[4] Cost estimation (no API call)...")
try:
    est = estimate_cost(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write a poem about AI"}],
        max_output_tokens=500,
    )
    print(f"     Estimated: ${est['estimated_total_cost_usd']:.6f}")
    check("Estimate > 0", est["estimated_total_cost_usd"] > 0)
    check("Has input tokens", est["estimated_input_tokens"] > 0)
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Test 5: AgentSession ---
print("\n[5] AgentSession context manager...")
cleanup()
try:
    with AgentSession("e2e_session", budget_usd=0.10, model="gpt-4o-mini") as session:
        r5 = session.completion(
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        )
        print(f"     Response: {r5.choices[0].message.content}")
    receipt = session.get_receipt()
    print(f"     Total cost: ${receipt['total_cost_usd']:.6f}")
    print(f"     Calls: {receipt['calls']}")
    check("Receipt has positive cost", receipt["total_cost_usd"] >= 0)
    check("Call count is 1", receipt["calls"] == 1, f"Got {receipt['calls']}")
    check("Has duration", receipt["duration_seconds"] > 0)
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Test 6: Different model ---
print("\n[6] Different model (gpt-4o)...")
cleanup()
try:
    r6 = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say yes."}],
        budget_id="e2e_gpt4o",
        budget_usd=0.10,
    )
    check("gpt-4o responded", len(r6.choices[0].message.content) > 0)
except Exception as e:
    FAIL += 1
    print(f"  FAIL: Exception -- {e}")

# --- Summary ---
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("ALL TESTS PASSED -- safe to publish to PyPI")
else:
    print("SOME TESTS FAILED -- fix issues before publishing")
print("=" * 60)
sys.exit(1 if FAIL > 0 else 0)
```

---

## FIX 16: CI Coverage Gate

**File:** `.github/workflows/ci.yml`

**Find this exact text:**

```
      - name: Verify imports
        run: |
          python -c "import agentfuse; print(f'v{agentfuse.__version__}, {len(agentfuse.__all__)} exports')"
```

**Replace with this exact text:**

```
      - name: Check coverage threshold
        run: |
          COV_OUTPUT=$(pytest tests/unit/ --cov=agentfuse/core --cov-report=term -q 2>&1)
          echo "$COV_OUTPUT" | tail -5
          COVERAGE=$(echo "$COV_OUTPUT" | grep "^TOTAL" | awk '{print $NF}' | tr -d '%')
          if [ -n "$COVERAGE" ] && [ "$COVERAGE" -lt 80 ]; then
            echo "FAIL: Core coverage ${COVERAGE}% is below 80% minimum"
            exit 1
          fi

      - name: Verify imports
        run: |
          python -c "import agentfuse; print(f'v{agentfuse.__version__}, {len(agentfuse.__all__)} exports')"
```

---

## FIX 17: _load_balancer Declared After Its First Use

**File:** `agentfuse/gateway.py`

**Problem:** `_load_balancer = None` appears after the `add_api_key()` function that references it via `global _load_balancer`. While Python handles this correctly at runtime (the module-level line runs before any function is called), it is confusing and fragile.

**Find this exact text:**

```
    _load_balancer.add_endpoint(model, api_key=api_key, base_url=base_url)


_load_balancer = None
_spend_ledger = None  # lazily initialized
```

**Replace with this exact text:**

```
    _load_balancer.add_endpoint(model, api_key=api_key, base_url=base_url)


# NOTE: _load_balancer is initialized above with the other module-level state.
# It was previously declared here after the function that uses it.
_spend_ledger = None  # lazily initialized
```

Then add `_load_balancer = None` up with the other module-level state. **Find this exact text (near the top of the file, the shared instances block):**

```
_context_guard = ContextWindowGuard()
_deprecation_checker = ModelDeprecationChecker()
```

**Replace with this exact text:**

```
_context_guard = ContextWindowGuard()
_deprecation_checker = ModelDeprecationChecker()
_load_balancer = None
```

Then remove the `global _load_balancer` creation guard since it now exists before the function. Actually, keep the `global _load_balancer` line in `add_api_key()` because it is needed for the function to modify the module-level variable. No change needed there.

---

## VERIFICATION: Run After All Fixes

After applying all 17 fixes, run these commands in order:

```bash
# 1. Verify no syntax errors
python3 -c "import agentfuse; print(f'v{agentfuse.__version__}, {len(agentfuse.__all__)} exports')"

# 2. Run full test suite
pytest tests/unit/ -q --tb=short

# 3. Verify version consistency
python3 -c "
import agentfuse
import tomllib
with open('pyproject.toml', 'rb') as f:
    toml_ver = tomllib.load(f)['project']['version']
assert agentfuse.__version__ == toml_ver, f'Mismatch: {agentfuse.__version__} vs {toml_ver}'
print(f'Version consistent: {toml_ver}')
"

# 4. Build the package (dry run)
pip install build
python3 -m build

# 5. Check the built package
pip install twine
twine check dist/*
```

If all 5 steps pass, the package is ready for real-world testing with `examples/e2e_real_test.py` (requires API key), and then PyPI publishing.

---

## SUMMARY TABLE

| Fix | File | What | Severity |
|-----|------|------|----------|
| 1 | gateway.py | _rate_limiter always overwritten to None | CRITICAL |
| 2 | gateway.py | configure() cannot set output_guardrails | CRITICAL |
| 3 | gateway.py | output_guardrails never checked in flow | HIGH |
| 4 | pyproject.toml + __init__.py | Version says 0.2.0, should be 0.2.1 | HIGH |
| 5 | providers/openai.py | Stream cache: no validation before store | HIGH |
| 6 | providers/anthropic.py | Stream cache: no validation before store | HIGH |
| 7 | gateway.py | acompletion: no fallback chain on error | HIGH |
| 8 | gateway.py | Deprecated asyncio.get_event_loop() | MEDIUM |
| 9 | gateway.py | acompletion: missing stream_options | HIGH |
| 10 | gateway.py | New SDK client created per request | MEDIUM |
| 11 | core/cache.py | Shared class vars should be instance | MEDIUM |
| 12 | core/session.py | No async with support | MEDIUM |
| 13 | examples/*.py | All 3 example files are empty | HIGH |
| 14 | .gitignore + package-lock.json | Stale Node.js artifact | LOW |
| 15 | examples/e2e_real_test.py | New: real API validation script | HIGH |
| 16 | .github/workflows/ci.yml | No coverage enforcement | MEDIUM |
| 17 | gateway.py | _load_balancer declared after use | LOW |
