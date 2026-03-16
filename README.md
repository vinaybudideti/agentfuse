# AgentFuse

> Intelligent LLM Agent Cost Optimization Runtime

![PyPI](https://img.shields.io/pypi/v/agentfuse-runtime)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Tests](https://img.shields.io/badge/tests-260%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/core%20coverage-86%25-green)

AgentFuse is a production-grade Python SDK that enforces per-run LLM budgets with semantic caching, graduated cost policies, and unified observability across OpenAI, Anthropic, Google Gemini, DeepSeek, Mistral, and 10+ providers.

## The Problem

AI agents burn money without warning. A stuck loop can cost $50 in 10 minutes. Retries on a failed call can triple your bill. There is no built-in way to enforce per-run budgets across LLM providers.

## The Solution

AgentFuse intercepts every LLM call with a two-tier semantic cache (Redis L1 exact-match + FAISS L2 vector similarity) achieving 87.5% hit rate, and enforces graduated budget policies that downgrade models, compress context, and terminate gracefully instead of burning your budget.

## Key Numbers

| Metric | Value |
|---|---|
| Cache hit rate | 87.5% on repeated and paraphrased prompts |
| Cost reduction | 71.8% ($0.24 vs $0.87 for same workload) |
| Tokens saved | 179,445 per 100 calls |
| Integration effort | 2 lines of code |
| Test suite | 260 unit tests, 86% core coverage |
| Models supported | 22+ with hot-reloadable pricing |
| Providers supported | 12 (OpenAI, Anthropic, Gemini, DeepSeek, Mistral, Groq, Together, xAI, Fireworks, OpenRouter, Ollama, vLLM) |

## Quickstart

```bash
pip install agentfuse-runtime
```

```python
from agentfuse import wrap_openai
import openai

wrap_openai(budget_usd=5.00, run_id="my_agent")
# All subsequent openai.chat.completions.create() calls are now
# budget-enforced and semantically cached
```

## Architecture

```
Request → Cache Lookup (L1 Redis → L2 FAISS) → Budget Check → LLM Call → Cost Recording → Cache Store
                                                    │
                                          ┌─────────┼─────────┐
                                          │         │         │
                                       60%: Alert  80%: Downgrade  90%: Compress  100%: Terminate
```

**Graduated Budget Policies:**
- **60% spent** — alert callback triggered
- **80% spent** — automatic model downgrade (e.g., GPT-4o to GPT-4o-mini, Claude Opus to Sonnet)
- **90% spent** — context compression (system prompt + last 6 messages) and model downgrade
- **100% spent** — graceful termination with partial results preserved

## Features

### Budget Enforcement
- Per-run budget limits with graduated degradation policies
- Atomic Redis Lua budget enforcement with microdollar precision (integer arithmetic, no floating-point drift)
- In-memory budget store with TTL and LRU eviction for zero-dependency deployments
- Async-safe budget operations via `asyncio.Lock`
- Anthropic overflow pricing: automatic 2x input / 1.5x output when requests exceed 200K input tokens
- Gemini Pro overflow pricing: automatic 2x when requests exceed 200K input tokens

### Two-Tier Semantic Cache
- **L1 (Redis):** SHA-256 exact-match lookup, sub-millisecond latency. Falls back to local `TTLCache` when Redis is unavailable
- **L2 (FAISS):** `IndexFlatIP` vector similarity search with `redis/langcache-embed-v2` embeddings (768-dim, purpose-built for semantic caching), 2-5ms latency
- Cross-model contamination prevention: model name in L1 hash key, model prefix post-filter on L2 results
- Tool-use queries never enter L2 semantic search (tool arguments are context-dependent)
- Requests with `temperature > 0.5` or side-effect tools (`send_email`, `execute_trade`, etc.) skip cache entirely
- 24-hour TTL with +/-10% jitter to prevent thundering herd
- FAISS index persistence to disk (`save_l2_index` / `load_l2_index`)
- LRU eviction at configurable max entries with full FAISS index rebuild

### Token Counting
- Provider-aware 4-tier fallback chain:
  - **Tier 1:** Exact local tokenizer — GPT-4o/GPT-4.1/o1/o3/o4 use `o200k_base`, GPT-4/GPT-3.5 use `cl100k_base`
  - **Tier 2:** Provider API counting (Anthropic/Gemini — planned)
  - **Tier 3:** `tiktoken cl100k_base` with safety margins — Anthropic 1.20x, Gemini 1.25x, Mistral 1.15x, DeepSeek/Llama 1.10x, Grok 1.15x
  - **Tier 4:** Character-based estimate (`len(text) / 3.5`) for unknown models
- Multimodal content block handling (Anthropic vision format)

### Model Registry
- Hot-reloadable pricing for 22+ models at March 2026 rates (GPT-4.1, o3, o4-mini, Claude Sonnet/Haiku/Opus 4.x, Gemini 2.x, DeepSeek V3/R1, Mistral, Grok, Llama)
- 4-tier pricing lookup: user overrides, exact match, fine-tuned model (2x base price), unknown (zero + warning)
- LiteLLM remote refresh for automatic pricing updates
- `cached_input_cost()` for provider cache discounts (Anthropic 90% off, DeepSeek 90% off)
- Configurable refresh interval via `AGENTFUSE_REGISTRY_REFRESH_HOURS` environment variable

### Provider Router
- Automatic provider detection from model name with base URL routing for OpenAI-compatible providers
- 12 providers: OpenAI, Anthropic (native SDK), Gemini, DeepSeek, Mistral, Groq, Together AI, xAI, Fireworks AI, OpenRouter, Ollama (local), vLLM (self-hosted)
- Fine-tuned model routing: `ft:gpt-4o:org:name` resolves to OpenAI
- `list_providers()` to enumerate all configured providers

### Unified Error Handling
- `classify_error()` across OpenAI, Anthropic, Google GenAI, and httpx exceptions
- Provider-specific handling:
  - OpenAI `insufficient_quota` (429) is **not** retryable (billing issue, not rate limit)
  - Anthropic `OverloadedError` (529) is **always** retryable
  - Context window exceeded (400) is **not** retryable
- `Retry-After` header extraction from provider responses
- `agentfuse_retry()` decorator with tenacity exponential backoff
- Cost-aware retry with automatic model downgrade on each retry attempt

### Reliability
- Semantic loop detection with FAISS sliding window and cost-aware thresholds
- Streaming cost middleware with real-time per-token cost tracking and abort capability
- Anthropic prompt caching middleware — auto-injects `cache_control` markers on static system messages above 1024 tokens
- Structured JSON cost receipts with per-step logging (model, tokens, cost, cache tier, latency)

### Framework Integrations

| Framework | Integration | Cache Support |
|---|---|---|
| OpenAI | `wrap_openai()` monkey-patch | Full (intercept + return cached) |
| Anthropic | `wrap_anthropic()` monkey-patch | Full (intercept + return cached) |
| LangChain | `AgentFuseChatModel` (BaseChatModel wrapper) | Full (wrapper checks cache before delegating) |
| CrewAI | `create_agentfuse_hooks()` (before/after hooks) | Full (before hook blocks call on cache hit) |
| OpenAI Agents SDK | `AgentFuseModel` / `AgentFuseModelProvider` | Full (async cache check in `get_response`) |

### Observability
- **OpenTelemetry:** GenAI semantic convention v1.40 spans with `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model` attributes
- **Structured Logging:** `structlog` JSON output with automatic OTel trace/span ID injection and Datadog `dd.trace_id` / `dd.span_id` correlation
- **Prometheus Metrics:**
  - `gen_ai_client_token_usage` — histogram, tokens per operation
  - `gen_ai_client_operation_duration_seconds` — histogram, latency
  - `agentfuse_cache_hits_total` / `agentfuse_cache_lookups_total` — counters by model and tier
  - `agentfuse_cost_usd_total` — counter by model and provider
  - `agentfuse_cost_per_request_usd` — histogram with buckets [0.0001 ... 10.0]
  - `agentfuse_budget_remaining_usd` — gauge per budget
  - `agentfuse_errors_total` — counter by error type and provider
  - `agentfuse_model_fallbacks_total` — counter by original and fallback model
- All observability calls wrapped in try/except — failures never propagate to user code

### Usage Normalization
- Unified `NormalizedUsage` dataclass across all providers
- Anthropic fix: `input_tokens` excludes cached tokens — AgentFuse adds `cache_read_input_tokens` + `cache_creation_input_tokens` for correct total
- OpenAI: `completion_tokens` already includes `reasoning_tokens` — no double-counting
- Gemini: `thoughts_token_count` billed as output — added to `candidates_token_count`

## Integration Examples

### OpenAI

```python
from agentfuse import wrap_openai
wrap_openai(budget_usd=5.00, run_id="my_agent")
```

### Anthropic

```python
from agentfuse import wrap_anthropic
wrap_anthropic(budget_usd=5.00, run_id="my_agent")
```

### LangChain

```python
from agentfuse.integrations.langchain import AgentFuseChatModel
model = AgentFuseChatModel(inner=ChatOpenAI(), budget=5.00)
response = model.invoke([HumanMessage(content="Hello")])
```

### CrewAI

```python
from agentfuse.integrations.crewai import create_agentfuse_hooks
before_hook, after_hook = create_agentfuse_hooks(budget=5.00)
```

### OpenAI Agents SDK

```python
from agentfuse.integrations.openai_agents import AgentFuseModelProvider
provider = AgentFuseModelProvider(inner=your_provider, budget=5.00)
result = await Runner.run(agent, run_config=RunConfig(model_provider=provider))
```

### Programmatic Usage

```python
from agentfuse import BudgetEngine, ModelPricingEngine, TokenCounterAdapter

engine = BudgetEngine("run_123", budget_usd=5.00, model="gpt-4o")
pricing = ModelPricingEngine()
tokenizer = TokenCounterAdapter()

messages = [{"role": "user", "content": "What is the capital of France?"}]
tokens = tokenizer.count_messages(messages, "gpt-4o")
estimated_cost = pricing.input_cost("gpt-4o", tokens)

result_messages, active_model = engine.check_and_act(estimated_cost, messages)
# active_model may be downgraded if budget is running low
```

## Comparison

| Feature | AgentFuse | AgentBudget | LiteLLM | Portkey |
|---|---|---|---|---|
| Per-run budget enforcement | Yes | Yes | No | No |
| Semantic caching (87.5% hit rate) | Yes | No | No | Basic |
| Two-tier cache (Redis + FAISS) | Yes | No | No | No |
| Cross-model contamination prevention | Yes | No | No | No |
| Mid-run model switching (at 80% budget) | Yes | No | No | No |
| Context compression (at 90% budget) | Yes | No | No | No |
| Graceful termination + partial results | Yes | No | No | No |
| Semantic loop detection | Yes | No | No | No |
| Retry storm prevention with model downgrade | Yes | No | Yes | Yes |
| Streaming cost abort | Yes | No | No | No |
| Auto Anthropic prompt caching | Yes | No | No | No |
| Unified error classifier (3+ providers) | Yes | No | Partial | No |
| Anthropic/Gemini overflow pricing | Yes | No | No | No |
| OTel GenAI spans + Prometheus metrics | Yes | No | Yes | Yes |
| Structured cost receipts (JSON) | Yes | No | Basic | Basic |
| LangChain / CrewAI / OpenAI Agents SDK | Yes | No | Yes | Yes |
| Hot-reloadable model pricing (22+ models) | Yes | No | Yes | No |
| Atomic Redis Lua budget enforcement | Yes | No | No | No |
| Provider-aware token counting (6 providers) | Yes | No | Partial | No |
| FAISS index persistence | Yes | No | No | No |

## Install

```bash
pip install agentfuse-runtime              # Core (in-memory cache + budget)
pip install agentfuse-runtime[redis]       # + Redis cache and budget store
pip install agentfuse-runtime[otel]        # + OpenTelemetry tracing
pip install agentfuse-runtime[openai]      # + OpenAI SDK
pip install agentfuse-runtime[anthropic]   # + Anthropic SDK
pip install agentfuse-runtime[langchain]   # + LangChain Core
pip install agentfuse-runtime[all]         # Everything
```

**Requirements:** Python 3.11+

## Changelog

### v0.2.0 — Production Rebuild (March 2026)

**New Modules:**
- `TwoTierCacheMiddleware` — Redis L1 exact-match + FAISS L2 semantic search with `redis/langcache-embed-v2`
- `ModelRegistry` — hot-reloadable pricing for 22+ models with LiteLLM remote refresh
- `ProviderRouter` — automatic provider detection and base URL routing for 12 providers
- `RedisBudgetStore` — atomic budget enforcement via Redis Lua scripts with microdollar precision
- `InMemoryBudgetStore` / `AsyncInMemoryBudgetStore` — thread-safe budget stores with TTL and LRU
- `NormalizedUsage` / `extract_usage()` — unified token usage extraction across all providers
- `classify_error()` / `ClassifiedError` — unified error classification with Retry-After extraction
- `agentfuse_retry()` — tenacity-based retry decorator with provider-aware predicate
- OTel GenAI spans, structlog JSON logging, Prometheus metrics (11 metric types)

**Framework Integrations (rebuilt):**
- LangChain: `AgentFuseChatModel` — `BaseChatModel` wrapper (callbacks are observe-only and cannot return cached responses)
- CrewAI: `create_agentfuse_hooks()` — before/after hooks with side-channel cached response injection
- OpenAI Agents SDK: `AgentFuseModel` / `AgentFuseModelProvider` — async Model interface with non-blocking cache store

**Bug Fixes:**
- Cache key design: SHA-256 with model always first component — cross-model contamination eliminated
- Token counting: GPT-4o/GPT-4.1/o-series use `o200k_base` (was incorrectly using `cl100k_base`)
- Safety margins: Anthropic 1.20x (was 1.15x), Gemini 1.25x (was 1.05x) — prevents budget underrun
- Thread safety: instance-level locks + `ContextVar` for per-run isolation (was using class-level locks causing serialization)
- L2 cache eviction: vectors stored alongside metadata for correct FAISS index rebuild (was losing all vectors)
- 90% budget policy: now correctly applies both model downgrade and context compression
- Retry module: uses `classify_error()` instead of string matching on exception names
- L2 metadata filter: correct prefix detection for Mistral, Grok, Llama models
- Double-wrapping prevention in `wrap_openai()` / `wrap_anthropic()`

**Pricing:**
- Anthropic overflow pricing (>200K input tokens: 2x input, 1.5x output)
- Gemini Pro overflow pricing (>200K input tokens: 2x)
- Cached input cost method for provider cache discounts (Anthropic 90%, DeepSeek 90%)

**Quality:**
- 260 unit tests, 0 construction-only tests, 86% core module coverage
- PEP 561 `py.typed` marker for type checking support

### v0.1.0 — Initial Prototype (February 2026)

- BudgetEngine with graduated policies, CacheMiddleware with Intent Atoms FAISS 3-tier cache
- `wrap_openai()`, `wrap_anthropic()` monkey-patches
- LoopDetectionMiddleware, CostAwareRetry, StreamingCostMiddleware, PromptCachingMiddleware, CostReceiptEmitter
- LangChain, CrewAI, OpenAI Agents SDK integrations (callback-based)
- 34 tests passing

## Roadmap

- [x] Python SDK v0.2.0 — production-ready
- [ ] TypeScript SDK
- [ ] Cloud dashboard with real-time cost monitoring
- [ ] Anomaly detection (per-agent-type baseline profiling)
- [ ] Batch pricing support (50% discount with 24h SLA)
- [ ] Multi-agent budget sharing

## Contributing

```bash
git clone https://github.com/vinaybudideti/agentfuse.git
cd agentfuse
pip install -e ".[all]"
pytest tests/unit/ -v
```

## License

MIT

---

Built by [@vinaybudideti](https://github.com/vinaybudideti)
