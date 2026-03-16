# AgentFuse Build Progress

## Current State: v0.2.0 (Production Rebuild Complete)

**175 unit tests, all green | 80% core module coverage**

## Completed Phases

### Phase 0 — Fix 3 Critical Production Bugs
- Cache key design: SHA-256 with model, cross-model contamination impossible
- Token counting: o200k_base for GPT-4o/o-series, safety margins for all providers
- Thread safety: instance-level locks + ContextVar for per-run isolation

### Phase 1 — ModelRegistry + ProviderRouter
- Hot-reloadable registry with 20+ models, LiteLLM remote refresh
- 4-tier lookup: overrides → exact → ft:prefix 2x → warn+zero
- ProviderRouter with 10 OpenAI-compatible providers + Anthropic native

### Phase 2 — Two-Tier Semantic Cache
- Redis L1 exact match + FAISS L2 semantic search
- redis/langcache-embed-v2 embeddings, 0.92 similarity threshold
- Cross-model prevention via key design + model prefix post-filter
- Tool-use queries never go through L2, temperature > 0.5 skips cache

### Phase 3 — Atomic Budget Enforcement
- RedisBudgetStore with Lua scripts (microdollar precision)
- InMemoryBudgetStore with TTLCache + RLock + async variant
- NormalizedUsage: unified extraction (Anthropic input_tokens fix)

### Phase 4 — Framework Integrations
- LangChain: BaseChatModel wrapper (callbacks are observe-only)
- CrewAI: create_agentfuse_hooks with side-channel cache
- OpenAI Agents SDK: AgentFuseModel + AgentFuseModelProvider

### Phase 5 — Unified Error Handling
- classify_error() across OpenAI, Anthropic, Google GenAI, httpx
- OpenAI insufficient_quota 429 is NOT retryable
- Anthropic OverloadedError 529 is always retryable
- agentfuse_retry decorator with tenacity

### Phase 6 — Observability
- OTel GenAI spans (semconv v1.40)
- structlog JSON logging with trace context injection
- Prometheus metrics: cache hits, cost, budget, errors, fallbacks

### Phase 7 — Final Test Suite
- 175 unit tests, all behavioral (no construction-only tests)
- 80% core module coverage

### Phase 8 — PyPI v0.2.0
- pyproject.toml v0.2.0, Python >=3.11
- Full public API in __init__.py
- CHANGELOG.md, updated README

## Improvement Loop Progress
- 10 loop iterations completed
- Fixed: L2 eviction data loss, 90% budget policy, store_compat vector tracking
- Added: Mistral/DeepSeek/Grok/Llama tokenization, o3/gpt-4.1 downgrade paths
- Added: get_stats(), get_budget_summary(), CHANGELOG.md
