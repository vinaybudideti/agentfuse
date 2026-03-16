# Changelog

## [0.2.0] — 2026-03-16

### Added
- **Two-tier cache**: Redis L1 exact match + FAISS L2 semantic search with `redis/langcache-embed-v2`
- **ModelRegistry**: Hot-reloadable pricing for 20+ models with LiteLLM remote refresh
- **ProviderRouter**: Automatic provider detection and base_url routing for 10+ providers
- **RedisBudgetStore**: Atomic budget enforcement via Redis Lua scripts (microdollar precision)
- **InMemoryBudgetStore**: Thread-safe budget store with TTL and LRU eviction
- **AsyncInMemoryBudgetStore**: Async-safe variant using `asyncio.Lock`
- **NormalizedUsage**: Unified token usage extraction (fixes Anthropic `input_tokens` gotcha)
- **Unified error classifier**: `classify_error()` across OpenAI, Anthropic, Google GenAI
- **OTel GenAI spans**: Semantic convention v1.40 with trace context injection
- **structlog JSON logging**: With automatic OTel + Datadog trace ID injection
- **Prometheus metrics**: Cache hits, cost, budget remaining, errors, model fallbacks
- **LangChain BaseChatModel wrapper**: Proper cache interception (callbacks are observe-only)
- **CrewAI hooks**: `create_agentfuse_hooks()` with side-channel cached responses
- **OpenAI Agents SDK**: `AgentFuseModel` and `AgentFuseModelProvider` classes
- Token counting for Mistral, DeepSeek, Grok, Llama (tiktoken + safety margins)
- Model downgrade paths for o3, gpt-4.1, o1

### Fixed
- Cache key design: SHA-256 with model always first — cross-model contamination impossible
- Token counting: o200k_base for GPT-4o/o-series, 1.20x margin for Anthropic, 1.25x for Gemini
- Thread safety: instance-level locks + ContextVar for per-run isolation
- L2 cache eviction: vectors now stored for FAISS index rebuild
- CostAwareRetry: uses `classify_error` instead of string matching

### Changed
- `ModelPricingEngine` now backed by `ModelRegistry` (returns zero for unknown models, not ValueError)
- Python requirement bumped to >=3.11
- Default embedding model changed from `all-mpnet-base-v2` to `redis/langcache-embed-v2`

## [0.1.0] — 2026-02-28

Initial prototype release.
- BudgetEngine with graduated policies (alert/downgrade/compress/terminate)
- CacheMiddleware with Intent Atoms FAISS 3-tier cache
- wrap_openai() and wrap_anthropic() monkey-patches
- LoopDetectionMiddleware, CostAwareRetry, StreamingCostMiddleware
- PromptCachingMiddleware, CostReceiptEmitter
- LangChain, CrewAI, OpenAI Agents SDK integrations (callback-based)
- 34 tests passing
