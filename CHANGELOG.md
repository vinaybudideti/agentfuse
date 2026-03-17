# Changelog

## [0.2.1] — 2026-03-16

### Added
- GPT-5 ($1.25/$10.00) and GPT-5.4 ($2.50/$15.00) model pricing
- GPT-4.1-mini ($0.40/$1.60) and GPT-4.1-nano ($0.10/$0.40) pricing
- Per-model-family cache discount multipliers (90% GPT-5, 75% GPT-4.1, 50% GPT-4o, 90% Gemini)
- `get_spend_report()` — persistent spend analytics (survives process restarts)
- SpendLedger wired into gateway for automatic cost and cache-hit recording
- Prometheus metrics wired into gateway (cache lookup, cost, errors, tokens)
- CacheAttack defense: dual-threshold verification (0.95 write, 0.90 read)
- Per-tenant L2 cache isolation (prevents cross-tenant cache poisoning)
- L2→L1 cache promotion (semantic hits promoted for sub-ms repeat latency)
- `ClassifiedError.counts_for_circuit_breaker` (excludes 429s from breaker)
- GPT-5 downgrade chain: 5.4→5→4.1→4.1-mini→4.1-nano
- GPT-5 fallback chains and routing pairs
- Anthropic `cache_creation` sub-object with TTL breakdowns (5-min vs 1-hour)
- 1-hour cache write TTL at 2.0× pricing (vs 1.25× for 5-min)
- Gemini `tool_use_prompt_token_count` added to input tokens
- 10 new GCRA rate limiter tests

### Fixed
- Handle all new finish/stop reasons: `content_filter`, `max_tokens`, `pause_turn`, `refusal`, `SAFETY`, `RECITATION`
- Skip Anthropic thinking blocks when extracting text for cache
- OpenAI Agents SDK Model.get_response() updated to v0.12.2 signature
- Anthropic prompt cache minimum thresholds: Opus 4.6→4096, Sonnet 4.6→2048, Haiku 4.5→4096
- Singleton RequestOptimizer/IntelligentModelRouter (avoid per-call allocation)

### Stats
- 511 unit tests, all green
- 90% core module coverage
- 48 public API exports
- 30+ models in registry

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
- Anthropic overflow pricing (>200K input: 2x input, 1.5x output)
- FAISS index persistence (`save_l2_index` / `load_l2_index`)
- `cached_input_cost()` for provider cache discounts
- `get_stats()` and `get_budget_summary()` for observability
- `__repr__` on BudgetEngine and NormalizedUsage
- `py.typed` marker for PEP 561 type checking
- 206 behavioral tests, 85% core coverage

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
