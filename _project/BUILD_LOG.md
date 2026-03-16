# AgentFuse Build Log

## Phase 0 — Fix 3 Critical Production Bugs
**Status:** Complete | **Date:** 2026-03-15

Fixed: cache key design (SHA-256 with model), token counting (o200k_base, safety margins), thread safety (instance locks + ContextVar).
Tests: 37/37 green.

## Phase 1 — ModelRegistry + ProviderRouter
**Status:** Complete | **Date:** 2026-03-16

Created: ModelRegistry (20+ models, LiteLLM refresh, 4-tier lookup), ProviderRouter (10 providers, OpenAI-compatible base_url routing).
Updated ModelPricingEngine to use registry. Tests: 13/13 green.

## Phase 2 — Two-Tier Semantic Cache Rebuild
**Status:** Complete | **Date:** 2026-03-16

Rebuilt CacheMiddleware as TwoTierCacheMiddleware: Redis L1 exact + FAISS L2 semantic. Cross-model prevention via key design + model prefix post-filter. Temperature > 0.5 and side-effect tools skip cache.
Tests: 7/7 unit green.

## Phase 3 — Atomic Budget Enforcement
**Status:** Complete | **Date:** 2026-03-16

Created: RedisBudgetStore (Lua scripts, microdollar precision), InMemoryBudgetStore (TTLCache + RLock), AsyncInMemoryBudgetStore, NormalizedUsage (Anthropic input_tokens fix).
Tests: 13/13 green.

## Phase 4 — Framework Integrations
**Status:** Complete | **Date:** 2026-03-16

Rebuilt: LangChain as BaseChatModel wrapper (not callbacks), CrewAI hooks with side-channel cache, OpenAI Agents SDK Model interface.
Tests: 9/9 green.

## Phase 5 — Unified Error Handling
**Status:** Complete | **Date:** 2026-03-16

Created: classify_error() across OpenAI, Anthropic, Google GenAI, httpx. CRITICAL: OpenAI insufficient_quota 429 is NOT retryable. Anthropic OverloadedError 529 is always retryable.
Tests: 8/8 green.

## Phase 6 — Observability
**Status:** Complete | **Date:** 2026-03-16

Created: OTel GenAI spans (semconv v1.40), structlog JSON logging with trace context, Prometheus metrics (cache hits, cost, budget, errors).
Tests: 4/4 green.

## Phase 7 — Final Test Suite Rebuild
**Status:** Complete | **Date:** 2026-03-16

Added behavioral tests for pricing, budget boundaries, compression exactness. Total: 103 unit tests, all green. No construction-only tests.

## Phase 8 — PyPI v0.2.0
**Status:** Complete | **Date:** 2026-03-16

Updated: pyproject.toml (v0.2.0, Python >=3.11, optional deps), __init__.py (full public API), README.md (production language, changelog, comparison table).
Version verified: `import agentfuse; agentfuse.__version__ == "0.2.0"`.

---

## Loop Iteration 1
**Date:** 2026-03-16

Fixes:
- Fixed L2 cache eviction data loss — vectors now stored alongside metadata for re-indexing
- Upgraded CostAwareRetry to use classify_error instead of string matching
- Added Llama/Groq/Together/o1 models to registry
- Removed unused imports in cache.py

Tests added: 8 new error classifier tests, 1 eviction test
Total: 112 unit tests, all green
Commits: 4

## Loop Iteration 2
**Date:** 2026-03-16

Fixes:
- Added Mistral/DeepSeek/Grok/Llama token counting with tiktoken + safety margins
- Added o3/gpt-4.1/o1 to BudgetEngine.DOWNGRADE_MAP
- Exported budget stores from storage.__init__
- Added 3 new downgrade path tests, 3 new tokenizer tests

Total: 118 unit tests, all green
Commits: 3
