# AgentFuse Deep Research Document
## Generated: 2026-03-17 | Based on codebase audit + online research + 3 research files

---

## CURRENT STATE SUMMARY
- **900 unit tests** | 76 exports | 93% core coverage | 231 commits
- **41 core modules** | 10 provider modules | 3 integration modules
- **20-step gateway flow** | 15 novel inventions
- **0 CVEs** | pip-audit clean

---

## CRITICAL GAPS IDENTIFIED

### GAP 1: No Actual Batch API Execution (MEDIUM-HIGH)
- `BatchEligibilityDetector` detects opportunities but does NOT submit batches
- OpenAI Batch API: 50% discount, 50K requests/file, 24h SLA
- Anthropic Message Batches: 50% discount, 100K requests/batch, stacks with caching (95% savings)
- **Impact**: Companies lose 50% savings on non-real-time workloads
- **Fix**: Implement `BatchSubmitter` with OpenAI/Anthropic batch endpoints

### GAP 2: No Redis HNSW Vector Search (HIGH)
- L2 cache uses FAISS IndexFlatIP (in-process, single machine)
- Production needs Redis HNSW for: persistence, concurrent access, 1M+ vectors
- Redis 8 has native vector search (no separate module needed)
- Research file 3 has exact code for HNSW index creation + KNN search
- **Impact**: L2 cache doesn't survive restarts, can't scale horizontally
- **Fix**: Add `RedisVectorStore` as alternative L2 backend behind protocol

### GAP 3: Dashboard is Stub Only (MEDIUM)
- `agentfuse/dashboard/` has empty server.py and routes.py
- Companies need: cost visualization, cache hit rate charts, budget utilization
- **Fix**: Add minimal FastAPI/Flask routes that serve JSON data from SpendLedger

### GAP 4: No Native Async Provider Calls (MEDIUM)
- `acompletion()` uses `run_in_executor` (wraps sync in thread pool)
- Should use `openai.AsyncOpenAI` and `anthropic.AsyncAnthropic` natively
- **Impact**: Thread pool overhead, less efficient for high-concurrency

### GAP 5: Streaming Response Caching (MEDIUM)
- Streaming responses are NOT cached (only non-streaming are)
- Pattern: accumulate full response after stream completes, then cache
- Research confirms this is the recommended approach

### GAP 6: Dead Code / Stubs (LOW)
- `storage/sqlite.py` — empty stub
- `storage/redis.py` — empty stub
- `dashboard/server.py` — empty stub
- `dashboard/routes.py` — empty stub
- `intent_atoms/` — legacy subtree (0% coverage, ~1000 lines)

### GAP 7: No `.env` / YAML Config (LOW)
- Only `configure()` function and env vars
- Production needs config file support for complex setups

---

## BUGS / CODE QUALITY ISSUES

### BUG 1: `_flexible_store` monkey-patch at module level (cache.py:490-510)
- Monkey-patches `TwoTierCacheMiddleware.store` at import time
- Works but is a code smell — should use proper method dispatch

### BUG 2: Backward compat aliases scattered
- `CacheMiddleware = TwoTierCacheMiddleware`
- `AgentFuseLangChainMiddleware = AgentFuseChatModel`
- `agentfuse_hooks = create_agentfuse_hooks`
- These should be documented or deprecated

### BUG 3: Provider wrappers have low coverage (12-24%)
- `openai.py` and `anthropic.py` are mostly untested
- They require real API keys to test fully
- Should add more mock-based tests

---

## COMPETITIVE INTELLIGENCE (March 2026)

### AgentBudget (13 stars)
- Has: budget enforcement, loop detection, tool cost tracking
- Missing: semantic caching, model routing, context compression
- **Our advantage**: 15 novel features they lack

### LiteLLM (16K+ stars, 91M downloads/month)
- Has: 100+ provider routing, budget enforcement, spend tracking
- Bugs: PostgreSQL bottleneck, 44% throughput drop at 500 RPS
- **Our advantage**: semantic caching, predictive routing, kill switch

### Bifrost (2.3K stars)
- Has: 50× faster than LiteLLM (Go-based), semantic caching
- Missing: Python SDK (it's a Go gateway)
- **Our advantage**: in-process Python SDK, no infrastructure needed

### Portkey (10.9K stars)
- Has: composable routing, semantic caching (proprietary)
- Missing: open-source semantic caching
- **Our advantage**: open-source semantic caching, budget enforcement

---

## PRICING UPDATES NEEDED

### Models to Add
- GPT-5.3 Instant → already added ✅
- Claude Sonnet 4.5 → already added ✅
- GPT Image 1.5 → in tool costs ✅

### Deprecation Updates
- GPT-4, GPT-4o, GPT-4.1 deprecated Feb 13, 2026 → tracked ✅
- Claude Haiku 3 deprecated Apr 19, 2026 → tracked ✅
- GPT-5.1 deprecated → GPT-5.3 replaced it → tracked ✅

---

## TEST COVERAGE GAPS

| Module | Coverage | Root Cause |
|--------|----------|------------|
| providers/anthropic.py | 12% | Requires real API key |
| providers/openai.py | 24% | Requires real API key |
| integrations/openai_agents.py | 60% | Async paths untested |
| observability/logging.py | 59% | OTel trace context injection |
| observability/metrics.py | 67% | Helper functions |
| gateway.py | 69% | Provider call paths |
| core/loop.py | 75% | SentenceTransformer init |
| core/cache.py | 79% | Redis paths, backward compat |
