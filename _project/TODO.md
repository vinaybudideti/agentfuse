# AgentFuse TODO List
## Generated from DEEP_RESEARCH_DOC.md | Priority-ordered

---

## PHASE A: Critical Production Fixes (Do First)

### A1. Implement Batch API Submission ✅ DONE
- [x] BatchSubmitter with OpenAI + Anthropic batch endpoints
- [x] 10 tests, estimate_savings()

### A2. Implement Redis Vector Store (L2 Backend) ✅ DONE
- [x] RedisVectorStore with HNSW + TAG hybrid filtering
- [x] 7 tests, graceful fallback

### A3. Add Streaming Response Caching ✅ DONE
- [ ] After stream completes, accumulate full response text
- [ ] Cache the accumulated response (same as non-streaming)
- [ ] On cache hit for a streaming request, simulate streaming from cached text
- [ ] Tests: 5+ tests for stream caching
- **Impact**: Cache savings apply to streaming workloads too

### A4. Native Async Provider Calls
- [ ] Use `openai.AsyncOpenAI` in `_call_openai_compatible_async()`
- [ ] Use `anthropic.AsyncAnthropic` in `_call_anthropic_async()`
- [ ] Update `acompletion()` to use native async instead of run_in_executor
- [ ] Tests: 5+ async tests
- **Impact**: Better performance for high-concurrency async apps

---

## PHASE B: Code Quality & Coverage (Do Second)

### B1. Remove Dead Code
- [ ] Remove `storage/sqlite.py` (empty stub)
- [ ] Remove `storage/redis.py` (empty stub, replaced by redis_store.py)
- [ ] Remove `dashboard/server.py` and `dashboard/routes.py` (empty stubs)
- [ ] Evaluate `intent_atoms/` — remove if truly unused

### B2. Improve Provider Wrapper Coverage
- [ ] Add mock-based tests for `providers/openai.py` (target: 60%+)
- [ ] Add mock-based tests for `providers/anthropic.py` (target: 60%+)
- [ ] Test wrap_openai/wrap_anthropic with mocked SDK clients

### B3. Improve Gateway Coverage
- [ ] Add tests for `_call_openai_compatible()` with mocked `openai.OpenAI`
- [ ] Add tests for `_call_anthropic()` with mocked `anthropic.Anthropic`
- [ ] Test auto_route path more thoroughly
- [ ] Test metadata parameter flow

### B4. Fix Backward Compat Aliases
- [ ] Document all aliases in a `MIGRATION.md`
- [ ] Add deprecation warnings to old aliases
- [ ] Plan removal for v0.3.0

---

## PHASE C: Dashboard & Admin (Do Third)

### C1. Minimal Cost Dashboard API
- [ ] Create `agentfuse/dashboard/api.py` with FastAPI routes
- [ ] GET /api/spend — returns get_spend_report()
- [ ] GET /api/forecast — returns CostForecast predictions
- [ ] GET /api/analytics — returns UsageAnalytics insights
- [ ] GET /api/health — returns system health
- [ ] GET /api/models — returns registry model list
- [ ] Tests: 6+ API tests

### C2. Config File Support
- [ ] Support `agentfuse.yaml` or `agentfuse.toml` config files
- [ ] Auto-load from project root or `~/.agentfuse/config.yaml`
- [ ] Support all configure() options in config file

---

## PHASE D: Advanced Features (Do Fourth)

### D1. RouteLLM ML-Based Router
- [ ] Integrate `pip install routellm` for ML-based complexity routing
- [ ] Use MF router with calibrated thresholds
- [ ] Fallback to heuristic router if routellm not installed
- [ ] Tests: 5+ tests

### D2. PII Detection Integration
- [ ] Optional `pip install presidio-analyzer` integration
- [ ] Pre-cache PII masking: "john@example.com" → "<EMAIL>"
- [ ] Reversible masking for response delivery
- [ ] Tests: 5+ tests

### D3. Context-Aware Cache Invalidation
- [ ] Flush cache entries when model changes (model name in key handles this ✅)
- [ ] Time-based invalidation for knowledge-sensitive queries
- [ ] Admin API to purge cache by tenant/model/pattern

---

## PHASE E: Continuous Improvement (Ongoing)

### E1. Keep Tests Growing
- [ ] Target: 1000 tests
- [ ] Add property-based tests (hypothesis)
- [ ] Add fuzz tests for input validation

### E2. Keep Coverage Growing
- [ ] Target: 95% core coverage
- [ ] Add edge case tests for every module below 90%

### E3. Monitor Competition
- [ ] Weekly check: AgentBudget, LiteLLM, Bifrost releases
- [ ] Monthly check: new LLM providers/models
- [ ] Quarterly check: pricing changes

### E4. Update Dependencies
- [ ] Monthly pip-audit
- [ ] Track tiktoken, sentence-transformers updates
- [ ] Track redis-py, faiss-cpu updates

---

## PHASE F: From Online Research (March 17, 2026)

### F1. MCP (Model Context Protocol) Integration — HIGH PRIORITY
- [ ] 97M+ monthly downloads — industry standard
- [ ] Expose AgentFuse as an MCP tool (budget check, cost estimate)
- [ ] Accept MCP tool costs in ToolCostTracker

### F2. OpenAI Responses API Support
- [ ] GPT-5.3/5.4 use Responses API (not Chat Completions)
- [ ] New `text.format` structure for structured outputs
- [ ] GPT-5.3-Codex is Responses API only

### F3. SAFE-CACHE Defense (NDSS Paper)
- [ ] Intent-level validation layer beyond cosine similarity
- [ ] Embedding version tracking (detect drift on model updates)
- [ ] Cache miss rate monitoring with alerting

### F4. LangGraph Integration
- [ ] LangGraph is production leader (38M monthly PyPI downloads)
- [ ] Cost-tracking callback/middleware for LangGraph nodes
- [ ] Budget enforcement for graph-based workflows

### F5. Pricing Registry Updates ✅ DONE
- [x] GPT-5.4 corrected to $10/$30 (was $2.50/$15)
- [x] Mistral Medium 3, DeepSeek V3.2, Llama 4 Maverick added
- [x] GPT-5.1 deprecation tracked
