# AgentFuse — Context for New Agent Sessions

## What This Project Is
AgentFuse is a production-grade Python SDK for LLM agent cost optimization. It provides per-run budget enforcement, semantic caching, intelligent model routing, and unified observability across 12+ LLM providers.

## Current State (2026-03-18)
- **Version:** 0.2.1
- **Tests:** 1041+ unit tests, ALL GREEN
- **Commits:** 254, all pushed to github.com/vinaybudideti/agentfuse
- **Exports:** 85 public API functions/classes
- **Coverage:** 93% on core modules
- **Core Modules:** 42
- **Test Files:** 82+

## Key Entry Points
- `agentfuse/gateway.py` — `completion()` and `acompletion()` functions
- `agentfuse/__init__.py` — all 85 public exports
- `agentfuse/core/` — 42 core modules
- `agentfuse/providers/` — 10 provider modules (pricing, tokenizer, registry, router, responses_api)
- `agentfuse/integrations/` — 5 integrations (langchain, crewai, openai_agents, mcp, langgraph)
- `agentfuse/storage/` — 4 storage modules (memory, redis_store, spend_ledger, redis_vector_store)
- `agentfuse/observability/` — 3 modules (otel, metrics, logging)
- `agentfuse/dashboard/` — FastAPI cost dashboard (5 endpoints)
- `tests/unit/` — 82+ test files

## Architecture — 20-Step Gateway Flow
1. Kill switch check (outside AI reasoning path)
2. Input validation (fail-fast)
3. Deprecation check (warns on GPT-4/4o/4.1)
4. Rate limiting (GCRA per-tenant)
5. Request optimization (remove empty/duplicate messages)
6. Intelligent model routing (RouteLLM-inspired)
7. Context window guard (auto-compress if overflow)
8. Budget check (graduated: alert 60%, downgrade 80%, compress 90%, terminate 100%)
9. Cache lookup (L1 Redis exact + L2 FAISS/Redis HNSW semantic)
10. Key pool selection (multiple API keys per model)
11. Request deduplication (coalesce identical in-flight)
12. Provider API call (OpenAI/Anthropic/Gemini + Responses API)
13. Automatic model fallback on retryable errors
14. Cost recording (normalized, Anthropic cache billing, audio tokens)
15. Cost alerts (webhook delivery)
16. Anomaly detection (EMA + z-score)
17. Prometheus metrics + OTel spans
18. SpendLedger recording (persistent JSONL)
19. Response validation + security check (XSS, PII)
20. Cache storage (with CacheAttack defense)

## Novel Inventions (17 — no competitor has these)
CostPredictiveRouter, PromptCompressor, ToolCostTracker,
ConversationCostEstimator, HierarchicalBudget, AgentSession,
KillSwitch, ContextWindowGuard, UsageAnalytics, ModelRecommender,
ResponseQualityScorer, CostForecast, ContentGuardrails,
ReportExporter, BatchSubmitter, RedisVectorStore, CacheMonitor

## Important Files
- `_project/BUILD_LOG.md` — complete build history
- `_project/DEEP_RESEARCH_DOC.md` — gap analysis + competitive intelligence
- `_project/TODO.md` — TODO list (ALL 12 COMPLETE)
- `_project/RESEARCH_QUESTIONS.md` — research questions
- `_project/FEATURE_IDEAS.md` — future roadmap
- `PROGRESS.md` — current progress summary
- `CHANGELOG.md` — version changelog

## Rules
- Do NOT mention "Claude" in git commit messages
- Do NOT include Co-Authored-By lines in commits
- Use `python3` not `python`
- All changes must be committed and pushed
- Do not write tests to satisfy errors — fix the actual implementation
- Build genuine solutions, not test-case manipulation

## How to Run Tests
```bash
cd /path/to/agentfuse
source .venv/bin/activate
pytest tests/unit/ -v  # unit tests
pytest tests/unit/ tests/test_*.py -v  # full safe suite
```

## Research Applied
Built with insights from LiteLLM, Portkey, Helicone, Bifrost architectures and 10+ academic papers (RouteLLM ICLR 2025, GPT Semantic Cache, ContextCache, Redis langcache-embed-v2, CacheAttack arXiv 2601.23088, SAFE-CACHE NDSS). Applied 3 deep research files with verified API schemas, pricing, and production patterns. See `_project/BUILD_LOG.md` for details.
