# AgentFuse — Context for New Agent Sessions

## What This Project Is
AgentFuse is a production-grade Python SDK for LLM agent cost optimization. It provides per-run budget enforcement, semantic caching, intelligent model routing, and unified observability across 12+ LLM providers.

## Current State (2026-03-16)
- **Version:** 0.2.0
- **Tests:** 439 unit tests, ALL GREEN
- **Commits:** 128, all pushed to github.com/vinaybudideti/agentfuse
- **Exports:** 47 public API functions/classes
- **Coverage:** 88% on core modules

## Key Entry Points
- `agentfuse/gateway.py` — `completion()` function, the unified entry point
- `agentfuse/__init__.py` — all 47 public exports
- `agentfuse/core/` — 20+ core modules (budget, cache, retry, routing, etc.)
- `agentfuse/providers/` — provider-specific code (openai, anthropic, pricing, tokenizer)
- `tests/unit/` — 30+ test files

## Architecture
The system uses a gateway pattern (like LiteLLM) where `completion()` handles:
1. Request optimization (remove empty/duplicate messages)
2. Intelligent model routing (simple queries → cheap models)
3. Cache lookup (L1 Redis exact + L2 FAISS semantic)
4. Budget enforcement (graduated: alert 60%, downgrade 80%, compress 90%, terminate 100%)
5. Provider routing (auto-detect from model name)
6. Cost recording (normalized across providers, Anthropic cache billing)
7. Response validation (don't cache truncated/garbage)
8. Observability (OTel spans, Prometheus metrics, structlog)

## Important Files
- `_project/BUILD_LOG.md` — complete build history
- `_project/RESEARCH_QUESTIONS.md` — blockers needing web research
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
cd /Users/vinaykumarreddy/Documents/llm-project-2026/Project_Building
source .venv/bin/activate
pytest tests/unit/ -v  # unit tests
pytest tests/unit/ tests/test_*.py -v  # full safe suite
```

## Research Applied
Built with insights from LiteLLM, Portkey, Helicone architectures and 8 academic papers (RouteLLM ICLR 2025, GPT Semantic Cache, ContextCache, Redis langcache-embed-v2, cache poisoning defense). See `_project/BUILD_LOG.md` for details.
