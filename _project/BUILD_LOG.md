# AgentFuse Build Log

## Phase 0 — Fix 3 Critical Production Bugs
**Status:** Complete
**Date:** 2026-03-15

Fixed:
- Cache key design: SHA-256 keys include model, temperature, tools, tenant_id — cross-model contamination impossible
- Token counting: o200k_base for GPT-4o/o-series, 1.20x margin for Anthropic, 1.25x for Gemini
- Thread safety: instance-level threading.Lock + lazy asyncio.Lock, ContextVar for per-run isolation

Tests: 37/37 green (13 budget + 14 cache key + 10 token counting)
