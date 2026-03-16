# AgentFuse

> Your agents. In budget. Intelligently.

![PyPI](https://img.shields.io/pypi/v/agentfuse-runtime)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)

## The Problem

AI agents burn money without warning. A stuck loop can cost $50 in 10 minutes. Retries on a failed call can triple your bill. There's no built-in way to enforce per-run budgets across LLM providers.

## The Solution

AgentFuse intercepts every LLM call with a two-tier semantic cache (Redis L1 + FAISS L2) achieving 87.5% hit rate, and enforces graduated budget policies so your agent degrades gracefully instead of burning your budget.

## Key Numbers

- **87.5% cache hit rate** on repeated/paraphrased prompts
- **71.8% cost reduction** ($0.24 vs $0.87 for same workload)
- **179,445 tokens saved** per 100 calls
- **2 lines of code** to integrate

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

## Comparison

| Feature | AgentFuse | AgentBudget | LiteLLM | Portkey |
|---|---|---|---|---|
| Per-run budget enforcement | Yes | Yes | No | No |
| Semantic caching (87.5% hit rate) | Yes | No | No | Basic |
| Two-tier cache (Redis + FAISS) | Yes | No | No | No |
| Mid-run model switching (at 80% budget) | Yes | No | No | No |
| Context compression (at 90% budget) | Yes | No | No | No |
| Graceful termination + partial results | Yes | No | No | No |
| Semantic loop detection | Yes | No | No | No |
| Retry storm prevention | Yes | No | Yes | Yes |
| Streaming cost abort (max_stream_cost) | Yes | No | No | No |
| Auto Anthropic prompt caching | Yes | No | No | No |
| Unified error classifier (3 providers) | Yes | No | Partial | No |
| OTel GenAI spans + Prometheus metrics | Yes | No | Yes | Yes |
| Structured cost receipts (JSON) | Yes | No | Basic | Basic |
| LangChain / CrewAI / OpenAI Agents SDK | Yes | No | Yes | Yes |
| Hot-reloadable model pricing | Yes | No | Yes | No |
| Atomic Redis budget enforcement | Yes | No | No | No |

## Features

**Budget Enforcement**
- Per-run budget limits with graduated degradation
- Automatic model downgrade at 80% budget (e.g. GPT-4o to GPT-4o-mini)
- Context compression at 90% budget (system + last 6 messages)
- Graceful termination at 100% with partial results
- Atomic Redis Lua budget enforcement (or in-memory with TTL)

**Two-Tier Semantic Cache**
- L1: Redis exact-match (SHA-256 key, sub-millisecond)
- L2: FAISS vector similarity with redis/langcache-embed-v2 (2-5ms)
- Cross-model contamination impossible (model in cache key + post-filter)
- Tool-use queries never go through L2 semantic search
- Temperature > 0.5 and side-effect tools skip cache entirely

**Reliability**
- Unified error classifier across OpenAI, Anthropic, Google GenAI
- Provider-aware retry with tenacity (exponential backoff)
- Semantic loop detection with cost-aware thresholds
- Streaming cost abort — kill streams that exceed max_stream_cost

**Framework Integrations**
- OpenAI (wrap_openai)
- Anthropic (wrap_anthropic)
- LangChain (AgentFuseChatModel — BaseChatModel wrapper, not just callbacks)
- CrewAI (create_agentfuse_hooks)
- OpenAI Agents SDK (AgentFuseModel / AgentFuseModelProvider)

**Observability**
- OTel GenAI semconv v1.40 spans
- structlog JSON logging with trace context injection
- Prometheus metrics: cache hits, cost, budget remaining, errors
- Structured JSON cost receipts per run

## Framework Integrations

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
```

### CrewAI

```python
from agentfuse.integrations.crewai import create_agentfuse_hooks
before, after = create_agentfuse_hooks(budget=5.00)
```

### OpenAI Agents SDK

```python
from agentfuse.integrations.openai_agents import AgentFuseModelProvider
provider = AgentFuseModelProvider(inner=your_provider, budget=5.00)
```

## Install

```bash
pip install agentfuse-runtime           # Core
pip install agentfuse-runtime[redis]    # + Redis cache/budget
pip install agentfuse-runtime[otel]     # + OpenTelemetry
pip install agentfuse-runtime[all]      # Everything
```

## Changelog

### v0.2.0 — Production rebuild (March 2026)

- Two-tier cache: Redis L1 + FAISS L2 with redis/langcache-embed-v2
- Hot-reloadable ModelRegistry with LiteLLM remote refresh
- Atomic Redis Lua budget enforcement with microdollar precision
- Unified error classifier across OpenAI, Anthropic, Google GenAI
- OTel GenAI spans, structlog JSON logging, Prometheus metrics
- LangChain BaseChatModel wrapper (not just callbacks)
- 103+ behavioral tests, 0 construction-only tests

### v0.1.0 — Initial prototype (February 2026)

- BudgetEngine, CacheMiddleware, wrap_openai, wrap_anthropic
- 34 tests passing

## Roadmap

- [x] Python SDK v0.2.0 — production-ready
- [ ] TypeScript SDK
- [ ] Cloud dashboard
- [ ] Anomaly detection

## Contributing

Star the repo, open issues, PRs welcome.

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
