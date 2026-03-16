# AgentFuse

> Your agents. In budget. Intelligently.

![PyPI](https://img.shields.io/pypi/v/agentfuse-runtime)
![npm](https://img.shields.io/npm/v/agentfuse)
![License](https://img.shields.io/badge/license-MIT-blue)

## The Problem

AI agents burn money without warning. A stuck loop can cost $50 in 10 minutes. Retries on a failed call can triple your bill. There's no built-in way to enforce per-run budgets across LLM providers.

## The Solution

AgentFuse intercepts every LLM call, caches semantically similar prompts at 87.5% hit rate, and enforces graduated budget policies so your agent degrades gracefully instead of burning your budget.

## Key Numbers

- **87.5% cache hit rate** on real Anthropic API calls (100-call benchmark)
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

wrap_openai(budget=5.00, run_id="my_agent")
# All subsequent openai.chat.completions.create() calls are now
# budget-enforced and semantically cached
```

## Comparison

| Feature | AgentFuse | AgentBudget | LiteLLM | Portkey |
|---|---|---|---|---|
| Per-run budget enforcement | Yes | Yes | No | No |
| Semantic caching (87.5% hit rate) | Yes | No | No | Basic |
| Mid-run model switching (at 80% budget) | Yes | No | No | No |
| Context compression (at 90% budget) | Yes | No | No | No |
| Graceful termination + partial results | Yes | No | No | No |
| Semantic loop detection | Yes | No | No | No |
| Retry storm prevention | Yes | No | Yes | Yes |
| Streaming cost abort (max_stream_cost) | Yes | No | No | No |
| Auto Anthropic prompt caching | Yes | No | No | No |
| Structured cost receipts (JSON) | Yes | No | Basic | Basic |
| LangChain / CrewAI integration | Yes | No | Yes | Yes |

## Features

**Budget Enforcement**
- Per-run budget limits with graduated degradation
- Automatic model downgrade at 80% budget (e.g. GPT-4o to GPT-4o-mini)
- Context compression at 90% budget
- Graceful termination at 100% with partial results

**Semantic Caching**
- 3-tier semantic cache powered by FAISS + sentence-transformers
- Tier 1 (similarity >= 0.85): Direct hit, zero cost
- Tier 2 (similarity 0.70-0.85): Adapted hit, minimal cost
- Tier 3 (miss): Full LLM call

**Reliability**
- Semantic loop detection with cost-aware thresholds
- Cost-aware retry with automatic model downgrade
- Streaming cost abort — kill streams that exceed max_stream_cost
- Anthropic prompt caching — auto-injects cache_control markers

**Framework Integrations**
- OpenAI (wrap_openai)
- Anthropic (wrap_anthropic)
- LangChain (AgentFuseLangChainMiddleware)
- CrewAI (agentfuse_hooks)
- OpenAI Agents SDK (AgentFuseRunHooks)

**Observability**
- Structured JSON cost receipts per run
- Per-step logging: model, tokens, cost, cache tier, latency
- Budget alert callbacks at configurable thresholds

## Framework Integrations

### OpenAI

```python
from agentfuse import wrap_openai
wrap_openai(budget=5.00, run_id="my_agent")
```

### Anthropic

```python
from agentfuse import wrap_anthropic
wrap_anthropic(budget=5.00, run_id="my_agent")
```

### LangChain

```python
from agentfuse.integrations.langchain import AgentFuseLangChainMiddleware
middleware = AgentFuseLangChainMiddleware(budget=5.00)
agent = initialize_agent(..., callbacks=[middleware])
```

### CrewAI

```python
from agentfuse.integrations.crewai import agentfuse_hooks
before, after = agentfuse_hooks(budget=5.00)
```

### OpenAI Agents SDK

```python
from agentfuse.integrations.openai_agents import AgentFuseRunHooks
result = await Runner.run(agent, hooks=AgentFuseRunHooks(budget=5.00))
```

## Install

```bash
pip install agentfuse-runtime        # Python
npm install agentfuse                # TypeScript (coming Week 5)
```

## Roadmap

- [x] Python SDK — budget enforcement + semantic caching
- [x] LangChain + CrewAI + OpenAI Agents SDK integrations
- [ ] TypeScript SDK (Week 5)
- [ ] Cloud dashboard (Week 6)
- [ ] Anomaly detection (Week 8)

## Contributing

Star the repo, open issues, PRs welcome.

```bash
git clone https://github.com/vinaybudideti/agentfuse.git
cd agentfuse
pip install -e ".[all]"
pytest tests/ -v
```

## License

MIT

---

Built by [@vinaybudideti](https://github.com/vinaybudideti)
