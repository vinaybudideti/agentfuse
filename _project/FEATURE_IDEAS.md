# AgentFuse Feature Ideas

## Future Features (Not Yet Implemented)

### TypeScript SDK
- Port core modules (BudgetEngine, TwoTierCache, ErrorClassifier)
- Use Mastra's inputProcessors for cache integration
- npm package: `agentfuse`

### Cloud Dashboard
- Real-time cost monitoring per agent/run
- Cache hit rate visualization
- Budget utilization alerts
- Team/org-level cost aggregation

### Anomaly Detection
- Per-agent-type baseline cost profiling
- Statistical anomaly detection on cost/latency
- Automatic alerting when agent exceeds expected cost

### Batch Pricing Support
- 50% discount for batch API calls (24h SLA)
- Automatic batch routing for non-time-sensitive tasks

### FAISS Index Auto-Persistence
- Save/load FAISS index on shutdown/startup
- Periodic checkpointing to disk
- Already have save_l2_index/load_l2_index — need lifecycle hooks

### Provider API Token Counting
- Anthropic count_tokens API (free, exact)
- Google count_tokens API
- Would replace tiktoken estimates with exact counts

### Tiered Pricing Engine
- Anthropic >200K overflow (2x input, 1.5x output) ✅ DONE
- Gemini Pro >200K overflow (2x) ✅ DONE
- Volume discounts for high-usage accounts

### Multi-Agent Budget Sharing
- Parent budget distributes to child agents
- Atomic cross-agent budget deduction via Redis

### Context-Aware Cache Invalidation
- Event-driven flush on model updates
- Pattern-matching Redis key deletion
- Corpus version tracking for RAG systems
