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

---

## Research-Backed Feature Ideas (from March 2026 research)

### Redis 8 Native Vector Search (L2 Backend)
- Replace FAISS with Redis HNSW for production L2 (single infrastructure dependency)
- Keep FAISS as local/dev fallback
- Redis hybrid filtering (TAG + vector in single query)
- VectorStore protocol to abstract both backends

### RouteLLM Deep Integration
- pip install routellm and use their MF router for real ML-based routing
- Currently using heuristic complexity score — RouteLLM is research-validated
- Achieves 85% cost reduction at 95% GPT-4 quality

### PII Detection (Microsoft Presidio)
- Pre-cache PII detection: "john@example.com" → "<EMAIL>"
- Prevents PII from entering shared cache
- Reversible masking for response delivery
- GDPR compliance: admin endpoint to purge user cache entries

### Async Ingestion Pipeline
- Move SpendLedger from sync file append to async queue
- Redis Streams or asyncio.Queue for non-blocking cost recording
- Optional ClickHouse sink for large-scale analytics

### Batch API Integration
- OpenAI Batch API (50% discount, 24h SLA)
- Anthropic Message Batches (50% discount, stackable with prompt caching)
- Automatic batch-eligible workload detection ✅ DONE (BatchEligibilityDetector)
- Need actual batch submission implementation

### Extended Thinking Cost Tracking
- Claude thinking tokens billed separately
- Track thinking vs output tokens in NormalizedUsage
- Don't cache thinking blocks (internal reasoning)
- Display thinking costs separately in reports

### Key Pool Rate Limit Multiplier
- 6 API keys × 90 RPM = 540 RPM effective capacity
- ModelLoadBalancer already supports multiple endpoints
- Need per-key rate tracking and automatic rotation

### Cross-Encoder Cache Verification (Krites Pattern)
- Serve cached response immediately
- Async verify with lightweight LLM
- Promote verified entries to "trusted" tier
- Helps with cache poisoning defense beyond threshold
