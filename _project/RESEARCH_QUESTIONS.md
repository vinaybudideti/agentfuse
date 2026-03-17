# AgentFuse — Research Questions for Web Claude

These are the current blockers and research needs for the AgentFuse project.
Copy each block to Web Claude and ask it to do deep research. Paste the answers back.

**STATUS: Blocks 1-10 answered in research files 1 and 2. New questions below.**

---

## BLOCK 11: Redis 8 native vector search for production L2 cache

The research confirmed Redis 8+ has vector search built-in (no separate Stack needed).
We currently use FAISS for L2 semantic cache. Questions:

1. What is the EXACT Python code to create an HNSW vector index in Redis 8 using `redis-py`?
   Need: full working example with `redis.commands.search.field.VectorField`, `IndexDefinition`, etc.
2. What is the EXACT Python code to perform a KNN search with pre-filtering by model name tag?
3. Memory footprint at 100K entries × 768-dim float32 — how much RAM does Redis HNSW use?
4. Can we run Redis 8 vector search AND our existing Redis L1 cache on the SAME instance?
5. What is the latency comparison: Redis HNSW KNN vs FAISS IndexFlatIP at 10K/100K/1M vectors?
6. Does Redis vector search support hybrid filtering (TAG + vector) in a single query?
7. What redis-py version supports vector search? Is it `redis>=5.0.0`?

---

## BLOCK 12: Production-grade async ingestion pipeline

The research showed LiteLLM's sync PostgreSQL in the request path is a bottleneck.
Our SpendLedger uses append-only JSONL. Questions for scaling:

1. What is the best Python async message queue for decoupling cost recording from the hot path?
   - Redis Streams vs Kafka (via aiokafka) vs simple asyncio.Queue?
2. For ClickHouse as the analytics DB (Helicone pattern), what is the Python client?
   - clickhouse-connect? clickhouse-driver?
3. How to batch-insert spend events into ClickHouse efficiently? (batch size, interval)
4. Is there a Python library for NATS JetStream that could replace Kafka for simpler deployments?
5. What is the recommended way to run ClickHouse locally for development? Docker image?

---

## BLOCK 13: Microsoft Presidio PII detection integration

The research recommended Presidio for pre-cache PII detection. Questions:

1. What is the current `presidio-analyzer` Python package version and API?
2. How to detect PII in text and replace with type placeholders? (e.g., "john@example.com" → "<EMAIL>")
3. What entity types does Presidio detect? (EMAIL, PHONE, SSN, CREDIT_CARD, etc.)
4. Performance: how long does Presidio take to scan a 1000-token prompt?
5. Can Presidio run entirely offline (no API calls)?
6. Is there a way to do reversible masking (replace PII, then restore later)?

---

## BLOCK 14: RouteLLM deep integration

AgentFuse has a lightweight complexity router. RouteLLM (ICLR 2025) is pip-installable. Questions:

1. What is the EXACT Python code to use RouteLLM's matrix factorization router?
   - `pip install routellm` → then what? What API do we call?
2. What model pairs are the pre-trained routers trained on? Can they be used for GPT-5 vs GPT-4.1-nano?
3. Does RouteLLM support custom model pairs? How to retrain for our models?
4. What is the latency overhead of the MF router? (ms per routing decision)
5. Can the MF router work with embedding-based routing for multi-tier routing?
   (e.g., GPT-5 for complex → GPT-4.1 for medium → GPT-4.1-nano for simple)

---

## BLOCK 15: Real-world production deployments

We need real-world validation. Questions:

1. What companies use LiteLLM Proxy in production? What scale (requests/day)?
2. What is the actual cost savings from semantic caching in production?
   - GPTCache claimed 50-70% but is now abandoned — were those real?
   - Portkey claims 20% hit rate — is that for RAG or chatbot workloads?
3. What is the typical semantic cache FALSE POSITIVE rate in production?
   - Research says <1% at threshold 0.90 — is this verified in production?
4. How do production teams handle cache INVALIDATION when they deploy new models?
   (e.g., upgrading from GPT-4o to GPT-4.1 — do they flush the cache?)
5. What is the typical budget enforcement failure mode?
   (e.g., budget exceeded before enforcement triggers because cost estimation is inaccurate)

---

## BLOCK 16: Streaming cost tracking accuracy

AgentFuse needs to track cost for streaming responses. Questions:

1. OpenAI `stream_options={"include_usage": True}` — does this work reliably?
   What does the final chunk look like? Is it always the LAST chunk?
2. Anthropic streaming — does `message_delta` always contain cumulative `output_tokens`?
3. For Gemini streaming — how are usage tokens reported?
4. What is the best pattern for accumulating token counts across stream chunks?
5. Should we cache streaming responses? If so, how to reconstruct the full response?

---

## BLOCK 17: OpenAI Batch API and Anthropic Message Batches

AgentFuse detects batch-eligible workloads. Questions for real implementation:

1. OpenAI Batch API — what is the EXACT Python code to submit a batch?
   - How to create the JSONL input file?
   - How to poll for completion?
   - How to retrieve results?
2. Anthropic Message Batches — what is the EXACT Python code?
   - `client.messages.batches.create(...)` — what's the schema?
3. Do batch discounts stack with prompt caching? (Research says yes for Anthropic)
4. What is the typical turnaround time for batch jobs? (minutes? hours?)
5. Is there a webhook for batch completion, or must we poll?

---

## BLOCK 18: Claude Extended Thinking cost implications

Claude models now support extended thinking. Questions:

1. Are `thinking` tokens billed at output rate or a different rate?
2. Does the `thinking` content appear in `output_tokens` or separately?
3. Can thinking content be cached? (Probably not useful, but need to confirm)
4. What is the `signature` field in thinking content blocks used for?
5. How to detect if a response used extended thinking? (Check for thinking blocks in content?)

---

## BLOCK 19: gpt-oss-* models and o200k_harmony encoding

The research mentions `gpt-oss-*` models use `o200k_harmony` encoding (new). Questions:

1. What are `gpt-oss-*` models? When were they released?
2. Is `o200k_harmony` available in tiktoken? What version?
3. What is the vocabulary size difference between `o200k_base` and `o200k_harmony`?
4. Are there any pricing differences for `gpt-oss-*` models?
5. Should AgentFuse support `gpt-oss-*` models? Who uses them?

---

## BLOCK 20: Competitive feature analysis — AgentBudget, Bifrost, agentmolt

The research identified new competitors. Deep dive:

1. AgentBudget — what features does it have that AgentFuse lacks?
   - How does `agentbudget.init("$5.00")` work under the hood?
   - Does it support semantic caching? Model routing?
2. Bifrost (Maxim AI) — they claim 50× faster than LiteLLM with 11µs overhead.
   - Is this real? What benchmarks prove it?
   - Does it support per-run budgets?
3. agentmolt — what is "kill-switch" functionality?
   - Is this similar to BudgetExhaustedGracefully?
4. Are any of these gaining significant GitHub traction? (stars, contributors)
