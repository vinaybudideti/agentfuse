# AgentFuse — Research Questions for Web Claude

These are the blockers and research needs for the AgentFuse project.
Copy each block to Web Claude and paste the answers back.

---

## BLOCK 1: Real API response format validation

I need the EXACT JSON structure of these API responses as of 2026:

1. OpenAI chat.completions.create() response — full JSON including:
   - usage.prompt_tokens_details (cached_tokens, audio_tokens)
   - usage.completion_tokens_details (reasoning_tokens, audio_tokens)
   - What does the response look like when stream=True with stream_options={"include_usage": True}?
   - What does the final chunk contain?

2. Anthropic messages.create() response — full JSON including:
   - usage.input_tokens, cache_read_input_tokens, cache_creation_input_tokens
   - What does stop_reason look like for tool_use vs end_turn vs max_tokens?
   - What is the exact structure of a tool_use content block?

3. Google Gemini generateContent response — full JSON including:
   - usageMetadata fields (promptTokenCount, candidatesTokenCount, etc.)
   - What is the exact structure when thinking/reasoning is enabled?

I need the RAW JSON, not documentation summaries. Our extract_usage() needs to handle every field correctly.

---

## BLOCK 2: tiktoken encoding accuracy for newer models

1. Does GPT-4.1 use o200k_base or a new encoding?
2. Does o3 use o200k_base?
3. Does o4-mini use o200k_base?
4. What encoding does Claude use internally? Is cl100k_base × 1.20 still a good approximation?
5. Has Anthropic released a public tokenizer for Python yet?
6. What tokenizer does Gemini 2.5 Pro use? Is there a Python package?

---

## BLOCK 3: Current model pricing (March 2026)

Verify these prices are still current (per 1M tokens):
- GPT-4.1: $2.00 input, $8.00 output — still correct?
- o3: $2.00 input, $8.00 output — still correct?
- o4-mini: $1.10 input, $4.40 output — still correct?
- Claude Sonnet 4.6: $3.00 input, $15.00 output — still correct?
- Claude Haiku 4.5: $1.00 input, $5.00 output — still correct?
- Claude Opus 4.6: $5.00 input, $25.00 output — still correct?
- Gemini 2.5 Pro: $1.25 input, $10.00 output — still correct?
- Are there any NEW models released since March 2026 we're missing?
- Has GPT-5 been released? What's its pricing?
- Has Claude 4.7/5 been released?

---

## BLOCK 4: Anthropic prompt caching exact behavior

1. What is the EXACT minimum token count for cache_control on each Claude model?
   - Sonnet: 1024? Still?
   - Haiku: 2048? Still?
   - Opus: 1024? Still?
2. What is the cache TTL? 5 minutes? Has it changed?
3. Can you cache_control multiple blocks or only the last one?
4. What is the exact pricing for cache_creation_input_tokens? 1.25x base?
5. What happens if you send cache_control on a block with fewer tokens than minimum?

---

## BLOCK 5: OpenAI Agents SDK current API

1. What is the current Model interface in openai-agents Python package?
2. Has the get_response() method signature changed?
3. What is ModelProvider.get_model() expected to return?
4. Is there a newer integration pattern than Model/ModelProvider?

---

## BLOCK 6: LangChain current API (v0.3+)

1. What is the current BaseChatModel interface in langchain-core?
2. Has _generate() or invoke() signature changed?
3. What message types exist? (HumanMessage, AIMessage, SystemMessage, ToolMessage, FunctionMessage?)
4. How does the callback system work now? Is on_chat_model_start still the primary hook?

---

## BLOCK 7: Redis semantic search capability

1. Does Redis now support vector similarity search natively (RediSearch)?
2. If so, should we use Redis HNSW index instead of FAISS for L2?
   This would give us a single L1+L2 backend instead of Redis+FAISS.
3. What is the query syntax for Redis vector search?
4. Performance: Redis HNSW vs FAISS IndexFlatIP at 100K vectors?

---

## BLOCK 8: Competitive landscape

1. Has any new LLM cost optimization tool launched since January 2026?
2. Has LiteLLM fixed their PostgreSQL bottleneck (GitHub #12067)?
3. Has Portkey open-sourced their semantic caching?
4. What is GPTCache's status — still abandoned?
5. Is there a RouteLLM Python package we can integrate directly?

---

## BLOCK 9: Production deployment patterns

1. What is the recommended way to deploy an LLM proxy in production?
   - Kubernetes sidecar?
   - Standalone FastAPI service?
   - Library import (our approach)?
2. How do companies handle API key rotation in production?
3. What monitoring/alerting stack do teams use for LLM costs?
   - Datadog? Grafana? Custom?

---

## BLOCK 10: Security concerns

1. What are the known prompt injection attacks that affect semantic caches?
2. How do production systems prevent API key leakage in logs?
3. Are there any CVEs in faiss-cpu, sentence-transformers, or tiktoken?
4. What is the recommended way to sanitize LLM responses before caching?
