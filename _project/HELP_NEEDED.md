# Help Needed from Vinay

## 1. API Keys for Integration Testing (HIGH PRIORITY)
The provider wrappers (openai.py, anthropic.py) have only 12-24% test coverage because they need real API keys to test the actual API call paths. If you can set these environment variables, I can write integration tests that make real (cheap) API calls:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Even temporary keys with low rate limits would help verify the full gateway flow end-to-end.

## 2. Redis Instance for Cache Testing (MEDIUM)
Our L1 cache and RedisVectorStore tests run in "fallback mode" (no Redis). A local Redis would let me test:
- L1 Redis exact-match cache
- Redis HNSW vector search (L2)
- RedisBudgetStore atomic Lua scripts
- Circuit breaker behavior

```bash
# Docker one-liner:
docker run -d -p 6379:6379 redis:8
```

## 3. Research File for New Questions (LOW)
I've updated `_project/RESEARCH_QUESTIONS.md` with blocks 11-20. If you can paste any of those blocks into web Claude and bring back the answers, I can apply them immediately. Most impactful blocks:

- **Block 11**: Redis 8 vector search exact Python code (partially answered in file 3)
- **Block 14**: RouteLLM deep integration (MF router calibration)
- **Block 17**: OpenAI/Anthropic Batch API exact schemas
- **Block 18**: Claude extended thinking cost implications

## 4. PyPI Publish (WHEN READY)
When you want to publish v0.2.1 to PyPI, I'll need:
```bash
export PYPI_TOKEN="pypi-..."
```
Then I can build and publish automatically.

## 5. GitHub Actions CI (NICE TO HAVE)
If you want automated testing on every push, I can create a `.github/workflows/ci.yml` file. Just confirm you want it.

## 6. Feedback on Direction
Are there specific features your target users/companies need most? For example:
- Should I focus more on **cost dashboards** (visual UI)?
- Should I focus on **more provider support** (Azure OpenAI, AWS Bedrock)?
- Should I focus on **performance optimization** (faster cache lookups)?
- Should I focus on **documentation** (API docs, tutorials)?

Your guidance helps me prioritize the continuous improvement loop.

---

*This file will be updated as new needs arise. Check back anytime.*
*Last updated: 2026-03-18*
