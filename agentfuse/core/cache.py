"""
CacheMiddleware — wraps Intent Atoms 3-tier semantic cache into the AgentFuse pipeline.

Tier 1 (similarity >= 0.85): Direct hit, zero cost.
Tier 2 (similarity 0.70–0.85): Adapted hit, cheap Haiku-level cost.
Tier 3 (similarity < 0.70): Cache miss, full LLM call needed.
"""

import os
import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheHit:
    tier: int       # 1 = direct hit, 2 = adapted hit
    response: str
    cost: float


@dataclass
class CacheMiss:
    pass


class CacheMiddleware:
    """
    Wraps the Intent Atoms FAISSStore to provide a 3-tier semantic cache
    for the AgentFuse budget pipeline.
    """

    def __init__(self, cache_dir="~/.agentfuse/cache"):
        self.engine = None
        self._embedder = None
        self._direct_threshold = 0.85
        self._adapt_threshold = 0.70

        try:
            from agentfuse.intent_atoms import FAISSStore

            expanded = os.path.expanduser(cache_dir)
            self.engine = FAISSStore(
                dimension=768,
                persist_dir=expanded,
            )
        except Exception as e:
            logger.warning("Intent Atoms FAISSStore unavailable: %s", e)
            self.engine = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-mpnet-base-v2")
        return self._embedder

    def _embed(self, text: str) -> list[float]:
        model = self._get_embedder()
        embedding = model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    def check(self, prompt: str, model: str, budget_engine=None):
        """
        Check cache for a prompt. Returns CacheHit or CacheMiss.
        """
        if self.engine is None:
            return CacheMiss()

        try:
            embedding = self._embed(prompt)

            results = self._run_async(
                self.engine.search(
                    embedding=embedding,
                    top_k=1,
                    threshold=self._adapt_threshold,
                )
            )

            if not results:
                return CacheMiss()

            cached_entry, similarity = results[0]

            if similarity >= self._direct_threshold:
                # Tier 1: Direct hit — zero cost
                return CacheHit(tier=1, response=cached_entry.response_text, cost=0.0)

            # Tier 2: Adapt hit — estimate Haiku adaptation cost
            from agentfuse.providers.pricing import ModelPricingEngine
            from agentfuse.providers.tokenizer import TokenCounterAdapter

            t = TokenCounterAdapter()
            p = ModelPricingEngine()
            token_count = t.count_tokens(prompt, "claude-haiku-4-5-20251001")
            adapt_cost = p.input_cost("claude-haiku-4-5-20251001", token_count)

            return CacheHit(tier=2, response=cached_entry.response_text, cost=adapt_cost)

        except Exception as e:
            logger.warning("Cache check failed: %s", e)
            return CacheMiss()

    def store(self, prompt: str, response: str, model: str):
        """
        Store a prompt-response pair in the cache.
        """
        if self.engine is None:
            return

        try:
            embedding = self._embed(prompt)

            self._run_async(
                self.engine.store(
                    embedding=embedding,
                    query_text=prompt,
                    response_text=response,
                    metadata={"model_used": model},
                )
            )
        except Exception as e:
            logger.warning("Cache store failed: %s", e)
