"""
RedisVectorStore — Redis 8 HNSW vector search as L2 cache backend.

Replaces FAISS for production deployments:
- Persistence: survives process restarts (RDB/AOF)
- Concurrency: multiple processes can read/write simultaneously
- Filtering: TAG + vector hybrid search in single query
- Scaling: Redis Cluster for horizontal scaling

Falls back to FAISS for local/dev (zero infrastructure).

Usage:
    store = RedisVectorStore(redis_url="redis://localhost:6379", dim=768)
    store.add(vector, metadata={"model": "gpt-4o", "response": "..."})
    results = store.search(query_vector, model_prefix="openai", top_k=5)
"""

import logging
import time
import numpy as np
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

INDEX_NAME = "idx:agentfuse_cache"
DOC_PREFIX = "l2:"


@dataclass
class VectorSearchResult:
    """Result from a vector search."""
    doc_id: str
    distance: float  # 0 = identical, 2 = opposite (cosine)
    model: str
    model_prefix: str
    has_tools: bool
    response: str
    tenant_id: str = ""

    @property
    def similarity(self) -> float:
        """Convert cosine distance to similarity (0-1)."""
        return max(0.0, 1.0 - self.distance)


class RedisVectorStore:
    """
    Redis 8 HNSW vector search backend for L2 semantic cache.

    Requires: pip install "redis[hiredis]>=5.0.0"
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        dim: int = 768,
        m: int = 16,
        ef_construction: int = 200,
        ef_runtime: int = 10,
    ):
        self._dim = dim
        self._m = m
        self._ef_construction = ef_construction
        self._ef_runtime = ef_runtime
        self._redis = None
        self._index_created = False

        try:
            import redis
            self._redis = redis.Redis.from_url(redis_url, decode_responses=False)
            self._redis.ping()
            self._ensure_index()
            logger.info("RedisVectorStore connected: %s (dim=%d)", redis_url, dim)
        except Exception as e:
            logger.warning("RedisVectorStore unavailable: %s", e)
            self._redis = None

    def _ensure_index(self):
        """Create HNSW index if it doesn't exist."""
        if self._index_created or self._redis is None:
            return

        try:
            from redis.commands.search.field import TagField, TextField, NumericField, VectorField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType

            # Check if index exists
            try:
                self._redis.ft(INDEX_NAME).info()
                self._index_created = True
                return
            except Exception:
                pass

            schema = (
                TagField("model_prefix"),
                TagField("has_tools"),
                TagField("tenant_id"),
                TextField("model"),
                TextField("response"),
                NumericField("timestamp"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._dim,
                        "DISTANCE_METRIC": "COSINE",
                        "M": self._m,
                        "EF_CONSTRUCTION": self._ef_construction,
                        "EF_RUNTIME": self._ef_runtime,
                    },
                ),
            )
            definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)
            self._redis.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
            self._index_created = True
            logger.info("Redis HNSW index created: %s", INDEX_NAME)
        except ImportError:
            logger.warning("redis[hiredis] >= 5.0.0 required for vector search")
        except Exception as e:
            logger.warning("Failed to create Redis index: %s", e)

    def add(
        self,
        vector: np.ndarray,
        model: str,
        model_prefix: str,
        has_tools: bool,
        response: str,
        tenant_id: str = "",
        ttl: int = 86400,
    ) -> Optional[str]:
        """Add a vector to the index with metadata."""
        if self._redis is None:
            return None

        doc_id = f"{DOC_PREFIX}{int(time.time() * 1000000)}"
        vec_bytes = vector.astype(np.float32).tobytes()

        try:
            mapping = {
                "embedding": vec_bytes,
                "model": model,
                "model_prefix": model_prefix,
                "has_tools": str(has_tools).lower(),
                "tenant_id": tenant_id or "default",
                "response": response,
                "timestamp": str(int(time.time())),
            }
            self._redis.hset(doc_id, mapping=mapping)
            if ttl > 0:
                self._redis.expire(doc_id, ttl)
            return doc_id
        except Exception as e:
            logger.warning("RedisVectorStore add failed: %s", e)
            return None

    def search(
        self,
        query_vector: np.ndarray,
        model_prefix: str,
        has_tools: bool = False,
        tenant_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors with TAG pre-filtering."""
        if self._redis is None:
            return []

        try:
            from redis.commands.search.query import Query

            vec_bytes = query_vector.astype(np.float32).tobytes()

            # Build filter string
            # TAG values with hyphens/dots MUST be escaped with double backslashes
            escaped_prefix = model_prefix.replace("-", "\\-").replace(".", "\\.")
            filter_parts = [f"@model_prefix:{{{escaped_prefix}}}"]
            tools_str = "true" if has_tools else "false"
            filter_parts.append(f"@has_tools:{{{tools_str}}}")
            if tenant_id:
                filter_parts.append(f"@tenant_id:{{{tenant_id}}}")

            filter_str = " ".join(filter_parts)

            q = (
                Query(f"({filter_str})=>[KNN {top_k} @embedding $vec AS distance]")
                .sort_by("distance")
                .return_fields("model", "model_prefix", "has_tools", "response",
                               "tenant_id", "distance")
                .paging(0, top_k)
                .dialect(2)
            )

            results = self._redis.ft(INDEX_NAME).search(
                q, query_params={"vec": vec_bytes}
            )

            return [
                VectorSearchResult(
                    doc_id=doc.id,
                    distance=float(doc.distance),
                    model=doc.model,
                    model_prefix=doc.model_prefix,
                    has_tools=doc.has_tools == "true",
                    response=doc.response,
                    tenant_id=getattr(doc, "tenant_id", ""),
                )
                for doc in results.docs
            ]
        except Exception as e:
            logger.warning("RedisVectorStore search failed: %s", e)
            return []

    @property
    def available(self) -> bool:
        """Check if Redis vector store is available."""
        return self._redis is not None and self._index_created

    def count(self) -> int:
        """Get number of vectors in the index."""
        if not self.available:
            return 0
        try:
            info = self._redis.ft(INDEX_NAME).info()
            return int(info.get("num_docs", 0))
        except Exception:
            return 0
