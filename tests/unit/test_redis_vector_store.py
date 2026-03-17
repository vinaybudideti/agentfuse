"""
Tests for RedisVectorStore (without Redis running — tests graceful fallback).
"""

from agentfuse.storage.redis_vector_store import RedisVectorStore, VectorSearchResult
import numpy as np


def test_unavailable_without_redis():
    """Without Redis, store must gracefully degrade."""
    store = RedisVectorStore(redis_url="redis://localhost:99999")
    assert store.available is False


def test_add_returns_none_when_unavailable():
    """add() without Redis must return None."""
    store = RedisVectorStore(redis_url="redis://localhost:99999")
    vec = np.random.rand(768).astype(np.float32)
    result = store.add(vec, model="gpt-4o", model_prefix="openai",
                       has_tools=False, response="test")
    assert result is None


def test_search_returns_empty_when_unavailable():
    """search() without Redis must return empty list."""
    store = RedisVectorStore(redis_url="redis://localhost:99999")
    vec = np.random.rand(768).astype(np.float32)
    results = store.search(vec, model_prefix="openai")
    assert results == []


def test_count_returns_zero_when_unavailable():
    """count() without Redis must return 0."""
    store = RedisVectorStore(redis_url="redis://localhost:99999")
    assert store.count() == 0


def test_vector_search_result_similarity():
    """Similarity must be 1 - distance."""
    result = VectorSearchResult(
        doc_id="test", distance=0.1, model="gpt-4o",
        model_prefix="openai", has_tools=False, response="test"
    )
    assert abs(result.similarity - 0.9) < 0.01


def test_vector_search_result_zero_distance():
    """Zero distance must give 1.0 similarity."""
    result = VectorSearchResult(
        doc_id="test", distance=0.0, model="gpt-4o",
        model_prefix="openai", has_tools=False, response="test"
    )
    assert result.similarity == 1.0


def test_custom_dimensions():
    """Store must accept custom vector dimensions."""
    store = RedisVectorStore(redis_url="redis://localhost:99999", dim=256)
    assert store._dim == 256
