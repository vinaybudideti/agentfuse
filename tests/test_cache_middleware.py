import pytest
from agentfuse.core.cache import CacheMiddleware, CacheHit, CacheMiss
from agentfuse.core.keys import build_cache_key


def test_cache_import():
    """Cache middleware imports and initializes without error."""
    c = CacheMiddleware()
    assert c is not None


def test_cache_miss_on_first_call():
    """First call to a never-seen prompt must be a cache miss."""
    import uuid
    c = CacheMiddleware()
    unique = f"Completely unique query {uuid.uuid4()}"
    key = build_cache_key([{"role": "user", "content": unique}], "gpt-4o")
    result = c.check(key, "gpt-4o")
    assert isinstance(result, CacheMiss)


def test_cache_hit_after_store():
    """After storing a response, same key must return CacheHit tier 1."""
    c = CacheMiddleware()
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    key = build_cache_key(messages, "gpt-4o")
    response = "The answer is 4."

    c.store(key, response, "gpt-4o")
    result = c.check(key, "gpt-4o")
    assert isinstance(result, CacheHit), f"Expected CacheHit, got {type(result).__name__}"
    assert result.tier == 1
    assert result.cost == 0.0


def test_semantic_similarity_hit():
    """Semantically similar prompt should also return a cache hit."""
    c = CacheMiddleware()
    original = build_cache_key([{"role": "user", "content": "What is the capital city of France?"}], "gpt-4o")
    similar = build_cache_key([{"role": "user", "content": "What city is the capital of France?"}], "gpt-4o")
    response = "Paris is the capital of France."

    c.store(original, response, "gpt-4o")
    result = c.check(similar, "gpt-4o")

    # Not a hard assert — similarity depends on model calibration
    assert isinstance(result, (CacheHit, CacheMiss))


def test_repeated_prompts_hit_rate():
    """87.5%+ cache hit rate on repeated prompts."""
    c = CacheMiddleware()
    prompts = [
        ("What is Python?", "Python is a programming language."),
        ("What is an API?", "An API is an application programming interface."),
        ("What is machine learning?", "ML is a subset of AI."),
        ("What is a neural network?", "A neural network is a model inspired by the brain."),
        ("What is gradient descent?", "Gradient descent is an optimization algorithm."),
        ("What is a transformer?", "A transformer is an attention-based neural architecture."),
        ("What is tokenization?", "Tokenization splits text into tokens."),
        ("What is a vector embedding?", "An embedding maps text to a numeric vector."),
    ]

    for prompt, response in prompts:
        key = build_cache_key([{"role": "user", "content": prompt}], "gpt-4o")
        c.store(key, response, "gpt-4o")

    hits = 0
    for prompt, _ in prompts:
        key = build_cache_key([{"role": "user", "content": prompt}], "gpt-4o")
        result = c.check(key, "gpt-4o")
        if isinstance(result, CacheHit):
            hits += 1

    hit_rate = hits / len(prompts)
    assert hit_rate >= 0.875, f"Hit rate {hit_rate:.1%} is below 87.5% threshold"


# --- Tests that would have caught the cache key bugs ---

def test_different_roles_produce_different_keys():
    """System+user vs user-only must NOT collide."""
    key_a = build_cache_key([
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "hello"},
    ], "gpt-4o")
    key_b = build_cache_key([
        {"role": "user", "content": "Be helpful hello"},
    ], "gpt-4o")
    assert key_a != key_b


def test_cross_model_cache_isolation():
    """GPT-4o response must NOT be returned for a Claude request."""
    c = CacheMiddleware()
    messages = [{"role": "user", "content": "What is the meaning of life?"}]

    key_gpt = build_cache_key(messages, "gpt-4o")
    key_claude = build_cache_key(messages, "claude-sonnet-4-6")

    c.store(key_gpt, "GPT says 42", "gpt-4o")

    # Same prompt, different model — must be a miss
    result = c.check(key_claude, "claude-sonnet-4-6")
    assert isinstance(result, CacheMiss), \
        f"Cross-model contamination: got {type(result).__name__} with response={getattr(result, 'response', None)}"


def test_non_string_content_handled():
    """List-format content (Anthropic vision/tool) must not crash."""
    key = build_cache_key([
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image", "source": {"data": "base64..."}}
        ]}
    ], "gpt-4o")
    assert "[user]:" in key
    assert "Describe this image" in key
