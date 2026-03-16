"""
Phase 0 — Cache key contamination tests.

These tests verify that the L1 cache key design prevents cross-model
contamination, tenant leakage, and tool/no-tool confusion.
"""

from agentfuse.core.keys import (
    build_l1_cache_key,
    build_cache_key,
    extract_semantic_content,
    build_l2_metadata_filter,
)


def test_different_models_never_share_keys():
    """CRITICAL: gpt-4o and claude must produce different cache keys for same prompt."""
    key_gpt = build_l1_cache_key("gpt-4o", [{"role": "user", "content": "hello"}])
    key_claude = build_l1_cache_key("claude-sonnet-4-6", [{"role": "user", "content": "hello"}])
    assert key_gpt != key_claude, "CRITICAL: cross-model contamination possible"


def test_same_model_same_content_matches():
    """Same model + same messages must produce identical keys."""
    key1 = build_l1_cache_key("gpt-4o", [{"role": "user", "content": "hello"}])
    key2 = build_l1_cache_key("gpt-4o", [{"role": "user", "content": "hello"}])
    assert key1 == key2


def test_tools_vs_no_tools_different_keys():
    """Requests with tools must not share keys with requests without tools."""
    key_no_tool = build_l1_cache_key("gpt-4o", [{"role": "user", "content": "hi"}])
    key_with_tool = build_l1_cache_key(
        "gpt-4o",
        [{"role": "user", "content": "hi"}],
        tools=[{"function": {"name": "search"}}],
    )
    assert key_no_tool != key_with_tool


def test_tenant_isolation():
    """Different tenants must never share cache keys."""
    key_a = build_l1_cache_key("gpt-4o", [{"role": "user", "content": "hi"}], tenant_id="org_a")
    key_b = build_l1_cache_key("gpt-4o", [{"role": "user", "content": "hi"}], tenant_id="org_b")
    assert key_a != key_b


def test_temperature_affects_key():
    """Different temperatures must produce different keys."""
    key_cold = build_l1_cache_key("gpt-4o", [{"role": "user", "content": "hi"}], temperature=0.0)
    key_hot = build_l1_cache_key("gpt-4o", [{"role": "user", "content": "hi"}], temperature=0.7)
    assert key_cold != key_hot


def test_message_order_affects_key():
    """Different message order must produce different keys."""
    key_a = build_l1_cache_key("gpt-4o", [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ])
    key_b = build_l1_cache_key("gpt-4o", [
        {"role": "user", "content": "second"},
        {"role": "user", "content": "first"},
    ])
    assert key_a != key_b


def test_key_is_deterministic_sha256():
    """Key must be a SHA-256 hash with agentfuse:v2:cache: prefix."""
    key = build_l1_cache_key("gpt-4o", [{"role": "user", "content": "hello"}])
    assert key.startswith("agentfuse:v2:cache:")
    # SHA-256 hex is 64 chars
    digest = key.split(":")[-1]
    assert len(digest) == 64


def test_extract_semantic_content_only_user_messages():
    """extract_semantic_content should only include user messages."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a language"},
        {"role": "user", "content": "Tell me more"},
    ]
    content = extract_semantic_content(messages)
    assert "What is Python?" in content
    assert "Tell me more" in content
    assert "You are helpful" not in content
    assert "Python is a language" not in content


def test_l2_metadata_filter_openai():
    """All OpenAI models should have 'openai' prefix."""
    assert build_l2_metadata_filter("gpt-4o")["model_prefix"] == "openai"
    assert build_l2_metadata_filter("gpt-4o-mini")["model_prefix"] == "openai"
    assert build_l2_metadata_filter("o1")["model_prefix"] == "openai"
    assert build_l2_metadata_filter("o1-mini")["model_prefix"] == "openai"
    assert build_l2_metadata_filter("o3")["model_prefix"] == "openai"
    assert build_l2_metadata_filter("o4-mini")["model_prefix"] == "openai"


def test_l2_metadata_filter_anthropic():
    """Anthropic models should have 'anthropic' prefix."""
    meta = build_l2_metadata_filter("claude-sonnet-4-6")
    assert meta["model_prefix"] == "anthropic"


def test_l2_metadata_filter_gemini():
    """Gemini models should have 'gemini' prefix."""
    meta = build_l2_metadata_filter("gemini-2.5-pro")
    assert meta["model_prefix"] == "gemini"


def test_l2_metadata_filter_tools():
    """Tools flag must be set correctly."""
    meta_no_tools = build_l2_metadata_filter("gpt-4o")
    assert meta_no_tools["has_tools"] is False

    meta_with_tools = build_l2_metadata_filter("gpt-4o", tools=[{"function": {"name": "search"}}])
    assert meta_with_tools["has_tools"] is True


def test_backward_compatible_build_cache_key():
    """build_cache_key (old API) must still work and produce deterministic embeddable keys."""
    messages = [{"role": "user", "content": "hello"}]
    key1 = build_cache_key(messages, "gpt-4o")
    key2 = build_cache_key(messages, "gpt-4o")
    assert key1 == key2
    # build_cache_key returns readable text (not hash) for FAISS embedding
    assert "model=gpt-4o" in key1
    assert "[user]: hello" in key1

    # Different model must produce different key
    key3 = build_cache_key(messages, "claude-sonnet-4-6")
    assert key1 != key3
    assert "model=claude-sonnet-4-6" in key3


def test_response_format_affects_key():
    """Different response formats must produce different keys."""
    key_normal = build_l1_cache_key("gpt-4o", [{"role": "user", "content": "hi"}])
    key_json = build_l1_cache_key(
        "gpt-4o",
        [{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    assert key_normal != key_json
