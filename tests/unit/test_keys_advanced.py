"""
Loop 10 — Advanced cache key tests.
"""

from agentfuse.core.keys import (
    _extract_text,
    build_l1_cache_key,
    extract_semantic_content,
    build_l2_metadata_filter,
    build_cache_key,
)


def test_extract_text_from_none():
    assert _extract_text(None) == ""


def test_extract_text_from_string():
    assert _extract_text("hello") == "hello"


def test_extract_text_from_list_of_dicts():
    blocks = [
        {"type": "text", "text": "Hello"},
        {"type": "image", "source": {}},
        {"type": "text", "text": "World"},
    ]
    result = _extract_text(blocks)
    assert "Hello" in result
    assert "World" in result


def test_extract_text_from_list_of_strings():
    result = _extract_text(["hello", "world"])
    assert "hello" in result
    assert "world" in result


def test_extract_text_from_mixed_list():
    blocks = [
        {"text": "dict text"},
        "raw string",
    ]
    result = _extract_text(blocks)
    assert "dict text" in result
    assert "raw string" in result


def test_extract_text_from_other_type():
    result = _extract_text(42)
    assert result == "42"


def test_build_cache_key_string_messages():
    """build_cache_key handles string messages."""
    key = build_cache_key(["hello"], "gpt-4o")
    assert "model=gpt-4o" in key
    assert "[user]: hello" in key


def test_l2_metadata_unknown_provider():
    """Unknown model prefix should derive from model name."""
    meta = build_l2_metadata_filter("custom-provider/model-v1")
    assert meta["model_prefix"] == "custom-provider"


def test_l2_metadata_deepseek():
    """DeepSeek models — prefix should be from before the /."""
    meta = build_l2_metadata_filter("deepseek/deepseek-chat")
    assert meta["model_prefix"] == "deepseek"


def test_l1_key_with_all_optional_params():
    """Key with all optional params must be different from key without."""
    base_key = build_l1_cache_key("gpt-4o", [{"role": "user", "content": "hi"}])
    full_key = build_l1_cache_key(
        "gpt-4o",
        [{"role": "user", "content": "hi"}],
        temperature=0.5,
        tools=[{"function": {"name": "search"}}],
        model_version="2026-03",
        tenant_id="org_123",
        response_format={"type": "json_object"},
    )
    assert base_key != full_key


def test_semantic_content_with_multimodal():
    """extract_semantic_content handles list content blocks."""
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this"},
            {"type": "image", "source": {}}
        ]},
    ]
    result = extract_semantic_content(msgs)
    assert "Describe this" in result


def test_l2_metadata_mistral():
    """Mistral models must have 'mistral' prefix."""
    meta = build_l2_metadata_filter("mistral-large-latest")
    assert meta["model_prefix"] == "mistral"


def test_l2_metadata_grok():
    """Grok models must have 'xai' prefix."""
    meta = build_l2_metadata_filter("grok-4.1-fast")
    assert meta["model_prefix"] == "xai"


def test_l2_metadata_llama():
    """Llama models must have 'meta' prefix."""
    meta = build_l2_metadata_filter("llama-3.3-70b")
    assert meta["model_prefix"] == "meta"
