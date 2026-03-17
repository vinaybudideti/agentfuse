"""
Tests for PromptCompressor — intelligent context compression.
"""

from agentfuse.core.prompt_compressor import PromptCompressor


def _make_messages(n, system=True):
    """Create a list of messages for testing."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": "You are a helpful assistant."})
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message number {i} with some content here."})
    return msgs


def test_smart_compress_removes_low_info():
    """Smart compression must remove greetings and acknowledgments."""
    compressor = PromptCompressor()
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "Sure, let me help with that detailed analysis."},
        {"role": "user", "content": "Explain quantum computing in detail"},
    ]
    compressed = compressor.compress(msgs, strategy="smart")
    # "hi" and "ok" should be removed
    assert len(compressed) < len(msgs)
    # System and substantive messages should remain
    assert compressed[0]["role"] == "system"
    assert any("quantum" in m.get("content", "") for m in compressed)


def test_truncate_keeps_system_and_recent():
    """Truncate must keep system + last 6 non-system messages."""
    compressor = PromptCompressor()
    msgs = _make_messages(20)
    compressed = compressor.compress(msgs, strategy="truncate")
    assert compressed[0]["role"] == "system"
    assert len(compressed) <= 7  # system + 6


def test_priority_keeps_high_info():
    """Priority compression must keep high-information messages."""
    compressor = PromptCompressor()
    msgs = [
        {"role": "system", "content": "You are an expert."},
        {"role": "user", "content": "ok"},  # low info
        {"role": "assistant", "content": "Thanks!"},  # low info
        {"role": "user", "content": "Explain the architecture of microservices in detail with examples"},
        {"role": "assistant", "content": "Microservices architecture involves..." + "x" * 200},
    ]
    compressed = compressor.compress(msgs, strategy="priority", target_tokens=200)
    # Should keep system + high-info messages
    assert compressed[0]["role"] == "system"
    assert any("microservices" in m.get("content", "").lower() for m in compressed)


def test_removes_consecutive_duplicates():
    """Consecutive identical messages must be removed."""
    compressor = PromptCompressor()
    msgs = [
        {"role": "user", "content": "What is the meaning of life in philosophy?"},
        {"role": "user", "content": "What is the meaning of life in philosophy?"},  # dup
        {"role": "user", "content": "What is the meaning of life in philosophy?"},  # dup
        {"role": "assistant", "content": "The meaning of life has been debated for centuries."},
    ]
    compressed = compressor.compress(msgs, strategy="smart")
    user_msgs = [m for m in compressed if m["role"] == "user"]
    assert len(user_msgs) == 1  # duplicates removed


def test_empty_messages():
    """Empty message list must return empty."""
    compressor = PromptCompressor()
    assert compressor.compress([]) == []


def test_compression_report():
    """Compression report must contain expected keys."""
    compressor = PromptCompressor()
    original = _make_messages(20)
    compressed = compressor.compress(original, strategy="truncate")
    report = compressor.get_compression_report(original, compressed)
    assert report["original_messages"] == 21  # 1 system + 20
    assert report["compressed_messages"] < report["original_messages"]
    assert report["tokens_saved"] > 0
    assert 0 < report["compression_ratio"] < 1


def test_low_info_detection():
    """Low-info patterns must be correctly detected."""
    compressor = PromptCompressor()
    assert compressor._is_low_info("ok") is True
    assert compressor._is_low_info("thanks!") is True
    assert compressor._is_low_info("hi") is True
    assert compressor._is_low_info("Explain quantum mechanics") is False
    assert compressor._is_low_info("yes") is True
    assert compressor._is_low_info("got it") is True


def test_info_score_long_content_high():
    """Long content must score higher than short content."""
    compressor = PromptCompressor()
    short_msg = {"role": "user", "content": "hi"}
    long_msg = {"role": "user", "content": "Explain the architecture of " + "x" * 300}
    assert compressor._info_score(long_msg) > compressor._info_score(short_msg)


def test_system_always_kept():
    """System messages must never be removed in any strategy."""
    compressor = PromptCompressor()
    for strategy in ["smart", "truncate", "priority"]:
        msgs = _make_messages(10)
        compressed = compressor.compress(msgs, strategy=strategy, target_tokens=200)
        assert compressed[0]["role"] == "system"
