"""Tests for StreamUsageTracker."""

from types import SimpleNamespace
from agentfuse.providers.stream_usage import StreamUsageTracker


def test_openai_final_chunk():
    """OpenAI final usage chunk must be extracted."""
    tracker = StreamUsageTracker(provider="openai")
    chunk = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=100, total_tokens=150),
    )
    result = tracker.process_chunk(chunk)
    assert result is not None
    assert result.input_tokens == 50
    assert result.output_tokens == 100
    assert result.finalized is True


def test_openai_content_chunk_no_usage():
    """OpenAI content chunks must not produce usage."""
    tracker = StreamUsageTracker(provider="openai")
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))],
        usage=None,
    )
    result = tracker.process_chunk(chunk)
    assert result is None


def test_anthropic_start_and_delta():
    """Anthropic: input from start, output from delta (cumulative)."""
    tracker = StreamUsageTracker(provider="anthropic")

    start = SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(usage=SimpleNamespace(input_tokens=200)),
    )
    tracker.process_chunk(start)

    delta = SimpleNamespace(
        type="message_delta",
        usage=SimpleNamespace(output_tokens=50),
    )
    result = tracker.process_chunk(delta)
    assert result is not None
    assert result.input_tokens == 200
    assert result.output_tokens == 50


def test_gemini_cumulative():
    """Gemini: every chunk has cumulative usage."""
    tracker = StreamUsageTracker(provider="gemini")

    chunk1 = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=100, candidates_token_count=10,
        cached_content_token_count=0, thoughts_token_count=0,
        total_token_count=110,
    ))
    tracker.process_chunk(chunk1)

    chunk2 = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=100, candidates_token_count=50,
        cached_content_token_count=0, thoughts_token_count=0,
        total_token_count=150,
    ))
    result = tracker.process_chunk(chunk2)
    assert result.output_tokens == 50  # cumulative final value


def test_get_final_usage():
    """get_final_usage must return the accumulated data."""
    tracker = StreamUsageTracker(provider="openai", model="gpt-4o")
    usage = tracker.get_final_usage()
    assert usage.model == "gpt-4o"
    assert usage.provider == "openai"
