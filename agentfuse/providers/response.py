"""
NormalizedUsage — unified token usage extraction across providers.

The biggest gotcha is Anthropic: its `input_tokens` field only counts uncached
tokens, unlike OpenAI where `prompt_tokens` is the total. This has caused
double-counting bugs across the industry.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NormalizedUsage:
    """Unified usage across all providers."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    provider: str = "unknown"

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def __repr__(self) -> str:
        return (f"NormalizedUsage(input={self.total_input_tokens}, output={self.total_output_tokens}, "
                f"cached={self.cached_input_tokens}, provider={self.provider!r})")


def extract_usage(provider: str, usage_obj) -> NormalizedUsage:
    """
    Normalize usage from any provider's response object.

    Provider-specific handling:
    - OpenAI: prompt_tokens includes cached; completion_tokens includes reasoning
    - Anthropic: input_tokens EXCLUDES cached — must add cache_read + cache_creation
    - Gemini: thoughts_token_count billed as output
    - Unknown: best-effort attribute lookup
    """
    if usage_obj is None:
        return NormalizedUsage(provider=provider)

    if provider == "openai":
        return _extract_openai(usage_obj)
    elif provider == "anthropic":
        return _extract_anthropic(usage_obj)
    elif provider in ("gemini", "google", "gcp.gemini"):
        return _extract_gemini(usage_obj)
    else:
        return _extract_unknown(usage_obj, provider)


def _extract_openai(usage) -> NormalizedUsage:
    """
    OpenAI: prompt_tokens is total (includes cached).
    completion_tokens already includes reasoning_tokens — do NOT add them.
    """
    prompt = _getattr_int(usage, "prompt_tokens")
    completion = _getattr_int(usage, "completion_tokens")

    # Cached tokens info (if available)
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        cached = _getattr_int(details, "cached_tokens")

    # Reasoning tokens (already included in completion_tokens)
    reasoning = 0
    comp_details = getattr(usage, "completion_tokens_details", None)
    if comp_details:
        reasoning = _getattr_int(comp_details, "reasoning_tokens")

    return NormalizedUsage(
        total_input_tokens=prompt,
        total_output_tokens=completion,
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
        provider="openai",
    )


def _extract_anthropic(usage) -> NormalizedUsage:
    """
    Anthropic: input_tokens EXCLUDES cached tokens.

    Anthropic billing:
    - input_tokens: uncached input (billed at full rate)
    - cache_creation_input_tokens: first-time cache write (billed at 1.25x for 5-min TTL)
    - cache_read_input_tokens: subsequent cache reads (billed at 0.1x input rate)
    - cache_creation.ephemeral_5m_input_tokens: 5-min TTL cache writes (1.25x)
    - cache_creation.ephemeral_1h_input_tokens: 1-hour TTL cache writes (2.0x)

    For TOTAL input tokens (what was actually processed), sum all three.
    For COST calculation, each component has its own rate — handled by pricing engine.
    """
    input_tokens = _getattr_int(usage, "input_tokens")
    output_tokens = _getattr_int(usage, "output_tokens")
    cache_read = _getattr_int(usage, "cache_read_input_tokens")
    cache_write = _getattr_int(usage, "cache_creation_input_tokens")

    # Handle new cache_creation sub-object with TTL breakdowns (March 2026 API)
    # If the cache_creation sub-object exists, it provides finer-grained TTL detail.
    # cache_creation_input_tokens is the legacy flat field; the sub-object breaks it down.
    cache_creation_obj = getattr(usage, "cache_creation", None)
    if cache_creation_obj is not None and cache_creation_obj is not False:
        ephemeral_5m = _getattr_int(cache_creation_obj, "ephemeral_5m_input_tokens")
        ephemeral_1h = _getattr_int(cache_creation_obj, "ephemeral_1h_input_tokens")
        # If sub-object has data but the flat field is 0, use sub-object sum
        if (ephemeral_5m + ephemeral_1h) > 0 and cache_write == 0:
            cache_write = ephemeral_5m + ephemeral_1h

    # Total input = uncached + cache_read + cache_write
    # This is correct per Anthropic docs: input_tokens only counts NEW uncached tokens.
    # cache_read and cache_write are separate billing line items but still input tokens.
    total_input = input_tokens + cache_read + cache_write

    return NormalizedUsage(
        total_input_tokens=total_input,
        total_output_tokens=output_tokens,
        cached_input_tokens=cache_read,
        cache_write_tokens=cache_write,
        provider="anthropic",
    )


def _extract_gemini(usage) -> NormalizedUsage:
    """
    Gemini: thoughts_token_count is billed as output.
    Add to candidates_token_count for total output.
    """
    input_tokens = _getattr_int(usage, "prompt_token_count",
                                _getattr_int(usage, "promptTokenCount"))
    output_tokens = _getattr_int(usage, "candidates_token_count",
                                 _getattr_int(usage, "candidatesTokenCount"))
    thoughts = _getattr_int(usage, "thoughts_token_count",
                            _getattr_int(usage, "thoughtsTokenCount"))

    cached = _getattr_int(usage, "cached_content_token_count",
                          _getattr_int(usage, "cachedContentTokenCount"))

    # Tool use prompt tokens (tokens from tool execution results)
    tool_tokens = _getattr_int(usage, "tool_use_prompt_token_count",
                               _getattr_int(usage, "toolUsePromptTokenCount"))
    # Tool tokens are part of input but not always in prompt_token_count
    if tool_tokens > 0:
        input_tokens += tool_tokens

    return NormalizedUsage(
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens + thoughts,
        cached_input_tokens=cached,
        reasoning_tokens=thoughts,
        provider="gemini",
    )


def _extract_unknown(usage, provider: str) -> NormalizedUsage:
    """Best-effort extraction for unknown providers."""
    input_tokens = (
        _getattr_int(usage, "prompt_tokens")
        or _getattr_int(usage, "input_tokens")
        or _getattr_int(usage, "prompt_token_count")
    )
    output_tokens = (
        _getattr_int(usage, "completion_tokens")
        or _getattr_int(usage, "output_tokens")
        or _getattr_int(usage, "candidates_token_count")
    )

    return NormalizedUsage(
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        provider=provider,
    )


def _getattr_int(obj, attr: str, default: int = 0) -> int:
    """Safely get an integer attribute, returning default if missing or None."""
    val = getattr(obj, attr, None)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default
