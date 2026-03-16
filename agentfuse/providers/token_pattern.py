"""
TokenPatternAdapter — automatically discovers how each LLM reports usage.

Different LLMs report token usage in wildly different ways:
- OpenAI: prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens
- Anthropic: input_tokens (EXCLUDES cached), cache_read_input_tokens, output_tokens
- Gemini: promptTokenCount, candidatesTokenCount, thoughtsTokenCount
- DeepSeek: prompt_tokens + prompt_cache_hit_tokens + prompt_cache_miss_tokens
- Custom/local: completely unpredictable field names

This module solves the problem by:
1. Probing a usage object to discover its field names
2. Mapping discovered fields to our normalized schema
3. Caching the mapping per provider for future calls
4. Falling back gracefully when fields are missing

This is a novel approach — no existing tool auto-discovers token patterns.
"""

import logging
from typing import Optional
from agentfuse.providers.response import NormalizedUsage

logger = logging.getLogger(__name__)

# Known field mappings per provider (cached after first discovery)
_discovered_patterns: dict[str, dict] = {}

# Standard field name variations we look for
_INPUT_FIELDS = [
    "prompt_tokens", "input_tokens", "prompt_token_count",
    "promptTokenCount", "total_input_tokens",
]
_OUTPUT_FIELDS = [
    "completion_tokens", "output_tokens", "candidates_token_count",
    "candidatesTokenCount", "total_output_tokens",
]
_CACHED_INPUT_FIELDS = [
    "cached_tokens", "cache_read_input_tokens",
    "cached_content_token_count", "cachedContentTokenCount",
    "prompt_cache_hit_tokens",
]
_CACHE_WRITE_FIELDS = [
    "cache_creation_input_tokens", "cache_write_tokens",
    "prompt_cache_miss_tokens",
]
_REASONING_FIELDS = [
    "reasoning_tokens", "thoughts_token_count", "thoughtsTokenCount",
]


def discover_usage_pattern(usage_obj, provider: str = "unknown") -> dict:
    """
    Probe a usage object to discover which fields it has.
    Returns a mapping: {standard_name: actual_field_name}.
    """
    if usage_obj is None:
        return {}

    pattern = {}

    # Find input token field
    for field in _INPUT_FIELDS:
        val = _safe_getattr(usage_obj, field)
        if val is not None:
            pattern["input"] = field
            break

    # Find output token field
    for field in _OUTPUT_FIELDS:
        val = _safe_getattr(usage_obj, field)
        if val is not None:
            pattern["output"] = field
            break

    # Find cached input field (may be nested)
    for field in _CACHED_INPUT_FIELDS:
        val = _safe_getattr(usage_obj, field)
        if val is not None:
            pattern["cached_input"] = field
            break
    # Check nested: prompt_tokens_details.cached_tokens (OpenAI)
    details = _safe_getattr(usage_obj, "prompt_tokens_details")
    if details and _safe_getattr(details, "cached_tokens") is not None:
        pattern["cached_input_nested"] = "prompt_tokens_details.cached_tokens"

    # Find cache write field
    for field in _CACHE_WRITE_FIELDS:
        val = _safe_getattr(usage_obj, field)
        if val is not None:
            pattern["cache_write"] = field
            break

    # Find reasoning field
    for field in _REASONING_FIELDS:
        val = _safe_getattr(usage_obj, field)
        if val is not None:
            pattern["reasoning"] = field
            break
    # Check nested: completion_tokens_details.reasoning_tokens (OpenAI)
    comp_details = _safe_getattr(usage_obj, "completion_tokens_details")
    if comp_details and _safe_getattr(comp_details, "reasoning_tokens") is not None:
        pattern["reasoning_nested"] = "completion_tokens_details.reasoning_tokens"

    return pattern


def extract_with_pattern(usage_obj, provider: str = "unknown") -> NormalizedUsage:
    """
    Extract usage using auto-discovered pattern.
    Caches the pattern per provider for future calls.
    """
    if usage_obj is None:
        return NormalizedUsage(provider=provider)

    # Check cached pattern
    if provider not in _discovered_patterns:
        _discovered_patterns[provider] = discover_usage_pattern(usage_obj, provider)

    pattern = _discovered_patterns[provider]

    # If we don't have a cached pattern or it's empty, discover fresh
    if not pattern:
        pattern = discover_usage_pattern(usage_obj, provider)
        _discovered_patterns[provider] = pattern

    input_tokens = _get_field(usage_obj, pattern.get("input")) or 0
    output_tokens = _get_field(usage_obj, pattern.get("output")) or 0

    cached_input = 0
    if "cached_input" in pattern:
        cached_input = _get_field(usage_obj, pattern["cached_input"]) or 0
    elif "cached_input_nested" in pattern:
        details = _safe_getattr(usage_obj, "prompt_tokens_details")
        cached_input = _safe_getattr_int(details, "cached_tokens") if details else 0

    cache_write = _get_field(usage_obj, pattern.get("cache_write")) or 0

    reasoning = 0
    if "reasoning" in pattern:
        reasoning = _get_field(usage_obj, pattern["reasoning"]) or 0
    elif "reasoning_nested" in pattern:
        details = _safe_getattr(usage_obj, "completion_tokens_details")
        reasoning = _safe_getattr_int(details, "reasoning_tokens") if details else 0

    # Anthropic-specific: input_tokens EXCLUDES cached, so add them
    total_input = input_tokens
    if provider == "anthropic":
        total_input = input_tokens + cached_input + cache_write

    # Gemini-specific: thoughts counted as output
    total_output = output_tokens
    if provider in ("gemini", "google") and reasoning > 0:
        total_output = output_tokens + reasoning

    return NormalizedUsage(
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        cached_input_tokens=cached_input,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        provider=provider,
    )


def _safe_getattr(obj, attr: str, default=None):
    """Safely get attribute, handling None objects."""
    if obj is None:
        return default
    return getattr(obj, attr, default)


def _safe_getattr_int(obj, attr: str, default: int = 0) -> int:
    """Safely get integer attribute."""
    val = _safe_getattr(obj, attr)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _get_field(obj, field_name: Optional[str]) -> Optional[int]:
    """Get a field value from object, return None if field not mapped."""
    if not field_name or obj is None:
        return None
    val = getattr(obj, field_name, None)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def reset_patterns():
    """Reset discovered patterns (for testing)."""
    _discovered_patterns.clear()
