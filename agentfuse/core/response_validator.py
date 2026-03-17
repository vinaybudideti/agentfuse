"""
ResponseValidator — validates LLM responses before caching and delivery.

Catches malformed, truncated, or suspicious responses that would corrupt
the cache or cause downstream failures. This prevents:
1. Empty/whitespace responses entering cache
2. Truncated responses (finish_reason != "stop") being cached as complete
3. Responses that are just error messages from the LLM being cached
4. Extremely short responses for complex queries (likely hallucination)

Production systems need this because LLMs occasionally return garbage,
and caching garbage means serving garbage forever.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of response validation."""
    valid: bool
    reason: str = ""
    should_cache: bool = True
    should_retry: bool = False


# Common LLM error patterns that should NOT be cached
_ERROR_PATTERNS = [
    r"i('m| am) (sorry|unable|not able)",
    r"as an ai( language model)?",
    r"i (cannot|can't) (help|assist|provide)",
    r"error:\s",
    r"rate limit",
    r"context (length|window) exceeded",
    r"maximum.*tokens",
]

_COMPILED_ERROR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _ERROR_PATTERNS]


def validate_response(
    response_text: str,
    model: str = "",
    finish_reason: Optional[str] = None,
    min_length: int = 1,
    max_error_ratio: float = 0.5,
) -> ValidationResult:
    """
    Validate an LLM response for caching and delivery.

    Args:
        response_text: The text content of the response
        model: Model name (for model-specific validation)
        finish_reason: "stop", "length", "tool_calls", etc.
        min_length: Minimum response length in characters
        max_error_ratio: Max fraction of response matching error patterns

    Returns:
        ValidationResult with valid flag and reason
    """
    if not response_text or not response_text.strip():
        return ValidationResult(
            valid=False, reason="empty response",
            should_cache=False, should_retry=True,
        )

    text = response_text.strip()

    # Check minimum length
    if len(text) < min_length:
        return ValidationResult(
            valid=True, reason="below min length",
            should_cache=False,  # too short to cache
        )

    # Check for truncation/incomplete responses
    # OpenAI: "length" (token limit), "content_filter" (safety filter)
    # Anthropic: "max_tokens", "pause_turn" (long-running), "refusal" (classifier)
    # Gemini: MAX_TOKENS, SAFETY, RECITATION
    NEVER_CACHE_FINISH_REASONS = {
        "length", "content_filter",           # OpenAI
        "max_tokens", "pause_turn", "refusal",  # Anthropic
        "MAX_TOKENS", "SAFETY", "RECITATION",   # Gemini
    }
    if finish_reason in NEVER_CACHE_FINISH_REASONS:
        return ValidationResult(
            valid=True, reason=f"not cacheable (finish_reason={finish_reason})",
            should_cache=False,
        )

    # Check for LLM refusal/error patterns
    error_matches = sum(1 for p in _COMPILED_ERROR_PATTERNS if p.search(text[:500]))
    if error_matches >= 2:  # multiple error patterns = likely refusal
        return ValidationResult(
            valid=True, reason="likely LLM refusal/error",
            should_cache=False,  # don't cache error responses
        )

    return ValidationResult(valid=True, should_cache=True)


def validate_for_cache(response_text: str, finish_reason: Optional[str] = None) -> bool:
    """Simple check: should this response be cached? Returns True/False."""
    result = validate_response(response_text, finish_reason=finish_reason)
    return result.should_cache
