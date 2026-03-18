"""
StreamUsageTracker — unified streaming usage extraction across providers.

From research file 4 (Block 16): Each provider reports streaming usage differently:
- OpenAI: opt-in via stream_options, final chunk only (choices=[])
- Anthropic: input_tokens in message_start, output_tokens cumulative in message_delta
- Gemini: every chunk has usageMetadata with cumulative counts

This module provides a unified pattern to extract usage from any provider's stream.

Usage:
    tracker = StreamUsageTracker(provider="openai")
    for chunk in stream:
        tracker.process_chunk(chunk)
    usage = tracker.get_final_usage()
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StreamUsage:
    """Unified streaming usage data."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    model: str = ""
    provider: str = ""
    finalized: bool = False


class StreamUsageTracker:
    """
    Extracts usage from streaming responses across providers.

    Handles the differences between OpenAI, Anthropic, and Gemini streaming.
    """

    def __init__(self, provider: str = "openai", model: str = ""):
        self._provider = provider
        self._model = model
        self._usage = StreamUsage(model=model, provider=provider)

    def process_chunk(self, chunk) -> Optional[StreamUsage]:
        """Process a streaming chunk. Returns StreamUsage when finalized."""
        if self._provider == "openai":
            return self._process_openai(chunk)
        elif self._provider == "anthropic":
            return self._process_anthropic(chunk)
        elif self._provider in ("gemini", "google"):
            return self._process_gemini(chunk)
        return None

    def _process_openai(self, chunk) -> Optional[StreamUsage]:
        """OpenAI: usage only in final chunk (choices=[], usage populated)."""
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._usage.input_tokens = getattr(usage, "prompt_tokens", 0)
            self._usage.output_tokens = getattr(usage, "completion_tokens", 0)
            self._usage.total_tokens = getattr(usage, "total_tokens", 0)

            details = getattr(usage, "prompt_tokens_details", None)
            if details:
                self._usage.cached_tokens = getattr(details, "cached_tokens", 0)

            comp_details = getattr(usage, "completion_tokens_details", None)
            if comp_details:
                self._usage.reasoning_tokens = getattr(comp_details, "reasoning_tokens", 0)

            self._usage.finalized = True
            return self._usage
        return None

    def _process_anthropic(self, chunk) -> Optional[StreamUsage]:
        """Anthropic: input from message_start, output cumulative in message_delta."""
        chunk_type = getattr(chunk, "type", "")

        if chunk_type == "message_start":
            msg = getattr(chunk, "message", None)
            if msg:
                usage = getattr(msg, "usage", None)
                if usage:
                    self._usage.input_tokens = getattr(usage, "input_tokens", 0)

        elif chunk_type == "message_delta":
            usage = getattr(chunk, "usage", None)
            if usage:
                # output_tokens is CUMULATIVE (total), not incremental
                self._usage.output_tokens = getattr(usage, "output_tokens", 0)
                self._usage.total_tokens = self._usage.input_tokens + self._usage.output_tokens
                self._usage.finalized = True
                return self._usage

        return None

    def _process_gemini(self, chunk) -> Optional[StreamUsage]:
        """Gemini: every chunk has usageMetadata with cumulative counts."""
        usage = getattr(chunk, "usage_metadata", None)
        if usage:
            self._usage.input_tokens = getattr(usage, "prompt_token_count", 0)
            self._usage.output_tokens = getattr(usage, "candidates_token_count", 0)
            self._usage.cached_tokens = getattr(usage, "cached_content_token_count", 0)
            self._usage.reasoning_tokens = getattr(usage, "thoughts_token_count", 0)
            self._usage.total_tokens = getattr(usage, "total_token_count", 0)
            # Last chunk has final totals — keep overwriting
            self._usage.finalized = True
            return self._usage
        return None

    def get_final_usage(self) -> StreamUsage:
        """Get the final usage data."""
        return self._usage
