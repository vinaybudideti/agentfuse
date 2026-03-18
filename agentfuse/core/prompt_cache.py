"""
PromptCachingMiddleware — Anthropic cache_control marker auto-injection.

Detects static system messages above MIN_CACHEABLE_TOKENS and injects
Anthropic's cache_control: {"type": "ephemeral"} markers to enable
server-side prompt caching. Only applies to Claude models.

FIX: Handles content that is already a list of blocks (doesn't overwrite).
FIX: More comprehensive dynamic pattern detection (UUIDs, hex strings).
FIX: Model-aware minimum cacheable tokens.
"""

import re
import copy


# Anthropic minimum cacheable tokens by model family
# Updated March 2026 — newer models require more tokens for caching
_MIN_TOKENS = {
    "claude-opus-4-6": 4096,     # Opus 4.6 requires 4,096
    "claude-opus-4-5": 4096,     # Opus 4.5 requires 4,096
    "claude-opus": 1024,          # Opus 4.1/4/3 require 1,024
    "claude-sonnet-4-6": 2048,   # Sonnet 4.6 requires 2,048
    "claude-sonnet": 1024,        # Sonnet 4.5/4 require 1,024
    "claude-haiku-4-5": 4096,    # Haiku 4.5 requires 4,096
    "claude-haiku": 2048,         # Haiku 3.5/3 require 2,048
}


class PromptCachingMiddleware:

    MIN_CACHEABLE_TOKENS = 1024  # default, overridden per model

    DYNAMIC_PATTERNS = [
        r"\d{4}-\d{2}-\d{2}",                           # ISO dates
        r"session_id[=:]\s*\S+",                         # Session IDs
        r"timestamp[=:]\s*\d+",                          # Timestamps
        r"request_id[=:]\s*\S+",                         # Request IDs
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",  # UUIDs
        r"nonce[=:]\s*\S+",                              # Nonces
        r"\d{10,13}",                                    # Unix timestamps (10 or 13 digits)
    ]

    def _get_min_tokens(self, model: str) -> int:
        """Get model-specific minimum cacheable tokens."""
        for prefix, min_tok in _MIN_TOKENS.items():
            if model.startswith(prefix):
                return min_tok
        return self.MIN_CACHEABLE_TOKENS

    def inject(self, messages: list, model: str) -> list:
        """
        Injects Anthropic cache_control markers into eligible messages.
        Only applies to claude models. Returns messages unchanged for others.
        """
        if not model.startswith("claude"):
            return messages

        from agentfuse.providers.tokenizer import TokenCounterAdapter
        tokenizer = TokenCounterAdapter()
        min_tokens = self._get_min_tokens(model)

        result = []
        breakpoints_used = 0

        for msg in messages:
            if (msg.get("role") == "system" and
                    self._is_static(self._get_text(msg))):
                token_count = tokenizer.count_tokens(self._get_text(msg), model)
                if (token_count >= min_tokens and
                        breakpoints_used < 4):
                    msg = self._add_cache_control(msg)
                    breakpoints_used += 1
            result.append(msg)

        return result

    def _get_text(self, msg: dict) -> str:
        """Extract text from message content (string or list of blocks)."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return str(content)

    def _is_static(self, content: str) -> bool:
        """Returns True if content has no dynamic patterns."""
        if not content:
            return False
        return not any(
            re.search(p, content)
            for p in self.DYNAMIC_PATTERNS
        )

    def _add_cache_control(self, msg: dict, ttl: str = "5m") -> dict:
        """Adds Anthropic cache_control marker to a message.

        If content is a string, wraps in a text block with cache_control.
        If content is already a list, adds cache_control to the last text block.

        TTL options:
        - "5m" (default): 5-minute ephemeral, 1.25× write cost
        - "1h": 1-hour extended, 2.0× write cost (better for batch workloads)
        """
        msg = copy.deepcopy(msg)
        content = msg.get("content")

        cache_control = {"type": "ephemeral"}
        if ttl == "1h":
            cache_control["ttl"] = "1h"  # Extended 1-hour TTL

        if isinstance(content, str):
            msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": cache_control,
                }
            ]
        elif isinstance(content, list):
            # Find the last text block and add cache_control to it
            for block in reversed(content):
                if isinstance(block, dict) and block.get("type") == "text":
                    block["cache_control"] = cache_control
                    break
        return msg
