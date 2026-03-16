"""
PromptCachingMiddleware — Anthropic cache_control marker auto-injection.

Detects static system messages above MIN_CACHEABLE_TOKENS and injects
Anthropic's cache_control: {"type": "ephemeral"} markers to enable
server-side prompt caching. Only applies to Claude models.
"""

import re
import copy


class PromptCachingMiddleware:

    MIN_CACHEABLE_TOKENS = 1024  # Anthropic minimum for Sonnet

    DYNAMIC_PATTERNS = [
        r"\d{4}-\d{2}-\d{2}",          # Dates
        r"session_id[=:]\s*\S+",        # Session IDs
        r"timestamp[=:]\s*\d+",         # Timestamps
        r"request_id[=:]\s*\S+",        # Request IDs
    ]

    def inject(self, messages: list, model: str) -> list:
        """
        Injects Anthropic cache_control markers into eligible messages.
        Only applies to claude models. Returns messages unchanged for others.
        """
        if not model.startswith("claude"):
            return messages

        from agentfuse.providers.tokenizer import TokenCounterAdapter
        tokenizer = TokenCounterAdapter()

        result = []
        breakpoints_used = 0

        for msg in messages:
            if (msg.get("role") == "system" and
                    self._is_static(msg.get("content", ""))):
                token_count = tokenizer.count_tokens(msg["content"], model)
                if (token_count >= self.MIN_CACHEABLE_TOKENS and
                        breakpoints_used < 4):
                    msg = self._add_cache_control(msg)
                    breakpoints_used += 1
            result.append(msg)

        return result

    def _is_static(self, content: str) -> bool:
        """Returns True if content has no dynamic patterns."""
        return not any(
            re.search(p, content)
            for p in self.DYNAMIC_PATTERNS
        )

    def _add_cache_control(self, msg: dict) -> dict:
        """Adds Anthropic cache_control marker to a message."""
        msg = copy.deepcopy(msg)
        if isinstance(msg.get("content"), str):
            msg["content"] = [
                {
                    "type": "text",
                    "text": msg["content"],
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return msg
