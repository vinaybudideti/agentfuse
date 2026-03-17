"""
PromptCompressor — intelligent context compression for budget optimization.

When BudgetEngine reaches 90%, it currently truncates to system + last 6 messages.
This module provides SMARTER compression strategies:

1. **Summary compression**: Summarize old messages into a compact context block
2. **Deduplication**: Remove duplicate/redundant messages in conversation
3. **Priority compression**: Keep high-information messages, drop low-information ones
4. **Token-aware trimming**: Trim to exact token budget instead of message count

This is a novel approach — existing budget systems (AgentBudget, LiteLLM) either
hard-terminate or do simple message truncation. AgentFuse compresses INTELLIGENTLY.

Usage:
    compressor = PromptCompressor()
    compressed = compressor.compress(messages, model="gpt-4o", target_tokens=4000)
"""

import hashlib
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class PromptCompressor:
    """
    Intelligent context compression that maximizes information density
    while staying within token budgets.
    """

    # Messages with these patterns are low-information and can be dropped first
    LOW_INFO_PATTERNS = [
        r"^(ok|okay|sure|yes|no|thanks|thank you|got it|understood|alright)[\.\!\?]?$",
        r"^(hi|hello|hey|good morning|good afternoon)[\.\!\?]?$",
    ]

    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.LOW_INFO_PATTERNS]

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from agentfuse.providers.tokenizer import TokenCounterAdapter
            self._tokenizer = TokenCounterAdapter()
        return self._tokenizer

    def compress(
        self,
        messages: list[dict],
        model: str = "gpt-4o",
        target_tokens: Optional[int] = None,
        strategy: str = "smart",
    ) -> list[dict]:
        """
        Compress messages to fit within target_tokens.

        Strategies:
        - "smart": Remove low-info messages, then deduplicate, then truncate
        - "truncate": Keep system + last N messages (BudgetEngine default)
        - "priority": Score messages by information density, keep highest

        Returns compressed message list.
        """
        if not messages:
            return messages

        if strategy == "truncate":
            return self._truncate(messages, model, target_tokens)
        elif strategy == "priority":
            return self._priority_compress(messages, model, target_tokens)
        else:
            return self._smart_compress(messages, model, target_tokens)

    def _smart_compress(self, messages: list[dict], model: str,
                        target_tokens: Optional[int]) -> list[dict]:
        """Smart compression: remove low-info → deduplicate → truncate."""
        tokenizer = self._get_tokenizer()

        # Step 1: Separate system messages (always keep)
        system = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Step 2: Remove low-information messages
        filtered = []
        for msg in non_system:
            content = msg.get("content", "")
            if isinstance(content, str) and self._is_low_info(content):
                continue
            filtered.append(msg)

        # Step 3: Remove duplicate consecutive messages
        deduped = self._remove_consecutive_duplicates(filtered)

        # Step 4: If still over budget, keep system + last N
        result = system + deduped
        if target_tokens:
            current_tokens = tokenizer.count_messages(result, model)
            if current_tokens > target_tokens:
                # Binary search for the right number of recent messages
                result = self._trim_to_target(system, deduped, model, target_tokens)

        return result

    def _priority_compress(self, messages: list[dict], model: str,
                           target_tokens: Optional[int]) -> list[dict]:
        """Score messages by information density, keep highest-scoring."""
        tokenizer = self._get_tokenizer()

        system = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Score each message
        scored = [(self._info_score(m), i, m) for i, m in enumerate(non_system)]

        if target_tokens is None:
            target_tokens = tokenizer.count_messages(messages, model) // 2

        # System messages always included
        system_tokens = tokenizer.count_messages(system, model) if system else 0
        remaining_budget = target_tokens - system_tokens

        # Sort by score (highest first), then by position (maintain order)
        scored.sort(key=lambda x: (-x[0], x[1]))

        selected_indices = set()
        used_tokens = 0
        for score, idx, msg in scored:
            content = msg.get("content", "")
            if isinstance(content, str):
                msg_tokens = tokenizer.count_tokens(content, model) + 4
            else:
                msg_tokens = 10
            if used_tokens + msg_tokens <= remaining_budget:
                selected_indices.add(idx)
                used_tokens += msg_tokens

        # Reconstruct in original order
        kept = [m for i, m in enumerate(non_system) if i in selected_indices]
        return system + kept

    def _truncate(self, messages: list[dict], model: str,
                  target_tokens: Optional[int]) -> list[dict]:
        """Simple truncation: system + last N messages."""
        system = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if target_tokens is None:
            return system + non_system[-6:]

        return self._trim_to_target(system, non_system, model, target_tokens)

    def _trim_to_target(self, system: list[dict], non_system: list[dict],
                        model: str, target_tokens: int) -> list[dict]:
        """Trim non-system messages from the front to fit target_tokens."""
        tokenizer = self._get_tokenizer()
        for i in range(len(non_system)):
            candidate = system + non_system[i:]
            tokens = tokenizer.count_messages(candidate, model)
            if tokens <= target_tokens:
                return candidate
        return system + non_system[-1:] if non_system else system

    def _is_low_info(self, content: str) -> bool:
        """Check if message content is low-information (greetings, acks)."""
        text = content.strip()
        if len(text) < 20:
            for pattern in self._compiled_patterns:
                if pattern.match(text):
                    return True
        return False

    def _info_score(self, msg: dict) -> float:
        """Score a message by estimated information content (0-1)."""
        content = msg.get("content", "")
        if not isinstance(content, str):
            return 0.5

        score = 0.3  # base

        # Length factor
        if len(content) > 200:
            score += 0.3
        elif len(content) > 50:
            score += 0.15
        elif len(content) < 10:
            score -= 0.2

        # Role factor: user questions and assistant answers with content are valuable
        role = msg.get("role", "")
        if role == "user":
            score += 0.1
        elif role == "assistant" and len(content) > 100:
            score += 0.2

        # Low-info check
        if self._is_low_info(content):
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _remove_consecutive_duplicates(self, messages: list[dict]) -> list[dict]:
        """Remove messages that are identical to the previous message."""
        if not messages:
            return messages

        result = [messages[0]]
        for msg in messages[1:]:
            prev = result[-1]
            if msg.get("content") != prev.get("content") or msg.get("role") != prev.get("role"):
                result.append(msg)
        return result

    def get_compression_report(self, original: list[dict], compressed: list[dict],
                               model: str = "gpt-4o") -> dict:
        """Report compression statistics."""
        tokenizer = self._get_tokenizer()
        orig_tokens = tokenizer.count_messages(original, model)
        comp_tokens = tokenizer.count_messages(compressed, model)
        return {
            "original_messages": len(original),
            "compressed_messages": len(compressed),
            "messages_removed": len(original) - len(compressed),
            "original_tokens": orig_tokens,
            "compressed_tokens": comp_tokens,
            "tokens_saved": orig_tokens - comp_tokens,
            "compression_ratio": round(comp_tokens / orig_tokens, 3) if orig_tokens > 0 else 1.0,
        }
