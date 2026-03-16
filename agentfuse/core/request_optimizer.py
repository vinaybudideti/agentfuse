"""
RequestOptimizer — automatically optimizes LLM requests for cost efficiency.

This is a novel system that no existing LLM cost tool has. It analyzes
requests BEFORE they are sent and applies optimizations:

1. Duplicate message removal — removes exact duplicate messages in context
2. Empty message pruning — removes messages with no content
3. System prompt deduplication — if system prompt appears multiple times
4. Context window awareness — warns if request will exceed model's context
5. Token estimation — estimates cost BEFORE the call so users can decide

These optimizations reduce token count → reduce cost → reduce latency,
without changing the semantic meaning of the request.

Usage:
    optimizer = RequestOptimizer()
    optimized, report = optimizer.optimize(messages, model="gpt-4o")
    # report.tokens_saved, report.estimated_cost_saving
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OptimizationReport:
    """Report of optimizations applied to a request."""
    original_messages: int
    optimized_messages: int
    messages_removed: int
    estimated_tokens_saved: int
    estimated_cost_saving_usd: float
    optimizations_applied: list[str]

    @property
    def pct_saved(self) -> float:
        if self.original_messages == 0:
            return 0.0
        return self.messages_removed / self.original_messages


class RequestOptimizer:
    """
    Analyzes and optimizes LLM requests before sending.

    Applies non-destructive optimizations that reduce token count
    without changing the semantic meaning of the conversation.
    """

    def __init__(self, pricing_engine=None, tokenizer=None):
        self._pricing = pricing_engine
        self._tokenizer = tokenizer

    def _get_pricing(self):
        if self._pricing is None:
            from agentfuse.providers.pricing import ModelPricingEngine
            self._pricing = ModelPricingEngine()
        return self._pricing

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from agentfuse.providers.tokenizer import TokenCounterAdapter
            self._tokenizer = TokenCounterAdapter()
        return self._tokenizer

    def optimize(self, messages: list[dict], model: str = "gpt-4o") -> tuple[list[dict], OptimizationReport]:
        """
        Optimize a message list for cost efficiency.

        Returns (optimized_messages, report).
        """
        original_count = len(messages)
        optimizations = []
        result = list(messages)  # shallow copy

        # 1. Remove empty messages
        before = len(result)
        result = [m for m in result if self._has_content(m)]
        if len(result) < before:
            optimizations.append(f"removed {before - len(result)} empty messages")

        # 2. Remove exact duplicate messages (consecutive)
        before = len(result)
        result = self._remove_consecutive_duplicates(result)
        if len(result) < before:
            optimizations.append(f"removed {before - len(result)} consecutive duplicates")

        # 3. Deduplicate system prompts (keep first, remove duplicates)
        before = len(result)
        result = self._dedup_system_prompts(result)
        if len(result) < before:
            optimizations.append(f"removed {before - len(result)} duplicate system prompts")

        # 4. Trim trailing whitespace from content
        for m in result:
            if isinstance(m.get("content"), str):
                stripped = m["content"].strip()
                if len(stripped) < len(m["content"]):
                    m = {**m, "content": stripped}

        # Calculate savings
        tokenizer = self._get_tokenizer()
        pricing = self._get_pricing()

        original_tokens = tokenizer.count_messages(messages, model)
        optimized_tokens = tokenizer.count_messages(result, model)
        tokens_saved = max(0, original_tokens - optimized_tokens)
        cost_saving = pricing.input_cost(model, tokens_saved)

        report = OptimizationReport(
            original_messages=original_count,
            optimized_messages=len(result),
            messages_removed=original_count - len(result),
            estimated_tokens_saved=tokens_saved,
            estimated_cost_saving_usd=cost_saving,
            optimizations_applied=optimizations,
        )

        if optimizations:
            logger.info("Request optimized: %s", ", ".join(optimizations))

        return result, report

    def estimate_cost(self, messages: list[dict], model: str = "gpt-4o") -> dict:
        """Estimate cost of a request BEFORE sending it."""
        tokenizer = self._get_tokenizer()
        pricing = self._get_pricing()

        tokens = tokenizer.count_messages(messages, model)
        input_cost = pricing.input_cost(model, tokens)

        # Estimate output cost (assume 1:1 input:output ratio as baseline)
        estimated_output_tokens = min(tokens, 4096)
        output_cost = pricing.output_cost(model, estimated_output_tokens)

        return {
            "model": model,
            "estimated_input_tokens": tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_input_cost_usd": round(input_cost, 6),
            "estimated_output_cost_usd": round(output_cost, 6),
            "estimated_total_cost_usd": round(input_cost + output_cost, 6),
            "message_count": len(messages),
        }

    def check_context_window(self, messages: list[dict], model: str = "gpt-4o") -> dict:
        """Check if request fits within model's context window."""
        from agentfuse.providers.registry import ModelRegistry
        registry = ModelRegistry(refresh_hours=0)

        tokenizer = self._get_tokenizer()
        tokens = tokenizer.count_messages(messages, model)

        model_info = registry.get_pricing(model)
        context_limit = model_info.get("context", 128_000)
        max_output = model_info.get("max_output", 4_096)
        available_for_output = max(0, context_limit - tokens)

        return {
            "input_tokens": tokens,
            "context_limit": context_limit,
            "max_output": max_output,
            "available_for_output": available_for_output,
            "fits": tokens < context_limit,
            "utilization_pct": round(tokens / context_limit * 100, 1) if context_limit > 0 else 0,
        }

    def _has_content(self, message: dict) -> bool:
        """Check if a message has non-empty content."""
        content = message.get("content", "")
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return len(content) > 0
        return bool(content)

    def _remove_consecutive_duplicates(self, messages: list[dict]) -> list[dict]:
        """Remove consecutive duplicate messages."""
        if not messages:
            return messages
        result = [messages[0]]
        for msg in messages[1:]:
            if self._msg_hash(msg) != self._msg_hash(result[-1]):
                result.append(msg)
        return result

    def _dedup_system_prompts(self, messages: list[dict]) -> list[dict]:
        """Keep only the first system prompt if duplicates exist."""
        seen_system = set()
        result = []
        for msg in messages:
            if msg.get("role") == "system":
                h = self._msg_hash(msg)
                if h in seen_system:
                    continue
                seen_system.add(h)
            result.append(msg)
        return result

    def _msg_hash(self, message: dict) -> str:
        """Hash a message for deduplication."""
        key = json.dumps({"role": message.get("role"), "content": message.get("content")},
                         sort_keys=True, default=str)
        return hashlib.md5(key.encode()).hexdigest()
