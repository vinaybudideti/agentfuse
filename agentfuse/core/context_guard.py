"""
ContextWindowGuard — prevents context window overflow before API call.

Production problem: Sending more tokens than a model's context window causes
a 400 Bad Request error, wasting the API call cost. This guard checks BEFORE
the call and auto-compresses if needed.

Combines with PromptCompressor for intelligent context reduction:
1. Check if messages fit the model's context window
2. If not, compress using the best available strategy
3. Reserve tokens for the expected output
4. Never send a request that will be rejected

Usage:
    guard = ContextWindowGuard()
    safe_messages = guard.ensure_fits(messages, model="gpt-4o", max_output=4000)
    # Returns original messages if they fit, compressed messages if not
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ContextWindowOverflow(Exception):
    """Raised when messages exceed context window even after compression."""
    def __init__(self, model: str, token_count: int, context_limit: int):
        self.model = model
        self.token_count = token_count
        self.context_limit = context_limit
        super().__init__(
            f"Context window overflow: {token_count} tokens > {context_limit} limit for {model}"
        )


class ContextWindowGuard:
    """
    Prevents context window overflow before API calls.

    Checks token count against model limits and auto-compresses if needed.
    """

    def __init__(self, tokenizer=None, registry=None, compressor=None):
        self._tokenizer = tokenizer
        self._registry = registry
        self._compressor = compressor

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from agentfuse.providers.tokenizer import TokenCounterAdapter
            self._tokenizer = TokenCounterAdapter()
        return self._tokenizer

    def _get_registry(self):
        if self._registry is None:
            from agentfuse.providers.registry import ModelRegistry
            self._registry = ModelRegistry(refresh_hours=0)
        return self._registry

    def _get_compressor(self):
        if self._compressor is None:
            from agentfuse.core.prompt_compressor import PromptCompressor
            self._compressor = PromptCompressor(self._get_tokenizer())
        return self._compressor

    def check(self, messages: list[dict], model: str,
              max_output_tokens: int = 4096) -> dict:
        """
        Check if messages fit within the model's context window.

        Returns dict with fits (bool), token_count, context_limit, headroom.
        """
        tokenizer = self._get_tokenizer()
        registry = self._get_registry()

        pricing = registry.get_pricing(model)
        context_limit = pricing.get("context", 0)
        if context_limit == 0:
            # Unknown model — can't check, assume it fits
            return {"fits": True, "token_count": 0, "context_limit": 0, "headroom": 0}

        token_count = tokenizer.count_messages(messages, model)
        available = context_limit - max_output_tokens
        fits = token_count <= available

        return {
            "fits": fits,
            "token_count": token_count,
            "context_limit": context_limit,
            "available_for_input": available,
            "headroom": available - token_count,
        }

    def ensure_fits(
        self,
        messages: list[dict],
        model: str,
        max_output_tokens: int = 4096,
        strategy: str = "smart",
    ) -> list[dict]:
        """
        Ensure messages fit within the model's context window.

        If they don't fit, compress using the specified strategy.
        If compression still doesn't fit, raises ContextWindowOverflow.

        Returns messages (original or compressed).
        """
        result = self.check(messages, model, max_output_tokens)

        if result["fits"]:
            return messages

        # Need compression
        target_tokens = result["available_for_input"]
        if target_tokens <= 0:
            raise ContextWindowOverflow(model, result["token_count"], result["context_limit"])

        logger.info("Context window guard: compressing %d → %d tokens for %s",
                     result["token_count"], target_tokens, model)

        compressor = self._get_compressor()
        compressed = compressor.compress(
            messages, model=model, target_tokens=target_tokens, strategy=strategy
        )

        # Verify compression worked
        new_check = self.check(compressed, model, max_output_tokens)
        if not new_check["fits"]:
            raise ContextWindowOverflow(
                model, new_check["token_count"], result["context_limit"]
            )

        return compressed

    def get_model_limits(self, model: str) -> dict:
        """Get context window limits for a model."""
        registry = self._get_registry()
        pricing = registry.get_pricing(model)
        return {
            "model": model,
            "context_window": pricing.get("context", 0),
            "max_output": pricing.get("max_output", 0),
            "available_for_input": pricing.get("context", 0) - pricing.get("max_output", 0),
        }
