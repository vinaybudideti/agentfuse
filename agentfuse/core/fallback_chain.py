"""
FallbackModelChain — automatic model fallback on failure.

When the primary model fails (rate limit, timeout, overloaded), this
automatically tries cheaper/more-available alternatives in order.

This is different from CostAwareRetry (which retries the SAME call):
FallbackModelChain switches to a DIFFERENT model entirely, preserving
the conversation context.

Production use case: If Claude Opus is overloaded (529), immediately
fall back to Claude Sonnet, then Haiku, without user intervention.
"""

import logging
from typing import Callable, Optional
from agentfuse.core.error_classifier import classify_error

logger = logging.getLogger(__name__)


# Default fallback chains per model family
DEFAULT_CHAINS: dict[str, list[str]] = {
    # Anthropic
    "claude-opus-4-6": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "claude-sonnet-4-6": ["claude-haiku-4-5-20251001"],
    # OpenAI legacy
    "gpt-4o": ["gpt-4o-mini", "gpt-4.1"],
    # OpenAI current
    "gpt-4.1": ["gpt-4.1-mini", "gpt-4.1-nano"],
    "gpt-4.1-mini": ["gpt-4.1-nano", "gpt-4o-mini"],
    "o3": ["o4-mini", "gpt-4.1"],
    "o1": ["o3", "o4-mini"],
    # OpenAI GPT-5 family
    "gpt-5.4": ["gpt-5", "gpt-4.1"],
    "gpt-5": ["gpt-4.1", "gpt-4.1-mini"],
    # Gemini
    "gemini-2.5-pro": ["gemini-2.0-flash"],
    "gemini-1.5-pro": ["gemini-1.5-flash", "gemini-2.0-flash"],
}


class FallbackModelChain:
    """
    Tries a chain of models until one succeeds.

    Usage:
        chain = FallbackModelChain("claude-opus-4-6")
        result = chain.call(fn, messages)
        # If Opus fails → tries Sonnet → tries Haiku
    """

    def __init__(
        self,
        primary_model: str,
        fallback_models: Optional[list[str]] = None,
        provider: str = "unknown",
    ):
        self.primary_model = primary_model
        self.fallbacks = fallback_models or DEFAULT_CHAINS.get(primary_model, [])
        self.provider = provider
        self.model_used = primary_model
        self.fallback_count = 0

    def call(self, fn: Callable, messages: list, **kwargs) -> any:
        """
        Call fn(messages, model, **kwargs) with automatic model fallback.

        fn must accept (messages, model) as first two arguments.
        Returns the response from whichever model succeeds.
        """
        all_models = [self.primary_model] + self.fallbacks

        last_error = None
        for model in all_models:
            try:
                result = fn(messages, model, **kwargs)
                self.model_used = model
                if model != self.primary_model:
                    self.fallback_count += 1
                    logger.info("Fallback successful: %s → %s", self.primary_model, model)
                return result
            except Exception as e:
                classified = classify_error(e, self.provider)
                if not classified.retryable:
                    raise  # non-retryable errors don't trigger fallback

                logger.warning("Model %s failed (%s), trying next fallback", model, classified.error_type)
                last_error = e

        # All models failed
        raise last_error

    async def call_async(self, fn: Callable, messages: list, **kwargs) -> any:
        """Async version of call()."""
        all_models = [self.primary_model] + self.fallbacks

        last_error = None
        for model in all_models:
            try:
                result = await fn(messages, model, **kwargs)
                self.model_used = model
                if model != self.primary_model:
                    self.fallback_count += 1
                return result
            except Exception as e:
                classified = classify_error(e, self.provider)
                if not classified.retryable:
                    raise
                last_error = e

        raise last_error

    def get_status(self) -> dict:
        return {
            "primary_model": self.primary_model,
            "model_used": self.model_used,
            "fallback_count": self.fallback_count,
            "available_fallbacks": self.fallbacks,
        }
