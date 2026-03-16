"""
CostAwareRetry — tenacity-style wrapper with per-retry cost tracking
and automatic model downgrade on retry.

Uses classify_error from error_classifier for provider-aware retry decisions.
Users should set max_retries=0 on their SDK clients (openai.OpenAI(), anthropic.Anthropic())
so AgentFuse has full control over retry behavior.

FIX: retry_cost_spent now resets per wrap() call to prevent premature budget exhaustion.
FIX: Logs final failure with context for debugging.
FIX: Recalculates token count after model downgrade for accurate cost tracking.
"""

import asyncio
import time
import logging

from agentfuse.core.error_classifier import classify_error

logger = logging.getLogger(__name__)


class RetryBudgetExhausted(Exception):
    def __init__(self, retry_cost_spent):
        self.retry_cost_spent = retry_cost_spent
        super().__init__(
            f"Retry budget exhausted: ${retry_cost_spent:.4f} spent on retries"
        )


class CostAwareRetry:

    # Use the same downgrade map as BudgetEngine for consistency
    RETRY_DOWNGRADE_MAP = {
        "gpt-4o": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4o-mini",
        "gpt-4.1": "o4-mini",
        "o1": "o3",
        "o3": "o4-mini",
        "claude-opus-4-6": "claude-sonnet-4-6",
        "claude-sonnet-4-6": "claude-haiku-4-5-20251001",
        "gemini-2.5-pro": "gemini-2.0-flash",
        "gemini-1.5-pro": "gemini-1.5-flash",
    }

    def __init__(self, budget_engine, max_retry_cost_usd=0.50, max_attempts=3,
                 provider: str = "unknown"):
        self.budget_engine = budget_engine
        self.max_retry_cost = max_retry_cost_usd
        self.retry_cost_spent = 0.0
        self.max_attempts = max_attempts
        self.provider = provider

    def wrap(self, fn, messages: list, model: str, provider: str = None):
        """
        Wraps an LLM call function with cost-aware retry logic.
        fn: callable that takes (messages, model) and returns a response.

        Uses classify_error for provider-aware retry decisions instead of
        string matching on exception names.
        """
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.providers.tokenizer import TokenCounterAdapter

        pricing = ModelPricingEngine()
        tokenizer = TokenCounterAdapter()
        prov = provider or self.provider

        # Reset per-call retry cost to prevent accumulation across calls
        call_retry_cost = 0.0

        attempt = 0
        current_model = model

        while attempt < self.max_attempts:
            try:
                return fn(messages, current_model)
            except Exception as e:
                classified = classify_error(e, prov)

                if not classified.retryable:
                    raise

                attempt += 1
                logger.info(
                    "Retry %d/%d: %s (%s), model=%s, wait=%ds",
                    attempt, self.max_attempts, classified.error_type,
                    classified.provider, current_model, 2 ** attempt,
                )

                if attempt >= self.max_attempts:
                    logger.warning(
                        "All %d retry attempts exhausted for model=%s, error=%s",
                        self.max_attempts, current_model, classified.error_type,
                    )
                    raise

                # Recalculate token count for current (possibly downgraded) model
                token_count = tokenizer.count_messages_tokens(
                    messages, current_model
                )
                retry_cost = pricing.input_cost(current_model, token_count)
                call_retry_cost += retry_cost
                self.retry_cost_spent += retry_cost

                if call_retry_cost > self.max_retry_cost:
                    raise RetryBudgetExhausted(
                        retry_cost_spent=call_retry_cost
                    )

                current_model = self.RETRY_DOWNGRADE_MAP.get(
                    current_model, current_model
                )

                # Respect Retry-After header if present
                wait_time = 2 ** attempt
                if classified.retry_after and classified.retry_after > wait_time:
                    wait_time = min(classified.retry_after, 60)

                time.sleep(wait_time)

    async def wrap_async(self, fn, messages: list, model: str, provider: str = None):
        """Async version of wrap() — uses asyncio.sleep instead of time.sleep.
        fn must be an async callable that takes (messages, model)."""
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.providers.tokenizer import TokenCounterAdapter

        pricing = ModelPricingEngine()
        tokenizer = TokenCounterAdapter()
        prov = provider or self.provider
        call_retry_cost = 0.0
        attempt = 0
        current_model = model

        while attempt < self.max_attempts:
            try:
                return await fn(messages, current_model)
            except Exception as e:
                classified = classify_error(e, prov)
                if not classified.retryable:
                    raise

                attempt += 1
                logger.info("Async retry %d/%d: %s, model=%s",
                            attempt, self.max_attempts, classified.error_type, current_model)

                if attempt >= self.max_attempts:
                    raise

                token_count = tokenizer.count_messages_tokens(messages, current_model)
                retry_cost = pricing.input_cost(current_model, token_count)
                call_retry_cost += retry_cost
                self.retry_cost_spent += retry_cost

                if call_retry_cost > self.max_retry_cost:
                    raise RetryBudgetExhausted(retry_cost_spent=call_retry_cost)

                current_model = self.RETRY_DOWNGRADE_MAP.get(current_model, current_model)

                wait_time = 2 ** attempt
                if classified.retry_after and classified.retry_after > wait_time:
                    wait_time = min(classified.retry_after, 60)

                await asyncio.sleep(wait_time)
