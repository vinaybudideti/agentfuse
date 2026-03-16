"""
CostAwareRetry — tenacity-style wrapper with per-retry cost tracking
and automatic model downgrade on retry.
"""

import time
import logging

logger = logging.getLogger(__name__)


class RetryBudgetExhausted(Exception):
    def __init__(self, retry_cost_spent):
        self.retry_cost_spent = retry_cost_spent
        super().__init__(
            f"Retry budget exhausted: ${retry_cost_spent:.4f} spent on retries"
        )


class CostAwareRetry:

    RETRY_DOWNGRADE_MAP = {
        "gpt-4o": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4o-mini",
        "claude-opus-4-6": "claude-sonnet-4-6",
        "claude-sonnet-4-6": "claude-haiku-4-5-20251001",
        "gemini-1.5-pro": "gemini-1.5-flash",
    }

    def __init__(self, budget_engine, max_retry_cost_usd=0.50, max_attempts=3):
        self.budget_engine = budget_engine
        self.max_retry_cost = max_retry_cost_usd
        self.retry_cost_spent = 0.0
        self.max_attempts = max_attempts

    def wrap(self, fn, messages: list, model: str):
        """
        Wraps an LLM call function with cost-aware retry logic.
        fn: callable that takes (messages, model) and returns a response.
        """
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.providers.tokenizer import TokenCounterAdapter

        pricing = ModelPricingEngine()
        tokenizer = TokenCounterAdapter()

        attempt = 0
        current_model = model

        while attempt < self.max_attempts:
            try:
                return fn(messages, current_model)
            except Exception as e:
                error_name = type(e).__name__.lower()
                error_msg = str(e).lower()
                retryable = any(
                    k in error_name or k in error_msg
                    for k in ["ratelimit", "apierror", "timeout", "serviceunavailable"]
                )
                if not retryable:
                    raise

                attempt += 1
                if attempt >= self.max_attempts:
                    raise

                token_count = tokenizer.count_messages_tokens(
                    messages, current_model
                )
                retry_cost = pricing.input_cost(current_model, token_count)
                self.retry_cost_spent += retry_cost

                if self.retry_cost_spent > self.max_retry_cost:
                    raise RetryBudgetExhausted(
                        retry_cost_spent=self.retry_cost_spent
                    )

                current_model = self.RETRY_DOWNGRADE_MAP.get(
                    current_model, current_model
                )

                time.sleep(2 ** attempt)
