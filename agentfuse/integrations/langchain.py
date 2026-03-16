"""
LangChain integration — BaseChatModel wrapper for cache + budget.

CRITICAL: LangChain callbacks fire AFTER dispatch — they cannot intercept
and return cached responses. The ONLY correct approach is a BaseChatModel wrapper.
"""

import logging
from uuid import uuid4
from typing import Any, Optional

from agentfuse.core.budget import BudgetEngine
from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit

logger = logging.getLogger(__name__)


def _extract_text(response) -> str:
    """Extract text from various response formats (OpenAI, Anthropic, raw)."""
    # LangChain ChatResult/AIMessage
    if hasattr(response, "generations") and response.generations:
        gen = response.generations[0]
        if hasattr(gen, "text"):
            return gen.text
        if hasattr(gen, "message") and hasattr(gen.message, "content"):
            return gen.message.content

    # OpenAI format
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        return getattr(msg, "content", "") or ""

    # Anthropic format
    if hasattr(response, "content") and isinstance(response.content, list):
        for block in response.content:
            if hasattr(block, "text"):
                return block.text

    # Direct string
    if isinstance(response, str):
        return response

    return str(response)


class AgentFuseChatModel:
    """
    BaseChatModel wrapper that checks cache before delegating to inner model.

    Usage:
        from agentfuse.integrations.langchain import AgentFuseChatModel
        model = AgentFuseChatModel(inner=ChatOpenAI(), budget=5.00)
    """

    def __init__(
        self,
        inner=None,
        budget: float = 10.0,
        run_id: Optional[str] = None,
        model: str = "gpt-4o",
        alert_cb=None,
    ):
        self.inner = inner
        self.run_id = run_id or str(uuid4())
        self.engine = BudgetEngine(self.run_id, budget, model, alert_cb)
        self.cache = TwoTierCacheMiddleware()

    @property
    def _llm_type(self) -> str:
        return "agentfuse"

    def _generate(self, messages: list, stop=None, run_manager=None, **kwargs):
        """Check cache first, return if hit, else delegate to inner model."""
        from agentfuse.core.keys import build_cache_key

        # Convert LangChain messages to dicts if needed
        msg_dicts = []
        for m in messages:
            if isinstance(m, dict):
                msg_dicts.append(m)
            elif hasattr(m, "type") and hasattr(m, "content"):
                role = getattr(m, "type", "user")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                msg_dicts.append({"role": role, "content": m.content})
            else:
                msg_dicts.append({"role": "user", "content": str(m)})

        # Check cache
        cache_key = build_cache_key(msg_dicts, self.engine.model)
        cache_result = self.cache.check(cache_key, self.engine.model)
        if isinstance(cache_result, CacheHit):
            return cache_result.response

        # Budget check
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.providers.tokenizer import TokenCounterAdapter

        pricing = ModelPricingEngine()
        tokenizer = TokenCounterAdapter()
        token_count = tokenizer.count_messages_tokens(msg_dicts, self.engine.model)
        est_cost = pricing.input_cost(self.engine.model, token_count)
        new_messages, active_model = self.engine.check_and_act(est_cost, msg_dicts)

        # Delegate to inner model
        if self.inner is None:
            raise RuntimeError("No inner model configured")

        try:
            response = self.inner.invoke(messages, stop=stop, **kwargs)
            response_text = _extract_text(response)

            # Record cost and cache
            if response_text:
                self.cache.store(cache_key, response_text, self.engine.model)

            return response
        except Exception:
            raise

    def invoke(self, messages, **kwargs):
        """Main entry point."""
        return self._generate(messages, **kwargs)

    def get_receipt(self):
        return {
            "run_id": self.run_id,
            "spent_usd": self.engine.spent,
            "budget_usd": self.engine.budget,
            "model": self.engine.model,
        }


# Keep backward compat alias
AgentFuseLangChainMiddleware = AgentFuseChatModel


def create_langchain_model(inner=None, budget: float = 10.0, **kwargs):
    """Factory function for creating an AgentFuse-wrapped LangChain model."""
    return AgentFuseChatModel(inner=inner, budget=budget, **kwargs)
