"""
LangChain integration — BaseChatModel wrapper for cache + budget.

CRITICAL: LangChain callbacks fire AFTER dispatch — they cannot intercept
and return cached responses. The ONLY correct approach is a BaseChatModel wrapper.

PRODUCTION FIX: Records actual cost after inner model call.
PRODUCTION FIX: Uses new two-tier cache lookup() API.
PRODUCTION FIX: Passes compressed messages to inner model after budget check.
"""

import logging
from uuid import uuid4
from typing import Optional

from agentfuse.core.budget import BudgetEngine
from agentfuse.core.cache import TwoTierCacheMiddleware, CacheHit
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.providers.tokenizer import TokenCounterAdapter

logger = logging.getLogger(__name__)


def _extract_text(response) -> str:
    """Extract text from various response formats (OpenAI, Anthropic, LangChain, raw)."""
    # LangChain ChatResult/AIMessage
    if hasattr(response, "generations") and response.generations:
        gen = response.generations[0]
        if hasattr(gen, "text"):
            return gen.text
        if hasattr(gen, "message") and hasattr(gen.message, "content"):
            return gen.message.content

    # LangChain AIMessage direct
    if hasattr(response, "content") and isinstance(response.content, str):
        return response.content

    # OpenAI format
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        return getattr(msg, "content", "") or ""

    # Anthropic format
    if hasattr(response, "content") and isinstance(response.content, list):
        for block in response.content:
            if hasattr(block, "text"):
                return block.text

    if isinstance(response, str):
        return response

    return str(response)


def _convert_messages(messages: list) -> list[dict]:
    """Convert LangChain messages to plain dicts."""
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
            elif role == "system":
                role = "system"
            msg_dicts.append({"role": role, "content": m.content})
        elif hasattr(m, "role") and hasattr(m, "content"):
            msg_dicts.append({"role": m.role, "content": m.content})
        else:
            msg_dicts.append({"role": "user", "content": str(m)})
    return msg_dicts


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
        self._pricing = ModelPricingEngine()
        self._tokenizer = TokenCounterAdapter()

    @property
    def _llm_type(self) -> str:
        return "agentfuse"

    def _generate(self, messages: list, stop=None, run_manager=None, **kwargs):
        """Check cache first, return if hit, else delegate to inner model."""
        msg_dicts = _convert_messages(messages)
        temperature = kwargs.get("temperature", 0.0)
        tools = kwargs.get("tools", None)

        # Step 1: Check cache using two-tier API
        cache_result = self.cache.lookup(
            model=self.engine.model,
            messages=msg_dicts,
            temperature=temperature,
            tools=tools,
        )
        if isinstance(cache_result, CacheHit):
            return cache_result.response

        # Step 2: Budget check
        token_count = self._tokenizer.count_messages_tokens(msg_dicts, self.engine.model)
        est_cost = self._pricing.input_cost(self.engine.model, token_count)
        new_messages, active_model = self.engine.check_and_act(est_cost, msg_dicts)

        # Step 3: Delegate to inner model
        if self.inner is None:
            raise RuntimeError("No inner model configured")

        response = self.inner.invoke(messages, stop=stop, **kwargs)
        response_text = _extract_text(response)

        # Step 4: Record actual cost
        # Estimate output tokens from response text
        if response_text:
            output_tokens = self._tokenizer.count_tokens(response_text, active_model)
            actual_cost = self._pricing.total_cost(active_model, token_count, output_tokens)
            self.engine.record_cost(actual_cost)

            # Step 5: Cache the response
            self.cache.store(
                model=active_model,
                messages=msg_dicts,
                response=response_text,
                temperature=temperature,
                tools=tools,
            )

        return response

    def invoke(self, messages, **kwargs):
        """Main entry point."""
        return self._generate(messages, **kwargs)

    def get_receipt(self):
        return {
            "run_id": self.run_id,
            "spent_usd": self.engine.spent,
            "budget_usd": self.engine.budget,
            "model": self.engine.model,
            "original_model": self.engine.original_model,
        }


# Keep backward compat alias
AgentFuseLangChainMiddleware = AgentFuseChatModel


def create_langchain_model(inner=None, budget: float = 10.0, **kwargs):
    """Factory function for creating an AgentFuse-wrapped LangChain model."""
    return AgentFuseChatModel(inner=inner, budget=budget, **kwargs)
