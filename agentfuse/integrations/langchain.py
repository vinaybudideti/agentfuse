import logging
from uuid import uuid4
from agentfuse.core.budget import BudgetEngine
from agentfuse.core.cache import CacheMiddleware, CacheHit

logger = logging.getLogger(__name__)


class AgentFuseLangChainMiddleware:
    """
    LangChain callback handler that enforces per-run budgets and
    caches semantically similar prompts.

    Usage (2 lines):
        from agentfuse.integrations.langchain import AgentFuseLangChainMiddleware
        middleware = AgentFuseLangChainMiddleware(budget=5.00)
        agent = initialize_agent(..., callbacks=[middleware])
    """

    def __init__(self, budget: float, run_id: str = None,
                 model: str = "gpt-4o", alert_cb=None, **kwargs):
        self.run_id = run_id or str(uuid4())
        self.engine = BudgetEngine(self.run_id, budget, model, alert_cb)
        self.cache = CacheMiddleware()
        self._cache_results = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called before every LLM call. Check cache and budget."""
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.providers.tokenizer import TokenCounterAdapter
        from agentfuse.core.keys import build_cache_key

        pricing = ModelPricingEngine()
        tokenizer = TokenCounterAdapter()

        for prompt in prompts:
            cache_key = build_cache_key(
                [{"role": "user", "content": prompt}], self.engine.model
            )
            cache_result = self.cache.check(cache_key, self.engine.model)
            if isinstance(cache_result, CacheHit):
                self._cache_results[prompt] = cache_result
                continue

            token_count = tokenizer.count_tokens(prompt, self.engine.model)
            est_cost = pricing.input_cost(self.engine.model, token_count)
            self.engine.check_and_act(est_cost, [{"role": "user", "content": prompt}])

    def on_llm_end(self, response, **kwargs):
        """Called after every LLM call. Record actual cost."""
        from agentfuse.providers.pricing import ModelPricingEngine

        pricing = ModelPricingEngine()

        try:
            llm_output = getattr(response, "llm_output", None)
            if llm_output is None:
                return
            usage = llm_output.get("token_usage", {})
            if usage:
                cost = pricing.total_cost(
                    self.engine.model,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0)
                )
                self.engine.record_cost(cost)
        except (AttributeError, KeyError, TypeError) as e:
            logger.warning("LangChain on_llm_end cost recording failed: %s", e)

    def on_llm_error(self, error, **kwargs):
        logger.debug("LangChain LLM error: %s", error)

    def get_receipt(self):
        """Returns current run cost state."""
        return {
            "run_id": self.run_id,
            "spent_usd": self.engine.spent,
            "budget_usd": self.engine.budget,
            "model": self.engine.model,
        }
