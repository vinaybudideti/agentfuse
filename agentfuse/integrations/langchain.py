from uuid import uuid4
from agentfuse.core.budget import BudgetEngine
from agentfuse.core.cache import CacheMiddleware, CacheHit


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
        self._cache_results = {}  # prompt -> CacheHit for intercepting

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called before every LLM call. Check cache and budget."""
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.providers.tokenizer import TokenCounterAdapter

        pricing = ModelPricingEngine()
        tokenizer = TokenCounterAdapter()

        for prompt in prompts:
            # Check cache first
            cache_result = self.cache.check(prompt, self.engine.model)
            if isinstance(cache_result, CacheHit):
                self._cache_results[prompt] = cache_result
                continue

            # Check budget
            token_count = tokenizer.count_tokens(prompt, self.engine.model)
            est_cost = pricing.input_cost(self.engine.model, token_count)
            self.engine.check_and_act(est_cost, [{"role": "user", "content": prompt}])

    def on_llm_end(self, response, **kwargs):
        """Called after every LLM call. Record actual cost."""
        from agentfuse.providers.pricing import ModelPricingEngine

        pricing = ModelPricingEngine()

        try:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                cost = pricing.total_cost(
                    self.engine.model,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0)
                )
                self.engine.record_cost(cost)
        except Exception:
            pass  # Never crash the agent

    def on_llm_error(self, error, **kwargs):
        pass  # Log but don't crash

    def get_receipt(self):
        """Returns current run cost state."""
        return {
            "run_id": self.run_id,
            "spent_usd": self.engine.spent,
            "budget_usd": self.engine.budget,
            "model": self.engine.model,
        }
