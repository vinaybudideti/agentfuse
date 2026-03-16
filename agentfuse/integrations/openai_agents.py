from uuid import uuid4
from agentfuse.core.budget import BudgetEngine, BudgetExhaustedGracefully
from agentfuse.core.cache import CacheMiddleware, CacheHit


class CacheHitException(Exception):
    def __init__(self, response):
        self.response = response


class AgentFuseRunHooks:
    """
    RunHooks implementation for OpenAI Agents SDK.

    Usage (2 lines):
        from agentfuse.integrations.openai_agents import AgentFuseRunHooks
        result = await Runner.run(agent, hooks=AgentFuseRunHooks(budget=5.00))
    """

    def __init__(self, budget: float, run_id: str = None,
                 model: str = "gpt-4o", alert_cb=None):
        self.run_id = run_id or str(uuid4())
        self.engine = BudgetEngine(self.run_id, budget, model, alert_cb)
        self.cache = CacheMiddleware()

    def on_llm_start(self, context, messages):
        """
        Called before each LLM call.
        Raises CacheHitException to return cached response.
        Raises BudgetExhaustedGracefully if budget is exhausted.
        """
        from agentfuse.providers.pricing import ModelPricingEngine
        from agentfuse.providers.tokenizer import TokenCounterAdapter

        pricing = ModelPricingEngine()
        tokenizer = TokenCounterAdapter()

        prompt = " ".join(
            m.get("content", "") for m in messages
            if isinstance(m.get("content"), str)
        )

        # Check cache first
        cache_result = self.cache.check(prompt, self.engine.model)
        if isinstance(cache_result, CacheHit):
            raise CacheHitException(response=cache_result.response)

        # Check budget
        token_count = tokenizer.count_messages_tokens(messages, self.engine.model)
        est_cost = pricing.input_cost(self.engine.model, token_count)
        new_messages, active_model = self.engine.check_and_act(est_cost, messages)

        if hasattr(context, "model_override"):
            context.model_override = active_model

    def on_llm_end(self, context, response):
        """Called after each LLM call. Record cost."""
        from agentfuse.providers.pricing import ModelPricingEngine

        pricing = ModelPricingEngine()

        try:
            usage = getattr(context, "usage", None)
            if usage:
                cost = pricing.total_cost(
                    self.engine.model,
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0)
                )
                self.engine.record_cost(cost)
        except Exception:
            pass

    def get_receipt(self):
        return {
            "run_id": self.run_id,
            "spent_usd": self.engine.spent,
            "budget_usd": self.engine.budget,
            "model": self.engine.model,
        }
