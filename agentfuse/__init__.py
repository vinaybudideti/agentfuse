# Public API: init(), track(), wrap_openai(), wrap_anthropic()

from agentfuse.core.budget import BudgetEngine, BudgetState, BudgetExhaustedGracefully
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.providers.tokenizer import TokenCounterAdapter
from agentfuse.storage.memory import InMemoryStore
from agentfuse.providers.openai import wrap_openai
from agentfuse.providers.anthropic import wrap_anthropic
from agentfuse.core.cache import CacheMiddleware
from agentfuse.core.loop import LoopDetectionMiddleware, LoopDetected
from agentfuse.core.retry import CostAwareRetry, RetryBudgetExhausted
from agentfuse.core.streaming import StreamingCostMiddleware, StreamCostLimitReached
from agentfuse.core.prompt_cache import PromptCachingMiddleware
from agentfuse.core.receipt import CostReceiptEmitter

__version__ = "0.1.0"
__all__ = [
    "BudgetEngine",
    "BudgetState",
    "BudgetExhaustedGracefully",
    "ModelPricingEngine",
    "TokenCounterAdapter",
    "InMemoryStore",
    "wrap_openai",
    "wrap_anthropic",
    "CacheMiddleware",
    "LoopDetectionMiddleware",
    "LoopDetected",
    "CostAwareRetry",
    "RetryBudgetExhausted",
    "StreamingCostMiddleware",
    "StreamCostLimitReached",
    "PromptCachingMiddleware",
    "CostReceiptEmitter",
]
