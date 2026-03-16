# AgentFuse — Intelligent Agent Cost Optimization Runtime

from agentfuse.core.budget import BudgetEngine, BudgetState, BudgetExhaustedGracefully
from agentfuse.providers.pricing import ModelPricingEngine
from agentfuse.providers.tokenizer import TokenCounterAdapter
from agentfuse.providers.registry import ModelRegistry
from agentfuse.providers.response import NormalizedUsage, extract_usage
from agentfuse.providers.router import resolve_provider
from agentfuse.storage.memory import InMemoryStore, InMemoryBudgetStore
from agentfuse.core.cache import TwoTierCacheMiddleware, CacheMiddleware, CacheHit, CacheMiss
from agentfuse.core.loop import LoopDetectionMiddleware, LoopDetected
from agentfuse.core.retry import CostAwareRetry, RetryBudgetExhausted
from agentfuse.core.streaming import StreamingCostMiddleware, StreamCostLimitReached
from agentfuse.core.prompt_cache import PromptCachingMiddleware
from agentfuse.core.receipt import CostReceiptEmitter
from agentfuse.core.error_classifier import classify_error, ClassifiedError
from agentfuse.providers.openai import wrap_openai
from agentfuse.providers.anthropic import wrap_anthropic

__version__ = "0.2.0"
__all__ = [
    # Core
    "BudgetEngine",
    "BudgetState",
    "BudgetExhaustedGracefully",
    # Cache
    "TwoTierCacheMiddleware",
    "CacheMiddleware",
    "CacheHit",
    "CacheMiss",
    # Providers
    "ModelPricingEngine",
    "TokenCounterAdapter",
    "ModelRegistry",
    "NormalizedUsage",
    "extract_usage",
    "resolve_provider",
    # Storage
    "InMemoryStore",
    "InMemoryBudgetStore",
    # Reliability
    "LoopDetectionMiddleware",
    "LoopDetected",
    "CostAwareRetry",
    "RetryBudgetExhausted",
    "StreamingCostMiddleware",
    "StreamCostLimitReached",
    "PromptCachingMiddleware",
    "CostReceiptEmitter",
    # Error handling
    "classify_error",
    "ClassifiedError",
    # Provider wrappers
    "wrap_openai",
    "wrap_anthropic",
]
