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
from agentfuse.core.rate_limiter import TokenBucketRateLimiter, RateLimitExceeded
from agentfuse.core.cost_alert import CostAlertManager, CostAlert
from agentfuse.core.anomaly import CostAnomalyDetector
from agentfuse.core.adaptive_threshold import AdaptiveSimilarityThreshold
from agentfuse.core.response_validator import validate_response, validate_for_cache
from agentfuse.core.fallback_chain import FallbackModelChain
from agentfuse.core.cost_tracker import CostTracker
from agentfuse.core.dedup import RequestDeduplicator
from agentfuse.core.request_optimizer import RequestOptimizer
from agentfuse.core.load_balancer import ModelLoadBalancer
from agentfuse.core.middleware import MiddlewarePipeline, LLMRequest, LLMResponse
from agentfuse.core.cache_quality import CacheQualityTracker
from agentfuse.core.gcra_limiter import GCRARateLimiter
from agentfuse.core.batch_detector import BatchEligibilityDetector
from agentfuse.core.predictive_router import CostPredictiveRouter
from agentfuse.core.prompt_compressor import PromptCompressor
from agentfuse.core.tool_cost_tracker import ToolCostTracker, ToolCostExceeded
from agentfuse.core.conversation_estimator import ConversationCostEstimator
from agentfuse.core.hierarchical_budget import HierarchicalBudget
from agentfuse.core.session import AgentSession
from agentfuse.core.kill_switch import kill_switch, AgentKilled
from agentfuse.core.analytics import UsageAnalytics
from agentfuse.core.security import (
    mask_api_key, validate_api_key_format, check_prompt_injection,
    validate_response_safety, SecurityEvent,
)
from agentfuse.providers.openai import wrap_openai
from agentfuse.providers.anthropic import wrap_anthropic
from agentfuse.gateway import completion, get_spend_report, configure, estimate_cost, add_api_key

__version__ = "0.2.0"
__version_info__ = (0, 2, 0)
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
    # Rate limiting & alerts
    "TokenBucketRateLimiter",
    "RateLimitExceeded",
    "CostAlertManager",
    "CostAlert",
    # Anomaly detection
    "CostAnomalyDetector",
    # Adaptive cache
    "AdaptiveSimilarityThreshold",
    # Response validation
    "validate_response",
    "validate_for_cache",
    # Fallback & tracking
    "FallbackModelChain",
    "CostTracker",
    # Request optimization
    "RequestDeduplicator",
    "RequestOptimizer",
    "ModelLoadBalancer",
    # Middleware
    "MiddlewarePipeline",
    "LLMRequest",
    "LLMResponse",
    # Research-backed features
    "CacheQualityTracker",
    "GCRARateLimiter",
    "BatchEligibilityDetector",
    "CostPredictiveRouter",
    "PromptCompressor",
    "ToolCostTracker",
    "ToolCostExceeded",
    "ConversationCostEstimator",
    "HierarchicalBudget",
    "AgentSession",
    "kill_switch",
    "AgentKilled",
    "UsageAnalytics",
    # Gateway (unified entry point)
    "completion",
    "get_spend_report",
    "configure",
    "estimate_cost",
    "add_api_key",
    # Security
    "mask_api_key",
    "validate_api_key_format",
    "check_prompt_injection",
    "validate_response_safety",
    "SecurityEvent",
    # Provider wrappers (legacy)
    "wrap_openai",
    "wrap_anthropic",
]
