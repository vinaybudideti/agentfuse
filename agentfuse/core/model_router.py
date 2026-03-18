"""
IntelligentModelRouter — routes queries to strong vs weak models based on complexity.

NOTE: For ML-based routing, install RouteLLM:
    pip install "routellm[serve,eval]"
    export OPENAI_API_KEY="sk-..."  # Required for embedding API

RouteLLM MF router: ~30-90ms overhead (reducible to <10ms with local embeddings),
generalizes to new model pairs without retraining (ICLR 2025).

Based on RouteLLM (ICLR 2025, UC Berkeley/Anyscale/Canva):
- Achieves 85% cost reduction at 95% of GPT-4 quality
- Routes only 26% of queries to expensive models, 74% to cheap models
- Uses query complexity heuristics since we can't ship the full MF model

The research doc calls this "the single largest optimization opportunity."

AgentFuse implements a lightweight version using heuristics:
1. Short simple queries → cheap model (haiku, gpt-4o-mini)
2. Complex reasoning/math/code → strong model (sonnet, gpt-4o)
3. Long context analysis → strong model
4. Simple factual Q&A → cheap model

This can be upgraded to use the full RouteLLM classifier later.

Usage:
    router = IntelligentModelRouter()
    model = router.route("gpt-4o", messages)
    # Returns "gpt-4o-mini" for simple queries, "gpt-4o" for complex ones
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Complexity indicators that suggest a strong model is needed
_COMPLEX_PATTERNS = [
    r"\b(?:explain|analyze|compare|evaluate|critique|debate)\b",
    r"\b(?:write|create|generate|compose)\b.*\b(?:detailed|comprehensive|thorough)\b",
    r"(?:step.by.step|chain.of.thought|think.carefully|reason.through)",
    r"\b(?:code|program|algorithm|implement)\b",
    r"\b(?:function|class|method|module|interface)\b",
    r"\b(?:math|equation|calcul|solve|proof|theorem)\b",
    r"\b(?:translate|summarize|paraphrase)\b.*\b(?:entire|full|complete)\b",
    r"\b(?:why|how)\b.*\b(?:work|happen|cause|affect)\b",
    r"\b(?:multi.step|complex|advanced|sophisticated|architecture|design|schema)\b",
    r"\b(?:tradeoff|consistency|sharding|scaling|distributed)\b",
]

# Simplicity indicators that suggest a cheap model is fine
_SIMPLE_PATTERNS = [
    r"^(?:what|who|where|when)\s+(?:is|are|was|were)\b",
    r"^(?:yes|no|true|false|define|list)\b",
    r"^(?:hi|hello|hey|thanks|thank you|ok|okay)\b",
    r"(?:convert|translate)\s+\d+\s+(?:to|into)\b",
    r"^(?:fix|correct)\s+(?:this|the)\s+(?:typo|error|mistake)\b",
]

_COMPILED_COMPLEX = [re.compile(p, re.IGNORECASE) for p in _COMPLEX_PATTERNS]
_COMPILED_SIMPLE = [re.compile(p, re.IGNORECASE) for p in _SIMPLE_PATTERNS]

# Model pairs: strong → weak
ROUTING_PAIRS = {
    # OpenAI legacy
    "gpt-4o": "gpt-4o-mini",
    # OpenAI current
    "gpt-4.1": "gpt-4.1-mini",
    "gpt-4.1-mini": "gpt-4.1-nano",
    "o3": "o4-mini",
    # OpenAI GPT-5 family
    "gpt-5.4": "gpt-5.3",
    "gpt-5.3": "gpt-5",
    "gpt-5": "gpt-4.1",
    # Anthropic
    "claude-opus-4-6": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6": "claude-haiku-4-5-20251001",
    # Gemini
    "gemini-2.5-pro": "gemini-2.0-flash",
    "gemini-1.5-pro": "gemini-1.5-flash",
}


class IntelligentModelRouter:
    """
    Routes queries to strong or weak models based on complexity analysis.

    Achieves cost savings by sending simple queries to cheaper models
    while preserving quality for complex queries that need it.
    """

    def __init__(self, threshold: float = 0.5, custom_pairs: Optional[dict] = None):
        """
        Args:
            threshold: Complexity score above which strong model is used (0.0-1.0)
            custom_pairs: Custom strong→weak model mapping
        """
        self._threshold = threshold
        self._pairs = {**ROUTING_PAIRS, **(custom_pairs or {})}
        self._total_routed = 0
        self._downrouted = 0

    def route(self, model: str, messages: list[dict]) -> str:
        """
        Decide whether to use the requested model or a cheaper alternative.

        Returns the model to actually use.
        """
        self._total_routed += 1

        # No cheaper alternative available
        if model not in self._pairs:
            return model

        # Analyze query complexity
        score = self._complexity_score(messages)

        if score < self._threshold:
            # Simple query → use cheap model
            cheap_model = self._pairs[model]

            # Context window check: ensure messages fit in the cheaper model
            try:
                from agentfuse.providers.registry import ModelRegistry
                registry = ModelRegistry(refresh_hours=0)
                cheap_pricing = registry.get_pricing(cheap_model)
                max_context = cheap_pricing.get("context", 0)
                if max_context > 0:
                    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
                    estimated_tokens = total_chars // 4  # rough estimate
                    if estimated_tokens > max_context * 0.8:
                        # Messages too long for cheap model — keep original
                        logger.debug("Query too long for %s (%d est. tokens) — keeping %s",
                                     cheap_model, estimated_tokens, model)
                        return model
            except Exception:
                pass  # if registry fails, still route

            self._downrouted += 1
            logger.debug("Query routed: %s → %s (complexity=%.2f)", model, cheap_model, score)
            return cheap_model

        # Complex query → keep original model
        return model

    def _complexity_score(self, messages: list[dict]) -> float:
        """
        Estimate query complexity from 0.0 (trivial) to 1.0 (very complex).

        Heuristics based on RouteLLM research:
        - Message length (longer = more complex)
        - Complex language patterns (reasoning, code, math)
        - Simple language patterns (factual Q&A)
        - Number of messages (multi-turn = more complex context)
        """
        # Extract the last user message
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_content = content if isinstance(content, str) else str(content)
                break

        if not user_content:
            return 0.5  # neutral

        score = 0.3  # base score

        # Length factor (longer queries tend to be more complex)
        char_count = len(user_content)
        if char_count > 500:
            score += 0.2
        elif char_count > 200:
            score += 0.1
        elif char_count < 30:
            score -= 0.1

        # Complex pattern matching
        complex_matches = sum(1 for p in _COMPILED_COMPLEX if p.search(user_content))
        score += min(0.3, complex_matches * 0.1)

        # Simple pattern matching
        simple_matches = sum(1 for p in _COMPILED_SIMPLE if p.search(user_content))
        score -= min(0.2, simple_matches * 0.1)

        # Multi-turn factor
        msg_count = len(messages)
        if msg_count > 6:
            score += 0.1  # long conversations are complex
        elif msg_count <= 2:
            score -= 0.05  # single exchange is simpler

        # System prompt complexity
        for msg in messages:
            if msg.get("role") == "system":
                sys_content = msg.get("content", "")
                if isinstance(sys_content, str) and len(sys_content) > 500:
                    score += 0.1  # complex system prompt suggests complex task
                break

        return max(0.0, min(1.0, score))

    def get_stats(self) -> dict:
        """Get routing statistics."""
        return {
            "total_routed": self._total_routed,
            "downrouted": self._downrouted,
            "downroute_rate": self._downrouted / self._total_routed if self._total_routed > 0 else 0.0,
            "estimated_savings_pct": self._downrouted / self._total_routed * 0.7 if self._total_routed > 0 else 0.0,
        }
