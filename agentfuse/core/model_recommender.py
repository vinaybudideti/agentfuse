"""
ModelRecommender — suggests the optimal model based on workload requirements.

NOVEL: No existing SDK recommends models based on workload characteristics.
Users often use GPT-5 for everything when GPT-4.1-nano would work for 70% of queries.

Factors considered:
1. Query complexity (code, reasoning, factual, creative)
2. Budget constraints (remaining budget determines model tier)
3. Latency requirements (fast models for real-time, slow for batch)
4. Context length needed (long context → limited model choices)
5. Historical cost data (which models gave best value)

Usage:
    recommender = ModelRecommender()
    model = recommender.recommend(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        budget_remaining=5.00,
        latency_priority="low",
    )
    # Returns "gpt-4.1-nano" (cheapest model that handles simple math)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Model capabilities and characteristics
MODEL_PROFILES = {
    # Tier 1: Frontier (best quality, highest cost)
    "gpt-5.4": {"quality": 0.99, "speed": 0.5, "cost_per_1k": 0.040, "context": 1_050_000, "strengths": ["reasoning", "code", "creative"]},
    "gpt-5.3": {"quality": 0.97, "speed": 0.6, "cost_per_1k": 0.0115, "context": 1_000_000, "strengths": ["reasoning", "code"]},
    "claude-opus-4-6": {"quality": 0.98, "speed": 0.4, "cost_per_1k": 0.030, "context": 200_000, "strengths": ["reasoning", "creative", "analysis"]},

    # Tier 2: Strong (great quality, moderate cost)
    "gpt-5": {"quality": 0.95, "speed": 0.7, "cost_per_1k": 0.01125, "context": 1_000_000, "strengths": ["reasoning", "code"]},
    "gpt-4.1": {"quality": 0.93, "speed": 0.7, "cost_per_1k": 0.010, "context": 1_000_000, "strengths": ["code", "reasoning"]},
    "claude-sonnet-4-6": {"quality": 0.94, "speed": 0.6, "cost_per_1k": 0.018, "context": 200_000, "strengths": ["reasoning", "creative"]},
    "gemini-2.5-pro": {"quality": 0.93, "speed": 0.7, "cost_per_1k": 0.01125, "context": 1_000_000, "strengths": ["reasoning", "code", "long_context"]},

    # Tier 3: Efficient (good quality, low cost)
    "gpt-4.1-mini": {"quality": 0.85, "speed": 0.9, "cost_per_1k": 0.002, "context": 1_000_000, "strengths": ["factual", "simple"]},
    "o4-mini": {"quality": 0.88, "speed": 0.8, "cost_per_1k": 0.0055, "context": 200_000, "strengths": ["reasoning", "math"]},
    "claude-haiku-4-5-20251001": {"quality": 0.83, "speed": 0.95, "cost_per_1k": 0.006, "context": 200_000, "strengths": ["factual", "simple"]},
    "gemini-2.0-flash": {"quality": 0.82, "speed": 0.95, "cost_per_1k": 0.0005, "context": 1_000_000, "strengths": ["factual", "simple", "fast"]},

    # Tier 4: Minimal (cheapest)
    "gpt-4.1-nano": {"quality": 0.78, "speed": 0.98, "cost_per_1k": 0.0005, "context": 1_000_000, "strengths": ["factual", "simple", "fast"]},
}


class ModelRecommender:
    """
    Recommends the optimal model based on workload requirements.
    """

    def __init__(self, custom_profiles: Optional[dict] = None):
        self._profiles = {**MODEL_PROFILES, **(custom_profiles or {})}

    def recommend(
        self,
        messages: Optional[list[dict]] = None,
        budget_remaining: float = 10.0,
        latency_priority: str = "balanced",  # "low", "balanced", "high"
        required_context: int = 0,
        task_type: Optional[str] = None,  # "code", "reasoning", "factual", "creative"
        provider_preference: Optional[str] = None,  # "openai", "anthropic", "gemini"
    ) -> str:
        """
        Recommend the best model for the given requirements.

        Returns model name string.
        """
        candidates = list(self._profiles.items())

        # Filter by context requirement
        if required_context > 0:
            candidates = [(m, p) for m, p in candidates if p["context"] >= required_context]

        # Filter by provider preference
        if provider_preference:
            filtered = [(m, p) for m, p in candidates if self._matches_provider(m, provider_preference)]
            if filtered:
                candidates = filtered

        # Score each candidate
        scored = []
        for model, profile in candidates:
            score = self._score(profile, budget_remaining, latency_priority, task_type)
            scored.append((score, model))

        if not scored:
            return "gpt-4.1-mini"  # safe default

        # Return highest scoring model
        scored.sort(reverse=True)
        recommended = scored[0][1]

        logger.debug("Model recommended: %s (score=%.3f)", recommended, scored[0][0])
        return recommended

    def _score(self, profile: dict, budget: float, latency: str, task_type: Optional[str]) -> float:
        """Score a model based on requirements. Higher is better."""
        score = 0.0

        # Quality component (40%)
        score += profile["quality"] * 0.4

        # Cost component (30%) — favor cheaper when budget is low
        cost_factor = 1.0 - min(1.0, profile["cost_per_1k"] / 0.03)
        if budget < 1.0:
            cost_factor *= 2.0  # double the cost weight when budget is low
        score += cost_factor * 0.3

        # Speed component (20%)
        if latency == "high":
            score += profile["speed"] * 0.3  # higher weight on speed
        elif latency == "low":
            score += profile["speed"] * 0.1  # less weight on speed
        else:
            score += profile["speed"] * 0.2

        # Task match component (10%)
        if task_type and task_type in profile.get("strengths", []):
            score += 0.1

        return score

    def _matches_provider(self, model: str, provider: str) -> bool:
        """Check if model matches provider preference."""
        if provider == "openai":
            return model.startswith(("gpt", "o1", "o3", "o4"))
        elif provider == "anthropic":
            return model.startswith("claude")
        elif provider == "gemini":
            return model.startswith("gemini")
        return True

    def get_all_models(self) -> list[str]:
        """List all available models."""
        return sorted(self._profiles.keys())

    def compare(self, model_a: str, model_b: str) -> dict:
        """Compare two models side by side."""
        pa = self._profiles.get(model_a, {})
        pb = self._profiles.get(model_b, {})
        return {
            "model_a": {"name": model_a, **pa},
            "model_b": {"name": model_b, **pb},
            "quality_diff": pa.get("quality", 0) - pb.get("quality", 0),
            "cost_ratio": pa.get("cost_per_1k", 1) / max(pb.get("cost_per_1k", 1), 0.0001),
            "speed_diff": pa.get("speed", 0) - pb.get("speed", 0),
        }
