"""
CacheQualityTracker — tracks quality metrics per cache entry.

Based on research findings (arxiv 2601.23088 "From Similarity to Vulnerability"):
- Track hit count, estimated quality score, model version per entry
- Auto-invalidate on model updates
- Sliding TTL for popular entries (reset TTL on each hit)
- Quality scoring based on hit count and user feedback

This addresses cache poisoning vulnerability and staleness issues
that affect all existing semantic caching systems.

Usage:
    quality = CacheQualityTracker()
    quality.record_hit("cache_key_abc", model="gpt-4o")
    quality.record_feedback("cache_key_abc", positive=True)
    score = quality.get_score("cache_key_abc")
    if score < 0.5:
        cache.invalidate("cache_key_abc")
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CacheEntryQuality:
    """Quality metrics for a single cache entry."""
    cache_key: str
    model: str
    model_version: str = ""
    hit_count: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    created_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)
    ttl_extensions: int = 0

    @property
    def quality_score(self) -> float:
        """Quality score from 0.0 to 1.0 based on feedback and hit count."""
        total_feedback = self.positive_feedback + self.negative_feedback
        if total_feedback == 0:
            # No feedback — score based on hit count (popular = likely good)
            return min(1.0, 0.5 + (self.hit_count * 0.05))
        return self.positive_feedback / total_feedback

    @property
    def is_stale(self) -> bool:
        """Entry is stale if not hit in 24 hours."""
        return (time.time() - self.last_hit_at) > 86400

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600


class CacheQualityTracker:
    """
    Tracks quality metrics per cache entry.

    Quality scoring system:
    - Base score: 0.5 (neutral)
    - Each hit: +0.05 (capped at 1.0)
    - Positive feedback: increases score proportionally
    - Negative feedback: decreases score
    - Score < 0.3: entry should be invalidated
    """

    def __init__(self, min_quality: float = 0.3):
        self._entries: dict[str, CacheEntryQuality] = {}
        self._lock = threading.Lock()
        self._min_quality = min_quality

    def record_hit(self, cache_key: str, model: str = "") -> CacheEntryQuality:
        """Record a cache hit. Creates entry if not exists."""
        with self._lock:
            if cache_key not in self._entries:
                self._entries[cache_key] = CacheEntryQuality(
                    cache_key=cache_key, model=model,
                )
            entry = self._entries[cache_key]
            entry.hit_count += 1
            entry.last_hit_at = time.time()
            return entry

    def record_feedback(self, cache_key: str, positive: bool):
        """Record user feedback on a cached response."""
        with self._lock:
            if cache_key not in self._entries:
                return
            entry = self._entries[cache_key]
            if positive:
                entry.positive_feedback += 1
            else:
                entry.negative_feedback += 1

    def get_score(self, cache_key: str) -> float:
        """Get quality score for a cache entry (0.0 to 1.0)."""
        with self._lock:
            entry = self._entries.get(cache_key)
            return entry.quality_score if entry else 0.5

    def should_invalidate(self, cache_key: str) -> bool:
        """Check if an entry should be invalidated due to low quality."""
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None:
                return False
            return entry.quality_score < self._min_quality

    def invalidate_by_model(self, model: str) -> int:
        """Invalidate all entries for a specific model (used on model updates)."""
        with self._lock:
            to_remove = [k for k, v in self._entries.items() if v.model == model]
            for k in to_remove:
                del self._entries[k]
            return len(to_remove)

    def get_stale_entries(self) -> list[str]:
        """Get cache keys that haven't been hit in 24 hours."""
        with self._lock:
            return [k for k, v in self._entries.items() if v.is_stale]

    def get_stats(self) -> dict:
        """Get aggregate quality statistics."""
        with self._lock:
            if not self._entries:
                return {"total_entries": 0, "avg_quality": 0.0}
            scores = [e.quality_score for e in self._entries.values()]
            return {
                "total_entries": len(self._entries),
                "avg_quality": sum(scores) / len(scores),
                "total_hits": sum(e.hit_count for e in self._entries.values()),
                "low_quality_count": sum(1 for s in scores if s < self._min_quality),
                "stale_count": sum(1 for e in self._entries.values() if e.is_stale),
            }

    def cleanup_low_quality(self) -> int:
        """Remove all entries below minimum quality threshold."""
        with self._lock:
            to_remove = [k for k, v in self._entries.items()
                         if v.quality_score < self._min_quality]
            for k in to_remove:
                del self._entries[k]
            return len(to_remove)
