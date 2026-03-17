"""
ResponseQualityScorer — automated quality scoring for LLM responses.

Scores responses on multiple dimensions BEFORE caching:
1. Relevancy: Does the response address the query?
2. Completeness: Is the response thorough enough?
3. Conciseness: Is it too verbose (wastes cache space)?
4. Coherence: Is the response well-structured?

Only responses above a quality threshold are cached, preventing
low-quality responses from being served to future users.

This is a NOVEL approach — no existing caching system scores
response quality before storing. GPTCache and Redis SemanticCache
cache everything, including garbage.

Usage:
    scorer = ResponseQualityScorer(cache_threshold=0.6)
    score = scorer.score(query="What is Python?", response="Python is...")
    if score.should_cache:
        cache.store(...)
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality assessment of an LLM response."""
    relevancy: float      # 0-1: does it address the query?
    completeness: float   # 0-1: is it thorough?
    conciseness: float    # 0-1: not too verbose?
    coherence: float      # 0-1: well-structured?
    overall: float        # 0-1: weighted average
    should_cache: bool    # True if quality meets threshold
    reason: str = ""      # why it should/shouldn't be cached


class ResponseQualityScorer:
    """
    Scores LLM responses for quality using heuristic analysis.

    No LLM call needed — uses pattern matching and text analysis.
    Fast enough for every response (~<1ms).
    """

    def __init__(self, cache_threshold: float = 0.5):
        self._threshold = cache_threshold

    def score(self, query: str, response: str) -> QualityScore:
        """Score a response for quality.

        Args:
            query: The original user query
            response: The LLM's response text

        Returns:
            QualityScore with dimension scores and cache recommendation
        """
        if not response or not response.strip():
            return QualityScore(0, 0, 0, 0, 0, False, "empty response")

        relevancy = self._score_relevancy(query, response)
        completeness = self._score_completeness(response)
        conciseness = self._score_conciseness(query, response)
        coherence = self._score_coherence(response)

        # Weighted average (relevancy most important)
        overall = (
            relevancy * 0.35 +
            completeness * 0.25 +
            conciseness * 0.20 +
            coherence * 0.20
        )

        should_cache = overall >= self._threshold

        reason = ""
        if not should_cache:
            if relevancy < 0.3:
                reason = "response appears irrelevant to query"
            elif completeness < 0.3:
                reason = "response too short/incomplete"
            elif conciseness < 0.3:
                reason = "response excessively verbose"
            else:
                reason = f"overall quality {overall:.2f} below threshold {self._threshold}"

        return QualityScore(
            relevancy=round(relevancy, 3),
            completeness=round(completeness, 3),
            conciseness=round(conciseness, 3),
            coherence=round(coherence, 3),
            overall=round(overall, 3),
            should_cache=should_cache,
            reason=reason,
        )

    def _score_relevancy(self, query: str, response: str) -> float:
        """Score how relevant the response is to the query."""
        if not query:
            return 0.5

        # Check word overlap between query and response
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "can", "shall",
                      "to", "of", "in", "for", "on", "with", "at", "by", "from",
                      "it", "this", "that", "and", "or", "but", "not", "what",
                      "how", "why", "when", "where", "who", "which", "i", "me", "my"}
        query_words -= stop_words
        response_words -= stop_words

        if not query_words:
            return 0.5

        overlap = query_words & response_words
        overlap_ratio = len(overlap) / len(query_words) if query_words else 0

        # Base score from overlap
        score = min(1.0, overlap_ratio * 1.5)

        # Bonus for longer, substantive responses
        if len(response) > 100:
            score = min(1.0, score + 0.1)

        return score

    def _score_completeness(self, response: str) -> float:
        """Score how complete/thorough the response is."""
        length = len(response)

        if length < 10:
            return 0.1
        elif length < 50:
            return 0.3
        elif length < 200:
            return 0.6
        elif length < 1000:
            return 0.8
        else:
            return 0.9

    def _score_conciseness(self, query: str, response: str) -> float:
        """Score conciseness — penalize excessive verbosity."""
        query_len = len(query) if query else 50
        response_len = len(response)

        ratio = response_len / max(query_len, 1)

        if ratio < 1:
            return 0.5  # response shorter than query
        elif ratio < 5:
            return 1.0  # good ratio
        elif ratio < 20:
            return 0.8  # slightly verbose
        elif ratio < 50:
            return 0.5  # verbose
        else:
            return 0.3  # excessively verbose

    def _score_coherence(self, response: str) -> float:
        """Score structural coherence."""
        score = 0.5  # base

        # Has sentence structure (periods, question marks)
        sentences = re.split(r'[.!?]+', response)
        if len(sentences) > 1:
            score += 0.2

        # Has paragraph structure (newlines)
        if '\n' in response:
            score += 0.1

        # No excessive repetition
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.5:
                score += 0.1
            elif unique_ratio < 0.2:
                score -= 0.2  # highly repetitive

        # Has formatting (lists, code blocks)
        if re.search(r'[-*]\s', response) or '```' in response:
            score += 0.1

        return min(1.0, max(0.0, score))
