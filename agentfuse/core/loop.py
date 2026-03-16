"""
LoopDetectionMiddleware — FAISS sliding window for semantic loop detection.

Uses cosine similarity on sentence-transformer embeddings to detect when
an agent is repeating similar prompts in a loop, wasting budget.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class LoopDetected(Exception):
    def __init__(self, loop_cost, similarity):
        self.loop_cost = loop_cost
        self.similarity = similarity
        super().__init__(
            f"Loop detected: ${loop_cost:.4f} spent on loops "
            f"(similarity={similarity:.3f})"
        )


class LoopDetectionMiddleware:

    def __init__(self, window=10, sim_threshold=0.92, cost_threshold=0.50):
        self.window = []  # list of (embedding, step_cost) tuples
        self.sim_threshold = sim_threshold
        self.cost_threshold = cost_threshold
        self.loop_cost = 0.0
        self.max_window = window
        self.embedder = None

        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-mpnet-base-v2")
        except Exception as e:
            logger.warning("Embedder not available: %s", e)
            self.embedder = None

    def check(self, prompt: str, step_cost: float = 0.0):
        """Raises LoopDetected if a loop is found. Otherwise returns None."""
        if self.embedder is None:
            return

        emb = self._embed(prompt)

        for prev_emb, prev_cost in self.window:
            sim = self._cosine_sim(emb, prev_emb)
            if sim > self.sim_threshold:
                self.loop_cost += step_cost
                if self.loop_cost > self.cost_threshold:
                    raise LoopDetected(
                        loop_cost=self.loop_cost,
                        similarity=sim,
                    )

        self.window.append((emb, step_cost))
        if len(self.window) > self.max_window:
            self.window.pop(0)

    def _embed(self, text: str) -> list:
        """Get embedding vector using sentence-transformers."""
        embedding = self.embedder.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()

    def _cosine_sim(self, a: list, b: list) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr, b_arr = np.array(a), np.array(b)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def reset(self):
        """Reset window for a new run."""
        self.window = []
        self.loop_cost = 0.0
