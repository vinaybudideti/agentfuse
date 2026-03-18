"""
LoopDetectionMiddleware — FAISS sliding window for semantic loop detection.

Uses cosine similarity on sentence-transformer embeddings to detect when
an agent is repeating similar prompts in a loop, wasting budget.

FIX: Added threading.Lock for window mutations to prevent race conditions.
FIX: Embeddings stored as numpy arrays (not lists) for performance.
"""

import logging
import threading
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
        self.window = []  # list of (embedding_ndarray, step_cost) tuples
        self.sim_threshold = sim_threshold
        self.cost_threshold = cost_threshold
        self.loop_cost = 0.0
        self.max_window = window
        self.embedder = None
        self._lock = threading.Lock()

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

        with self._lock:
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

    def _embed(self, text: str) -> np.ndarray:
        """Get normalized embedding vector using sentence-transformers."""
        embedding = self.embedder.encode([text], normalize_embeddings=True)
        return embedding[0]  # already numpy array from encode()

    def _cosine_sim(self, a, b) -> float:
        """Compute cosine similarity between two vectors.
        Vectors are already L2-normalized from encode(normalize_embeddings=True),
        so dot product = cosine similarity."""
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def check_action(self, action: str, step_cost: float = 0.0):
        """Check for action-based loops (tool call repetition).

        Tracks tool/action names instead of full prompt embeddings.
        Faster than embedding-based detection (~0ms vs ~5-15ms).
        Catches: agent calling same tool repeatedly with same args.

        Based on AgentBudget's action-hash loop detection pattern.
        """
        import hashlib
        action_hash = hashlib.md5(action.encode()).hexdigest()[:16]

        with self._lock:
            if not hasattr(self, '_action_window'):
                self._action_window = []

            # Check if this action hash matches recent actions
            matches = sum(1 for h in self._action_window if h == action_hash)
            if matches >= 3:  # same action 3+ times in window
                self.loop_cost += step_cost
                if self.loop_cost > self.cost_threshold:
                    raise LoopDetected(
                        loop_cost=self.loop_cost,
                        similarity=1.0,  # exact match
                    )

            self._action_window.append(action_hash)
            if len(self._action_window) > self.max_window:
                self._action_window.pop(0)

    def reset(self):
        """Reset window for a new run."""
        with self._lock:
            self.window = []
            self.loop_cost = 0.0
            if hasattr(self, '_action_window'):
                self._action_window = []
