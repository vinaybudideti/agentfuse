"""
Intent Atoms — re-export from the nested intent_atoms package.
The git subtree brings in the full repo; the actual Python package
lives one level deeper at agentfuse/intent_atoms/intent_atoms/.
"""

from .intent_atoms import (
    IntentAtomsEngineV3,
    IntentAtomsEngineV2,
    IntentAtomsEngine,
    Atom,
    CachedQuery,
    QueryResult,
    CacheStats,
)
from .intent_atoms.faiss_store import FAISSStore
from .intent_atoms.providers import get_provider, LLMProvider

__all__ = [
    "IntentAtomsEngineV3",
    "IntentAtomsEngineV2",
    "IntentAtomsEngine",
    "Atom",
    "CachedQuery",
    "QueryResult",
    "CacheStats",
    "FAISSStore",
    "get_provider",
    "LLMProvider",
]
