"""
Cache key builder — deterministic SHA-256 keys for L1 exact match,
plus semantic content extraction for L2 similarity search.

L1 keys include model, messages, temperature, tools, tenant_id to prevent
cross-model contamination and ensure cache isolation.

L2 extracts only user-facing content for embedding, with metadata filters
applied post-search to prevent cross-model/cross-tool results.
"""

import hashlib
import json
from typing import Optional


def _extract_text(content) -> str:
    """Extract text from various content formats (string, list of blocks, None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", block.get("content", "")))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


def build_l1_cache_key(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    tools: Optional[list[dict]] = None,
    model_version: Optional[str] = None,
    tenant_id: Optional[str] = None,
    response_format: Optional[dict] = None,
) -> str:
    """
    Deterministic SHA-256 key. Model is always first component so
    different models ALWAYS produce different hashes (prevents contamination).
    """
    key_parts: dict = {
        "model": model,
        "messages": messages,
        "temperature": round(temperature, 4),
    }
    if tools:
        key_parts["tools"] = sorted(
            tools, key=lambda t: t.get("function", {}).get("name", "")
        )
    if model_version:
        key_parts["model_version"] = model_version
    if tenant_id:
        key_parts["tenant_id"] = tenant_id
    if response_format:
        key_parts["response_format"] = response_format

    raw = json.dumps(key_parts, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"agentfuse:v2:cache:{digest}"


def extract_semantic_content(messages: list[dict], max_user_messages: int = 5) -> str:
    """Extract user-facing content for semantic embedding (L2 cache).

    Per ContextCache paper (arxiv 2506.22791): embed the last 3-5 user messages
    (not assistant messages) for multi-turn context. Single-turn uses only the
    latest user message. This yields 10.9% precision improvement over single-turn.

    Negation prefix: prepends "NOT:" when user message contains negation to help
    the embedding model distinguish "I want X" from "I don't want X" — addresses
    the negation blindness false positive issue (InfoQ banking case study).
    """
    user_contents = []
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), (str, list)):
            text = _extract_text(m.get("content", ""))
            # Add negation signal to help embedding model differentiate
            if _has_negation(text):
                text = "NOT: " + text
            user_contents.append(text)
    # Take last N user messages for multi-turn context
    recent = user_contents[-max_user_messages:]
    return " ".join(recent)


def _has_negation(text: str) -> bool:
    """Check if text contains negation words that change semantic meaning.

    Addresses the negation blindness problem where "I want to cancel" and
    "I don't want to cancel" produce very similar embeddings.
    """
    lower = text.lower()
    negation_words = {"not", "don't", "doesn't", "didn't", "won't", "can't",
                      "cannot", "shouldn't", "wouldn't", "never", "no",
                      "without", "neither", "nor", "nothing", "nobody"}
    words = set(lower.split())
    return bool(words & negation_words)


def build_l2_metadata_filter(model: str, tools: Optional[list] = None) -> dict:
    """
    Post-filter for FAISS results: after semantic search returns candidates,
    ONLY accept entries where model prefix matches.
    Never serve OpenAI results for Anthropic queries.
    """
    if model.startswith(("gpt", "o1", "o3", "o4")):
        prefix = "openai"
    elif model.startswith("claude"):
        prefix = "anthropic"
    elif model.startswith("gemini"):
        prefix = "gemini"
    elif model.startswith("mistral"):
        prefix = "mistral"
    elif model.startswith("grok"):
        prefix = "xai"
    elif model.startswith(("llama", "meta-llama")):
        prefix = "meta"
    else:
        prefix = model.split("/")[0] if "/" in model else model.split("-")[0]

    return {
        "model_prefix": prefix,
        "has_tools": bool(tools),
    }


# Backward compatibility — used by CacheMiddleware for FAISS semantic embedding.
# This produces a READABLE text string (not a hash) because the current
# CacheMiddleware embeds this string for cosine similarity search.
# The SHA-256 build_l1_cache_key is for future L1 exact-match (Phase 2).
def build_cache_key(messages: list, model: str) -> str:
    """
    Build a deterministic, embeddable cache key from messages + model.

    Returns a human-readable string suitable for sentence-transformer embedding.
    Includes model name to prevent cross-model contamination in FAISS search.
    """
    parts = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "unknown")
            text = _extract_text(m.get("content"))
            parts.append(f"[{role}]: {text}")
        elif isinstance(m, str):
            parts.append(f"[user]: {m}")
    key = f"model={model}\n" + "\n".join(parts)
    return key
