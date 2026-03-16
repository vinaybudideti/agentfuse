"""
Cache key builder — produces a deterministic string from messages that
preserves role boundaries, handles non-string content, and includes
model identity to prevent cross-model cache contamination.
"""


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


def build_cache_key(messages: list, model: str) -> str:
    """
    Build a deterministic cache key from messages + model.

    Includes role prefixes so different conversation structures
    never collide, and model name so GPT-4o and Claude responses
    are cached separately.

    Returns a single string suitable for embedding.
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
