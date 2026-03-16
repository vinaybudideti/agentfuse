"""
TokenCounterAdapter: provider-aware token counter with a 4-tier fallback chain.

FIXED in Phase 0:
- GPT-4o/o3/o4 now use o200k_base encoding (was using cl100k_base via encoding_for_model)
- Anthropic safety margin increased from 1.15x to 1.20x (prevents budget underrun)
- Gemini safety margin increased from 1.05x to 1.25x (prevents budget underrun)
- Added multimodal content block handling in count_messages
- Added reply priming tokens (3) to message count
- Fallback uses len(text) / 3.5 (more conservative than // 4)

Tier 1: Exact local tokenizer (OpenAI via tiktoken)
Tier 2: Provider API (Anthropic/Gemini count_tokens — free, not implemented yet)
Tier 3: tiktoken cl100k_base with safety margin (Anthropic/Gemini fallback)
Tier 4: len(text) / 3.5 character estimate (unknown models)
"""

import functools

import tiktoken


class TokenCounterAdapter:
    """
    Provider-aware token counter with a 4-tier fallback chain.
    """

    @functools.lru_cache(maxsize=8)
    def _get_tiktoken_encoder(self, encoding_name: str):
        return tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str, model: str) -> int:
        if not text:
            return 0
        try:
            return self._count_exact(text, model)
        except Exception:
            return self._count_fallback(text, model)

    def _count_exact(self, text: str, model: str) -> int:
        # OpenAI GPT-4o, GPT-4.1+, o1, o3, o4 models: use o200k_base encoding
        if model.startswith(("gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5", "o1", "o3", "o4")):
            enc = tiktoken.encoding_for_model("gpt-4o")  # o200k_base
            return len(enc.encode(text))

        # OpenAI GPT-4-turbo, GPT-4, GPT-3.5 (legacy): use cl100k_base encoding
        if model.startswith(("gpt-4", "gpt-3.5")):
            enc = tiktoken.encoding_for_model("gpt-4")  # cl100k_base
            return len(enc.encode(text))

        # Anthropic: tiktoken cl100k_base + 20% safety margin
        if model.startswith("claude"):
            enc = self._get_tiktoken_encoder("cl100k_base")
            return int(len(enc.encode(text)) * 1.20)

        # Gemini: tiktoken cl100k_base + 25% safety margin
        if model.startswith("gemini"):
            enc = self._get_tiktoken_encoder("cl100k_base")
            return int(len(enc.encode(text)) * 1.25)

        # Mistral: tiktoken cl100k_base + 15% safety margin
        if model.startswith("mistral"):
            enc = self._get_tiktoken_encoder("cl100k_base")
            return int(len(enc.encode(text)) * 1.15)

        # Llama / Groq / Together hosted models
        if any(model.startswith(p) for p in ("llama", "groq/", "together/")):
            enc = self._get_tiktoken_encoder("cl100k_base")
            return int(len(enc.encode(text)) * 1.10)

        # DeepSeek: tiktoken cl100k_base + 10% safety margin
        if model.startswith("deepseek"):
            enc = self._get_tiktoken_encoder("cl100k_base")
            return int(len(enc.encode(text)) * 1.10)

        # Grok / xAI
        if model.startswith("grok"):
            enc = self._get_tiktoken_encoder("cl100k_base")
            return int(len(enc.encode(text)) * 1.15)

        raise ValueError(f"Unknown model for exact counting: {model}")

    def _count_fallback(self, text: str, model: str) -> int:
        """Universal fallback: character-based estimate."""
        # ~3.5 chars/token is conservative (better to overestimate for budgets)
        return max(1, int(len(text) / 3.5))

    def count_messages(self, messages: list[dict], model: str) -> int:
        """Count tokens for a full message list including role overhead."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content, model)
            elif isinstance(content, list):  # multi-modal content blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += self.count_tokens(block.get("text", ""), model)
            total += 4  # role + format overhead per message
        total += 3  # reply priming tokens
        return total

    # Backward compatibility alias
    def count_messages_tokens(self, messages: list, model: str) -> int:
        """Backward-compatible alias for count_messages."""
        return self.count_messages(messages, model)
