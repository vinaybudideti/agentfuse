"""
TokenCounterAdapter: tiktoken for OpenAI, provider-specific estimation
for Anthropic and Google models.

OpenAI: exact via tiktoken
Anthropic: cl100k_base * 1.15 correction factor (Anthropic tokenizer runs ~15% higher)
Google: cl100k_base * 1.05 correction factor (Gemini tokenizer is closer to cl100k)
Unknown: len(text) // 4 fallback
"""

from agentfuse.core.keys import _extract_text


# Correction factors based on empirical comparison of tokenizers.
# Anthropic's tokenizer produces ~15% more tokens than cl100k_base on average.
# Gemini's tokenizer produces ~5% more tokens than cl100k_base on average.
_ANTHROPIC_CORRECTION = 1.15
_GEMINI_CORRECTION = 1.05


class TokenCounterAdapter:

    def _get_cl100k(self):
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str, model: str) -> int:
        if not text:
            return 0

        if model.startswith("gpt") or model.startswith("o3"):
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = self._get_cl100k()
            return len(enc.encode(text))

        if model.startswith("claude"):
            enc = self._get_cl100k()
            base_count = len(enc.encode(text))
            return int(base_count * _ANTHROPIC_CORRECTION)

        if model.startswith("gemini"):
            enc = self._get_cl100k()
            base_count = len(enc.encode(text))
            return int(base_count * _GEMINI_CORRECTION)

        return len(text) // 4

    def count_messages_tokens(self, messages: list, model: str) -> int:
        total = 0
        for message in messages:
            content = message.get("content", "") if isinstance(message, dict) else ""
            text = _extract_text(content)
            total += self.count_tokens(text, model)
            total += 4  # role/format overhead per message
        return total
