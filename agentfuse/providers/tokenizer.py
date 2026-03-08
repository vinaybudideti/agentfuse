# TokenCounterAdapter: tiktoken for OpenAI + Anthropic provider-specific token counting


class TokenCounterAdapter:
    def count_tokens(self, text, model):
        if model.startswith("gpt") or model == "o3":
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))

        if model.startswith("claude") or model.startswith("gemini"):
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))

        return len(text) // 4

    def count_messages_tokens(self, messages, model):
        total = 0
        for message in messages:
            total += self.count_tokens(message["content"], model)
            total += 4  # role/format overhead per message
        return total
