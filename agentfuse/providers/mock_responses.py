"""
Mock response objects for cache hits.

These mimic the real OpenAI and Anthropic SDK response objects closely enough
that downstream code (LangChain callbacks, user attribute access, .model_dump())
does not crash.
"""

import json
import time


class _DictMixin:
    """Provides .dict(), .model_dump(), .json() for compatibility."""

    def _to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, _DictMixin):
                result[key] = value._to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v._to_dict() if isinstance(v, _DictMixin) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def dict(self):
        return self._to_dict()

    def model_dump(self):
        return self._to_dict()

    def json(self, **kwargs):
        return json.dumps(self._to_dict(), **kwargs)


class MockOpenAIUsage(_DictMixin):
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0


class MockOpenAIMessage(_DictMixin):
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"
        self.tool_calls = None
        self.function_call = None
        self.refusal = None


class MockOpenAIChoice(_DictMixin):
    def __init__(self, content: str):
        self.message = MockOpenAIMessage(content)
        self.finish_reason = "stop"
        self.index = 0
        self.logprobs = None


class MockOpenAIResponse(_DictMixin):
    """Mimics openai.types.chat.ChatCompletion."""

    def __init__(self, content: str, model: str):
        self.id = "cache_hit"
        self.object = "chat.completion"
        self.created = int(time.time())
        self.model = model
        self.choices = [MockOpenAIChoice(content)]
        self.usage = MockOpenAIUsage()
        self.system_fingerprint = None
        self.service_tier = None


class MockAnthropicUsage(_DictMixin):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0


class MockAnthropicTextBlock(_DictMixin):
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockAnthropicResponse(_DictMixin):
    """Mimics anthropic.types.Message."""

    def __init__(self, content: str, model: str):
        self.id = "cache_hit"
        self.type = "message"
        self.role = "assistant"
        self.model = model
        self.content = [MockAnthropicTextBlock(content)]
        self.stop_reason = "end_turn"
        self.stop_sequence = None
        self.usage = MockAnthropicUsage()
