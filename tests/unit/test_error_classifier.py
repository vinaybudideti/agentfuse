"""
Phase 5 — Unified error classifier tests.
"""

from agentfuse.core.error_classifier import classify_error, ClassifiedError


class _MockException(Exception):
    """Mock exception with status_code attribute."""
    def __init__(self, msg="", status_code=None):
        super().__init__(msg)
        self.status_code = status_code
        self.__module__ = "test"


class _MockOpenAIRateLimitError(_MockException):
    __qualname__ = "RateLimitError"
    def __init__(self, msg=""):
        super().__init__(msg, status_code=429)


class _MockOpenAIAuthError(_MockException):
    __qualname__ = "AuthenticationError"
    def __init__(self, msg=""):
        super().__init__(msg, status_code=401)


class _MockOpenAIBadRequest(_MockException):
    __qualname__ = "BadRequestError"
    def __init__(self, msg=""):
        super().__init__(msg, status_code=400)


class _MockOpenAIServerError(_MockException):
    __qualname__ = "InternalServerError"
    def __init__(self, msg=""):
        super().__init__(msg, status_code=500)


class _MockOpenAITimeout(_MockException):
    __qualname__ = "APITimeoutError"


class _MockAnthropicOverloaded(_MockException):
    __qualname__ = "OverloadedError"
    def __init__(self, msg=""):
        super().__init__(msg, status_code=529)


# Override __name__ for type detection
for cls in [_MockOpenAIRateLimitError, _MockOpenAIAuthError, _MockOpenAIBadRequest,
            _MockOpenAIServerError, _MockOpenAITimeout, _MockAnthropicOverloaded]:
    cls.__name__ = cls.__qualname__


def test_openai_rate_limit_is_retryable():
    exc = _MockOpenAIRateLimitError("Rate limit exceeded")
    result = classify_error(exc, "openai")
    assert result.retryable is True
    assert result.error_type == "rate_limit"


def test_openai_insufficient_quota_NOT_retryable():
    """CRITICAL: billing 429 != rate limit 429."""
    exc = _MockOpenAIRateLimitError("You exceeded your current quota. insufficient_quota")
    result = classify_error(exc, "openai")
    assert result.retryable is False
    assert result.error_type == "insufficient_quota"


def test_anthropic_overloaded_529_is_retryable():
    """Anthropic's unique 529 OverloadedError is always retryable."""
    exc = _MockAnthropicOverloaded("Overloaded")
    result = classify_error(exc, "anthropic")
    assert result.retryable is True
    assert result.error_type == "overloaded"
    assert result.status_code == 529


def test_context_window_400_NOT_retryable():
    exc = _MockOpenAIBadRequest("context window exceeded")
    result = classify_error(exc, "openai")
    assert result.retryable is False
    assert result.error_type == "bad_request"


def test_auth_401_NOT_retryable():
    exc = _MockOpenAIAuthError("Invalid API key")
    result = classify_error(exc, "openai")
    assert result.retryable is False
    assert result.error_type == "auth"


def test_server_500_is_retryable():
    exc = _MockOpenAIServerError("Internal server error")
    result = classify_error(exc, "openai")
    assert result.retryable is True
    assert result.error_type == "server"


def test_timeout_is_retryable():
    exc = _MockOpenAITimeout("Request timed out")
    result = classify_error(exc, "openai")
    assert result.retryable is True
    assert result.error_type == "timeout"


def test_unknown_exception_defaults_retryable():
    exc = Exception("Something unknown")
    result = classify_error(exc, "unknown-provider")
    assert result.retryable is True
    assert result.error_type == "unknown"


def test_anthropic_rate_limit_retryable():
    """Anthropic 429 rate limit is retryable."""
    class RateLimitError(_MockException):
        pass
    RateLimitError.__name__ = "RateLimitError"
    exc = RateLimitError("Rate limited", status_code=429)
    result = classify_error(exc, "anthropic")
    assert result.retryable is True
    assert result.error_type == "rate_limit"


def test_anthropic_auth_not_retryable():
    class AuthenticationError(_MockException):
        pass
    AuthenticationError.__name__ = "AuthenticationError"
    exc = AuthenticationError("Bad key", status_code=401)
    result = classify_error(exc, "anthropic")
    assert result.retryable is False


def test_google_rate_limit():
    class ClientError(_MockException):
        pass
    ClientError.__name__ = "ClientError"
    exc = ClientError("Too many requests", status_code=429)
    result = classify_error(exc, "gemini")
    assert result.retryable is True


def test_google_auth_not_retryable():
    class ClientError(_MockException):
        pass
    ClientError.__name__ = "ClientError"
    exc = ClientError("Unauthorized", status_code=401)
    result = classify_error(exc, "gemini")
    assert result.retryable is False


def test_google_server_error_retryable():
    class ServerError(_MockException):
        pass
    ServerError.__name__ = "ServerError"
    exc = ServerError("Internal error", status_code=500)
    result = classify_error(exc, "gemini")
    assert result.retryable is True


def test_classified_error_has_message():
    exc = _MockOpenAIServerError("detailed error message")
    result = classify_error(exc, "openai")
    assert "detailed error message" in result.message


def test_openai_connection_error_retryable():
    class APIConnectionError(_MockException):
        pass
    APIConnectionError.__name__ = "APIConnectionError"
    exc = APIConnectionError("Connection refused")
    result = classify_error(exc, "openai")
    assert result.retryable is True
    assert result.error_type == "connection"


def test_agentfuse_retry_decorator():
    """agentfuse_retry decorator should work."""
    from agentfuse.core.error_classifier import agentfuse_retry
    decorator = agentfuse_retry(max_attempts=2, max_wait=5, provider="openai")
    assert callable(decorator)


def test_google_bad_request_not_retryable():
    """Google 400 bad request is not retryable."""
    class ClientError(_MockException):
        pass
    ClientError.__name__ = "ClientError"
    exc = ClientError("Bad request", status_code=400)
    result = classify_error(exc, "gemini")
    assert result.retryable is False


def test_anthropic_bad_request_not_retryable():
    """Anthropic 400 context window exceeded is not retryable."""
    class BadRequestError(_MockException):
        pass
    BadRequestError.__name__ = "BadRequestError"
    exc = BadRequestError("context too long", status_code=400)
    result = classify_error(exc, "anthropic")
    assert result.retryable is False


def test_anthropic_server_error_retryable():
    """Anthropic 500 is retryable."""
    class InternalServerError(_MockException):
        pass
    InternalServerError.__name__ = "InternalServerError"
    exc = InternalServerError("Server error", status_code=500)
    result = classify_error(exc, "anthropic")
    assert result.retryable is True


def test_openai_status_408_retryable():
    """OpenAI 408 (request timeout) is retryable."""
    exc = _MockException("Request timeout", status_code=408)
    result = classify_error(exc, "openai")
    assert result.retryable is True
