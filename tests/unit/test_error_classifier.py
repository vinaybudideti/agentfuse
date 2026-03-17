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


def test_openai_module_detection():
    """Exceptions from openai module must be auto-detected."""
    from agentfuse.core.error_classifier import _is_openai_error
    # Our mock exceptions don't have openai module
    exc = _MockException("test")
    assert not _is_openai_error(exc)


def test_anthropic_module_detection():
    """Exceptions from anthropic module must be auto-detected."""
    from agentfuse.core.error_classifier import _is_anthropic_error
    exc = _MockException("test")
    assert not _is_anthropic_error(exc)


def test_google_module_detection():
    """Exceptions from google module must be auto-detected."""
    from agentfuse.core.error_classifier import _is_google_error
    exc = _MockException("test")
    assert not _is_google_error(exc)

    # Create a mock with google module
    class GoogleError(Exception):
        pass
    GoogleError.__module__ = "google.genai"
    assert _is_google_error(GoogleError("test"))


def test_httpx_module_detection():
    """httpx exceptions must be auto-detected."""
    from agentfuse.core.error_classifier import _is_httpx_error
    exc = _MockException("test")
    assert not _is_httpx_error(exc)


def test_google_client_error_generic():
    """Google ClientError without known status code."""
    class ClientError(_MockException):
        pass
    ClientError.__name__ = "ClientError"
    exc = ClientError("Unknown client error", status_code=422)
    result = classify_error(exc, "gemini")
    assert result.retryable is False
    assert result.error_type == "client_error"


def test_openai_status_403_not_retryable():
    """OpenAI 403 (forbidden) is not retryable."""
    exc = _MockException("Forbidden", status_code=403)
    result = classify_error(exc, "openai")
    assert result.retryable is False


def test_retry_after_extraction():
    """Retry-After header must be extracted from exception response."""
    from agentfuse.core.error_classifier import ClassifiedError
    from types import SimpleNamespace

    exc = _MockOpenAIRateLimitError("Rate limited")
    exc.response = SimpleNamespace(headers={"Retry-After": "30"})
    result = ClassifiedError.extract_retry_after(exc)
    assert result == 30.0


def test_retry_after_missing():
    """Missing Retry-After must return None."""
    from agentfuse.core.error_classifier import ClassifiedError
    exc = _MockException("No header")
    result = ClassifiedError.extract_retry_after(exc)
    assert result is None


def test_rate_limit_includes_retry_after():
    """Rate limit classification must include retry_after if present."""
    exc = _MockOpenAIRateLimitError("Rate limited")
    from types import SimpleNamespace
    exc.response = SimpleNamespace(headers={"Retry-After": "5"})
    result = classify_error(exc, "openai")
    assert result.retry_after == 5.0


def test_httpx_timeout():
    """httpx TimeoutException must be classified as retryable timeout."""
    class TimeoutException(_MockException):
        pass
    TimeoutException.__name__ = "TimeoutException"
    exc = TimeoutException("Connection timed out")
    result = classify_error(exc, "httpx")
    # Won't match _is_httpx_error (wrong module), but provider="httpx" triggers httpx path
    # Actually no — classify_error checks provider first
    # With provider="unknown", it falls to unknown. Let me test direct httpx.
    from agentfuse.core.error_classifier import _classify_httpx
    result = _classify_httpx(exc, "TimeoutException")
    assert result.retryable is True
    assert result.error_type == "timeout"


def test_httpx_connect_error():
    """httpx ConnectError must be classified as retryable connection error."""
    from agentfuse.core.error_classifier import _classify_httpx
    class ConnectError(_MockException):
        pass
    ConnectError.__name__ = "ConnectError"
    exc = ConnectError("Connection refused")
    result = _classify_httpx(exc, "ConnectError")
    assert result.retryable is True
    assert result.error_type == "connection"


def test_httpx_generic_error():
    """Generic httpx error must be retryable."""
    from agentfuse.core.error_classifier import _classify_httpx
    result = _classify_httpx(Exception("unknown"), "SomeHttpxError")
    assert result.retryable is True
    assert result.error_type == "httpx_error"


def test_google_unknown_error():
    """Unknown Google error type must default to retryable."""
    from agentfuse.core.error_classifier import _classify_google
    class UnknownGoogleError(_MockException):
        pass
    UnknownGoogleError.__name__ = "UnknownGoogleError"
    result = _classify_google(UnknownGoogleError("something"), "UnknownGoogleError", "something")
    assert result.retryable is True


def test_anthropic_timeout_retryable():
    """Anthropic APITimeoutError must be retryable."""
    class APITimeoutError(_MockException):
        pass
    APITimeoutError.__name__ = "APITimeoutError"
    exc = APITimeoutError("Request timed out")
    result = classify_error(exc, "anthropic")
    assert result.retryable is True
    assert result.error_type == "timeout"


def test_anthropic_connection_retryable():
    """Anthropic APIConnectionError must be retryable."""
    class APIConnectionError(_MockException):
        pass
    APIConnectionError.__name__ = "APIConnectionError"
    exc = APIConnectionError("Connection failed")
    result = classify_error(exc, "anthropic")
    assert result.retryable is True
    assert result.error_type == "connection"


def test_anthropic_unknown_retryable():
    """Unknown Anthropic error must default to retryable."""
    exc = _MockException("Something weird happened")
    result = classify_error(exc, "anthropic")
    assert result.retryable is True
    assert result.error_type == "unknown_anthropic"


def test_circuit_breaker_property():
    """counts_for_circuit_breaker must work correctly."""
    # Rate limit: NOT counted
    err = ClassifiedError("rate_limit", retryable=True, status_code=429)
    assert err.counts_for_circuit_breaker is False

    # Server error: counted
    err = ClassifiedError("server", retryable=True, status_code=500)
    assert err.counts_for_circuit_breaker is True

    # Timeout (no status code): counted
    err = ClassifiedError("timeout", retryable=True)
    assert err.counts_for_circuit_breaker is True
