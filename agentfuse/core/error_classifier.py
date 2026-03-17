"""
Unified error classifier across OpenAI, Anthropic, and Google GenAI.

CRITICAL RULES:
- OpenAI 429 with "insufficient_quota" → NOT retryable (billing issue)
- Anthropic OverloadedError (529) → ALWAYS retryable
- Context window exceeded (400) → NOT retryable (fix the request)
- Auth 401 → NOT retryable (fix the key)
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedError:
    """Result of classifying an API error."""
    error_type: str        # e.g. "rate_limit", "auth", "server", "timeout"
    retryable: bool
    status_code: Optional[int] = None
    provider: str = "unknown"
    message: str = ""
    retry_after: Optional[float] = None  # seconds to wait before retry

    @property
    def counts_for_circuit_breaker(self) -> bool:
        """Rate limits (429) should NOT count toward circuit breaker failures.

        A 429 indicates the provider is healthy but busy — tripping the circuit
        breaker would prevent requests even after the rate limit window passes.
        Only server errors (5xx) and connection failures should count.
        """
        if self.error_type == "rate_limit":
            return False
        if self.status_code and 400 <= self.status_code < 500:
            return False  # Client errors (auth, bad request) aren't provider failures
        return True

    @staticmethod
    def extract_retry_after(exc: Exception) -> Optional[float]:
        """Extract Retry-After header value from exception, if present.

        Handles both numeric seconds ("30") and HTTP-date format
        ("Wed, 21 Oct 2015 07:28:00 GMT").
        """
        response = getattr(exc, "response", None)
        if response:
            headers = getattr(response, "headers", {})
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after:
                # Try numeric first (most common)
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass
                # Try HTTP-date format
                try:
                    from email.utils import parsedate_to_datetime
                    import time
                    target = parsedate_to_datetime(retry_after)
                    delta = target.timestamp() - time.time()
                    return max(0.0, delta)
                except Exception:
                    pass
        return None


def classify_error(exc: Exception, provider: str = "unknown") -> ClassifiedError:
    """
    Classify any provider exception into a unified format.

    Returns ClassifiedError with retryable flag.
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc).lower()

    # --- OpenAI ---
    if provider == "openai" or _is_openai_error(exc):
        return _classify_openai(exc, exc_type, exc_msg)

    # --- Anthropic ---
    if provider == "anthropic" or _is_anthropic_error(exc):
        return _classify_anthropic(exc, exc_type, exc_msg)

    # --- Google GenAI ---
    if provider in ("gemini", "google", "gcp.gemini") or _is_google_error(exc):
        return _classify_google(exc, exc_type, exc_msg)

    # --- httpx errors ---
    if _is_httpx_error(exc):
        return _classify_httpx(exc, exc_type)

    # --- Unknown ---
    logger.warning("Unknown exception type %s from provider %s — defaulting retryable=True", exc_type, provider)
    return ClassifiedError(
        error_type="unknown",
        retryable=True,
        provider=provider,
        message=str(exc),
    )


def _classify_openai(exc, exc_type, exc_msg) -> ClassifiedError:
    status = getattr(exc, "status_code", None)

    if exc_type == "RateLimitError" or status == 429:
        retry_after = ClassifiedError.extract_retry_after(exc)
        # CRITICAL: "insufficient_quota" is NOT retryable
        if "insufficient_quota" in exc_msg:
            return ClassifiedError("insufficient_quota", retryable=False, status_code=429, provider="openai", message=str(exc))
        return ClassifiedError("rate_limit", retryable=True, status_code=429, provider="openai", message=str(exc), retry_after=retry_after)

    if exc_type == "AuthenticationError" or status == 401:
        return ClassifiedError("auth", retryable=False, status_code=401, provider="openai", message=str(exc))

    if exc_type == "BadRequestError" or status == 400:
        return ClassifiedError("bad_request", retryable=False, status_code=400, provider="openai", message=str(exc))

    if exc_type == "InternalServerError" or (status and status >= 500):
        return ClassifiedError("server", retryable=True, status_code=status or 500, provider="openai", message=str(exc))

    if exc_type == "APITimeoutError":
        return ClassifiedError("timeout", retryable=True, provider="openai", message=str(exc))

    if exc_type == "APIConnectionError":
        return ClassifiedError("connection", retryable=True, provider="openai", message=str(exc))

    # Status-based fallback
    if status:
        if status in (408, 409):
            return ClassifiedError("retryable_client", retryable=True, status_code=status, provider="openai", message=str(exc))
        if status in (400, 401, 403, 404, 413, 422):
            return ClassifiedError("client_error", retryable=False, status_code=status, provider="openai", message=str(exc))

    return ClassifiedError("unknown_openai", retryable=True, provider="openai", message=str(exc))


def _classify_anthropic(exc, exc_type, exc_msg) -> ClassifiedError:
    status = getattr(exc, "status_code", None)

    retry_after = ClassifiedError.extract_retry_after(exc)

    if exc_type == "OverloadedError" or status == 529:
        return ClassifiedError("overloaded", retryable=True, status_code=529, provider="anthropic", message=str(exc), retry_after=retry_after)

    if exc_type == "RateLimitError" or status == 429:
        return ClassifiedError("rate_limit", retryable=True, status_code=429, provider="anthropic", message=str(exc), retry_after=retry_after)

    if exc_type == "AuthenticationError" or status == 401:
        return ClassifiedError("auth", retryable=False, status_code=401, provider="anthropic", message=str(exc))

    if exc_type == "BadRequestError" or status == 400:
        return ClassifiedError("bad_request", retryable=False, status_code=400, provider="anthropic", message=str(exc))

    if exc_type == "InternalServerError" or (status and status >= 500):
        return ClassifiedError("server", retryable=True, status_code=status or 500, provider="anthropic", message=str(exc))

    if exc_type == "APITimeoutError":
        return ClassifiedError("timeout", retryable=True, provider="anthropic", message=str(exc))

    if exc_type == "APIConnectionError":
        return ClassifiedError("connection", retryable=True, provider="anthropic", message=str(exc))

    return ClassifiedError("unknown_anthropic", retryable=True, provider="anthropic", message=str(exc))


def _classify_google(exc, exc_type, exc_msg) -> ClassifiedError:
    status = getattr(exc, "code", getattr(exc, "status_code", None))

    if exc_type == "ClientError":
        if status == 429:
            return ClassifiedError("rate_limit", retryable=True, status_code=429, provider="gemini", message=str(exc))
        if status == 401:
            return ClassifiedError("auth", retryable=False, status_code=401, provider="gemini", message=str(exc))
        if status == 400:
            return ClassifiedError("bad_request", retryable=False, status_code=400, provider="gemini", message=str(exc))
        return ClassifiedError("client_error", retryable=False, status_code=status, provider="gemini", message=str(exc))

    if exc_type == "ServerError":
        return ClassifiedError("server", retryable=True, status_code=status or 500, provider="gemini", message=str(exc))

    return ClassifiedError("unknown_gemini", retryable=True, provider="gemini", message=str(exc))


def _classify_httpx(exc, exc_type) -> ClassifiedError:
    if exc_type == "TimeoutException" or "timeout" in exc_type.lower():
        return ClassifiedError("timeout", retryable=True, provider="httpx", message=str(exc))
    if exc_type == "ConnectError" or "connect" in exc_type.lower():
        return ClassifiedError("connection", retryable=True, provider="httpx", message=str(exc))
    return ClassifiedError("httpx_error", retryable=True, provider="httpx", message=str(exc))


def _is_openai_error(exc) -> bool:
    return type(exc).__module__.startswith("openai") if hasattr(type(exc), "__module__") else False


def _is_anthropic_error(exc) -> bool:
    return type(exc).__module__.startswith("anthropic") if hasattr(type(exc), "__module__") else False


def _is_google_error(exc) -> bool:
    mod = getattr(type(exc), "__module__", "")
    return "google" in mod or "genai" in mod


def _is_httpx_error(exc) -> bool:
    return type(exc).__module__.startswith("httpx") if hasattr(type(exc), "__module__") else False


def agentfuse_retry(max_attempts: int = 5, max_wait: int = 60, provider: str = "unknown"):
    """
    Decorator: retry with exponential backoff, using classify_error for predicate.

    Uses tenacity for robust retry logic.
    """
    from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

    def _should_retry(exc):
        classified = classify_error(exc, provider)
        return classified.retryable

    return retry(
        retry=retry_if_exception(_should_retry),
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1.0, min=1, max=max_wait),
        reraise=True,
    )
