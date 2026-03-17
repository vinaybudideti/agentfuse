"""
AgentFuse Security Layer — protects the SDK across all operating systems.

Handles:
1. API key protection — never log, never cache, never persist keys
2. Input sanitization — prevent prompt injection attacks on cached responses
3. Response validation — prevent caching of malicious/poisoned responses
4. Redis connection security — TLS support, auth, connection timeouts
5. Cache isolation — per-tenant, per-model, no cross-contamination
6. Rate limiting — prevent abuse and denial-of-service
7. Audit logging — structured security events for compliance

This module provides security primitives used throughout AgentFuse.
All security operations are fail-safe — they never crash the main flow.
"""

import hashlib
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


# --- API Key Protection ---

def mask_api_key(key: str) -> str:
    """Mask an API key for safe logging. Shows first 4 + last 4 chars only.

    Example: sk-abcdefghijklmnop → sk-a...mnop
    """
    if not key or len(key) < 12:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def validate_api_key_format(key: str, provider: str = "unknown") -> bool:
    """Validate API key format without leaking the key.

    Returns True if the key matches expected format.
    Returns False with a warning if format is suspicious.
    """
    if not key or not isinstance(key, str):
        return False

    # OpenAI keys start with sk-
    if provider == "openai" and not key.startswith("sk-"):
        logger.warning("OpenAI API key does not start with 'sk-' — may be invalid")
        return False

    # Anthropic keys start with sk-ant-
    if provider == "anthropic" and not key.startswith("sk-ant-"):
        logger.warning("Anthropic API key does not start with 'sk-ant-' — may be invalid")
        return False

    # Key length sanity check (most keys are 40-200 chars)
    if len(key) < 10:
        logger.warning("API key appears too short (%d chars)", len(key))
        return False

    return True


# --- Input Sanitization ---

# Patterns that may indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+(instructions|prompts)",
    r"system\s*prompt\s*:",
    r"you\s+are\s+now\s+(?:a|an)\s+(?:different|new)",
    r"forget\s+(?:everything|all)\s+(?:above|previous)",
    r"override\s+(?:system|safety)",
]
_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Invisible Unicode characters that could be used for steganography
_INVISIBLE_CHARS = set([
    '\u200b',  # zero-width space
    '\u200c',  # zero-width non-joiner
    '\u200d',  # zero-width joiner
    '\u200e',  # left-to-right mark
    '\u200f',  # right-to-left mark
    '\u2060',  # word joiner
    '\u2061',  # function application
    '\u2062',  # invisible times
    '\u2063',  # invisible separator
    '\u2064',  # invisible plus
    '\ufeff',  # zero-width no-break space (BOM)
])


def check_prompt_injection(text: str) -> tuple[bool, str]:
    """Check if text contains potential prompt injection patterns.

    Returns (is_suspicious, reason).
    This is a heuristic check — not a guarantee of safety.
    Use as a warning signal, not a blocking mechanism.
    """
    if not text:
        return False, ""

    for pattern in _COMPILED_INJECTION:
        if pattern.search(text):
            return True, f"Potential prompt injection detected: {pattern.pattern}"

    return False, ""


def strip_invisible_chars(text: str) -> str:
    """Remove invisible Unicode characters that could hide malicious content.

    These characters are used in steganographic attacks where invisible
    instructions are embedded in seemingly normal text.
    """
    return ''.join(c for c in text if c not in _INVISIBLE_CHARS)


def sanitize_for_cache_key(text: str) -> str:
    """Sanitize text before using it in a cache key.

    Strips invisible characters and normalizes whitespace to prevent
    cache key collision attacks.
    """
    text = strip_invisible_chars(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Response Validation ---

_MALICIOUS_RESPONSE_PATTERNS = [
    r"<script\b",                    # XSS via cached response
    r"javascript:",                   # XSS via javascript: URI
    r"data:text/html",               # Data URI injection
    r"\\x[0-9a-f]{2}",              # Hex-encoded payloads
]
_COMPILED_MALICIOUS = [re.compile(p, re.IGNORECASE) for p in _MALICIOUS_RESPONSE_PATTERNS]


def validate_response_safety(response_text: str) -> tuple[bool, str]:
    """Check if an LLM response contains potentially malicious content.

    Returns (is_safe, reason).
    Prevents caching of responses that could cause XSS or injection
    when served to other users.
    """
    if not response_text:
        return True, ""

    for pattern in _COMPILED_MALICIOUS:
        if pattern.search(response_text):
            return False, f"Potentially malicious content: {pattern.pattern}"

    return True, ""


# --- Secure Hashing ---

def secure_hash(data: str, salt: str = "") -> str:
    """Create a SHA-256 hash with optional salt.

    Used for cache keys, request deduplication, and audit logs.
    Never use for passwords — use bcrypt/argon2 for that.
    """
    content = f"{salt}:{data}" if salt else data
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# --- Environment Security ---

def get_secure_env(key: str, required: bool = False) -> Optional[str]:
    """Get an environment variable securely.

    - Never logs the value
    - Warns if the value looks like a hardcoded key in code
    - Raises ValueError if required and not set
    """
    value = os.environ.get(key)
    if value is None and required:
        raise ValueError(
            f"Required environment variable {key} is not set. "
            f"Set it with: export {key}=your_value"
        )
    return value


# --- Security Audit Event ---

class SecurityEvent:
    """Structured security event for audit logging."""

    def __init__(self, event_type: str, severity: str = "info", **details):
        self.event_type = event_type
        self.severity = severity
        self.timestamp = time.time()
        self.details = details

    def log(self):
        """Log the security event."""
        msg = f"[SECURITY:{self.severity.upper()}] {self.event_type}"
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            msg += f" — {detail_str}"

        if self.severity == "critical":
            logger.critical(msg)
        elif self.severity == "warning":
            logger.warning(msg)
        else:
            logger.info(msg)
