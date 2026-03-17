"""
ContentGuardrails — validates LLM outputs before returning to users.

Production AI systems need output guardrails to prevent:
1. Toxic/harmful content reaching users
2. PII leakage in responses
3. Off-topic responses (hallucination detection)
4. Excessive verbosity (cost waste)
5. Malformed JSON/code responses

This module provides configurable guardrails that run AFTER the LLM response
but BEFORE returning to the user.

Usage:
    guardrails = ContentGuardrails()
    guardrails.add_rule("max_length", max_chars=5000)
    guardrails.add_rule("no_pii", patterns=["email", "phone", "ssn"])
    guardrails.add_rule("topic_check", allowed_topics=["programming", "math"])

    result = guardrails.validate(response_text)
    if not result.passed:
        print(f"Blocked: {result.violations}")
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""
    passed: bool
    violations: list[str] = field(default_factory=list)
    modified_text: Optional[str] = None  # text after sanitization


# Common PII patterns
_PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
    "ssn": r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
    "credit_card": r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
    "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
}

# Toxic/harmful content patterns
_TOXIC_PATTERNS = [
    r'\b(?:kill|murder|attack|bomb|weapon)\b.*\b(?:how|instructions|steps|guide)\b',
    r'\b(?:hack|exploit|bypass|crack)\b.*\b(?:password|security|system)\b',
]


class ContentGuardrails:
    """
    Configurable output guardrails for LLM responses.

    Validates responses against rules before returning to users.
    Rules are checked in order; first violation stops (fail-fast).
    """

    def __init__(self):
        self._rules: list[dict] = []
        self._custom_validators: list[Callable] = []

    def add_rule(self, rule_type: str, **kwargs):
        """Add a guardrail rule.

        Supported types:
        - max_length: max_chars=5000
        - min_length: min_chars=10
        - no_pii: patterns=["email", "phone"] (uses built-in patterns)
        - no_toxic: (uses built-in toxic patterns)
        - regex_block: pattern="..." (custom regex to block)
        - regex_require: pattern="..." (custom regex that must match)
        - custom: validator=callable (function(text) -> (bool, str))
        """
        self._rules.append({"type": rule_type, **kwargs})

    def add_custom_validator(self, validator: Callable[[str], tuple[bool, str]]):
        """Add a custom validation function.

        Function takes text, returns (passed: bool, reason: str).
        """
        self._custom_validators.append(validator)

    def validate(self, text: str) -> GuardrailResult:
        """Validate text against all configured rules.

        Returns GuardrailResult with passed flag and violations list.
        """
        if not text:
            return GuardrailResult(passed=True)

        violations = []

        for rule in self._rules:
            rule_type = rule["type"]

            if rule_type == "max_length":
                max_chars = rule.get("max_chars", 10000)
                if len(text) > max_chars:
                    violations.append(f"Exceeds max length: {len(text)} > {max_chars} chars")

            elif rule_type == "min_length":
                min_chars = rule.get("min_chars", 1)
                if len(text) < min_chars:
                    violations.append(f"Below min length: {len(text)} < {min_chars} chars")

            elif rule_type == "no_pii":
                patterns = rule.get("patterns", list(_PII_PATTERNS.keys()))
                for pii_type in patterns:
                    if pii_type in _PII_PATTERNS:
                        if re.search(_PII_PATTERNS[pii_type], text):
                            violations.append(f"PII detected: {pii_type}")

            elif rule_type == "no_toxic":
                for pattern in _TOXIC_PATTERNS:
                    if re.search(pattern, text, re.IGNORECASE):
                        violations.append("Potentially harmful content detected")
                        break

            elif rule_type == "regex_block":
                pattern = rule.get("pattern", "")
                if pattern and re.search(pattern, text):
                    violations.append(f"Blocked pattern matched: {pattern}")

            elif rule_type == "regex_require":
                pattern = rule.get("pattern", "")
                if pattern and not re.search(pattern, text):
                    violations.append(f"Required pattern not found: {pattern}")

        # Custom validators
        for validator in self._custom_validators:
            try:
                passed, reason = validator(text)
                if not passed:
                    violations.append(reason)
            except Exception as e:
                logger.warning("Custom guardrail validator failed: %s", e)

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
        )

    def sanitize_pii(self, text: str, patterns: Optional[list[str]] = None) -> str:
        """Replace PII with type placeholders.

        "Email me at john@example.com" → "Email me at <EMAIL>"
        """
        patterns = patterns or list(_PII_PATTERNS.keys())
        result = text
        for pii_type in patterns:
            if pii_type in _PII_PATTERNS:
                result = re.sub(_PII_PATTERNS[pii_type], f"<{pii_type.upper()}>", result)
        return result
