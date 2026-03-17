"""
Tests for ContentGuardrails — output validation for LLM responses.
"""

from agentfuse.core.guardrails import ContentGuardrails


def test_empty_text_passes():
    """Empty text must pass all guardrails."""
    g = ContentGuardrails()
    g.add_rule("max_length", max_chars=100)
    assert g.validate("").passed is True


def test_max_length_violation():
    """Text exceeding max_length must be flagged."""
    g = ContentGuardrails()
    g.add_rule("max_length", max_chars=10)
    result = g.validate("This is a long text exceeding 10 chars")
    assert result.passed is False
    assert "max length" in result.violations[0].lower()


def test_max_length_passes():
    """Text within max_length must pass."""
    g = ContentGuardrails()
    g.add_rule("max_length", max_chars=1000)
    assert g.validate("Short text").passed is True


def test_min_length_violation():
    """Text below min_length must be flagged."""
    g = ContentGuardrails()
    g.add_rule("min_length", min_chars=20)
    result = g.validate("Too short")
    assert result.passed is False


def test_pii_email_detected():
    """Email addresses must be detected as PII."""
    g = ContentGuardrails()
    g.add_rule("no_pii", patterns=["email"])
    result = g.validate("Contact me at john@example.com for details")
    assert result.passed is False
    assert "email" in result.violations[0].lower()


def test_pii_phone_detected():
    """Phone numbers must be detected as PII."""
    g = ContentGuardrails()
    g.add_rule("no_pii", patterns=["phone"])
    result = g.validate("Call me at 555-123-4567")
    assert result.passed is False


def test_pii_ssn_detected():
    """SSN patterns must be detected as PII."""
    g = ContentGuardrails()
    g.add_rule("no_pii", patterns=["ssn"])
    result = g.validate("My SSN is 123-45-6789")
    assert result.passed is False


def test_no_pii_in_clean_text():
    """Clean text must pass PII check."""
    g = ContentGuardrails()
    g.add_rule("no_pii")
    assert g.validate("The capital of France is Paris.").passed is True


def test_regex_block():
    """Custom regex block must detect matching patterns."""
    g = ContentGuardrails()
    g.add_rule("regex_block", pattern=r"confidential|secret|classified")
    result = g.validate("This is confidential information")
    assert result.passed is False


def test_regex_require():
    """Required pattern must pass when present, fail when absent."""
    g = ContentGuardrails()
    g.add_rule("regex_require", pattern=r"\d+")  # require at least one number
    assert g.validate("There are 42 answers").passed is True
    assert g.validate("No numbers here").passed is False


def test_custom_validator():
    """Custom validator function must be called."""
    g = ContentGuardrails()
    g.add_custom_validator(lambda text: (len(text) > 5, "Too short for custom"))
    assert g.validate("Hello World!").passed is True
    result = g.validate("Hi")
    assert result.passed is False
    assert "Too short" in result.violations[0]


def test_multiple_violations():
    """Multiple rule violations must all be reported."""
    g = ContentGuardrails()
    g.add_rule("max_length", max_chars=5)
    g.add_rule("no_pii", patterns=["email"])
    result = g.validate("Contact me at john@example.com for details")
    assert result.passed is False
    assert len(result.violations) >= 2


def test_sanitize_pii():
    """PII sanitization must replace PII with placeholders."""
    g = ContentGuardrails()
    result = g.sanitize_pii("Email john@example.com or call 555-123-4567")
    assert "john@example.com" not in result
    assert "<EMAIL>" in result
    assert "<PHONE>" in result


def test_toxic_content_detection():
    """Potentially harmful content must be flagged."""
    g = ContentGuardrails()
    g.add_rule("no_toxic")
    result = g.validate("Here are instructions on how to hack a password system")
    assert result.passed is False


def test_safe_content_passes_toxic():
    """Normal content must pass toxic check."""
    g = ContentGuardrails()
    g.add_rule("no_toxic")
    assert g.validate("Python is a great programming language").passed is True


def test_custom_validator_error_doesnt_crash():
    """Failing custom validator must not crash."""
    g = ContentGuardrails()
    g.add_custom_validator(lambda text: 1/0)  # division by zero
    result = g.validate("test")
    assert result.passed is True  # error swallowed, validation passes


def test_credit_card_detected():
    """Credit card numbers must be detected as PII."""
    g = ContentGuardrails()
    g.add_rule("no_pii", patterns=["credit_card"])
    result = g.validate("My card is 4111-1111-1111-1111")
    assert result.passed is False


def test_ip_address_detected():
    """IP addresses must be detected as PII."""
    g = ContentGuardrails()
    g.add_rule("no_pii", patterns=["ip_address"])
    result = g.validate("Server at 192.168.1.100")
    assert result.passed is False


def test_sanitize_preserves_non_pii():
    """Sanitization must preserve non-PII text."""
    g = ContentGuardrails()
    text = "The answer is 42."
    assert g.sanitize_pii(text) == text


def test_no_rules_passes():
    """With no rules configured, any text must pass."""
    g = ContentGuardrails()
    assert g.validate("Anything goes here").passed is True


def test_multiple_custom_validators():
    """Multiple custom validators must all be checked."""
    g = ContentGuardrails()
    g.add_custom_validator(lambda text: (len(text) > 2, "too short"))
    g.add_custom_validator(lambda text: ("bad" not in text, "contains bad"))
    result = g.validate("this is bad content")
    assert result.passed is False
    assert any("bad" in v for v in result.violations)
