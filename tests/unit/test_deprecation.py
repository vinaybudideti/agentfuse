"""
Tests for ModelDeprecationChecker.
"""

from agentfuse.core.deprecation import ModelDeprecationChecker


def test_deprecated_model_detected():
    """Deprecated model must be detected."""
    checker = ModelDeprecationChecker()
    assert checker.is_deprecated("gpt-4o") is True
    assert checker.is_deprecated("gpt-5") is False


def test_replacement_suggestion():
    """Deprecated model must have a replacement."""
    checker = ModelDeprecationChecker()
    assert checker.get_replacement("gpt-4o") == "gpt-5"
    assert checker.get_replacement("gpt-5") is None


def test_auto_redirect():
    """Auto-redirect must return the replacement model."""
    checker = ModelDeprecationChecker(auto_redirect=True)
    result = checker.check_and_suggest("gpt-4o")
    assert result == "gpt-5"


def test_no_redirect_by_default():
    """Without auto_redirect, original model must be returned."""
    checker = ModelDeprecationChecker(auto_redirect=False)
    result = checker.check_and_suggest("gpt-4o")
    assert result == "gpt-4o"  # returns original with warning


def test_non_deprecated_unchanged():
    """Non-deprecated model must pass through unchanged."""
    checker = ModelDeprecationChecker()
    assert checker.check_and_suggest("gpt-5") == "gpt-5"


def test_claude_haiku_3_deprecated():
    """Claude Haiku 3 must be marked as deprecated."""
    checker = ModelDeprecationChecker()
    assert checker.is_deprecated("claude-3-haiku-20240307") is True
    assert checker.get_replacement("claude-3-haiku-20240307") == "claude-haiku-4-5-20251001"


def test_get_all_deprecated():
    """get_all_deprecated must return a non-empty dict."""
    checker = ModelDeprecationChecker()
    all_deprecated = checker.get_all_deprecated()
    assert len(all_deprecated) > 0
    assert "gpt-4o" in all_deprecated
