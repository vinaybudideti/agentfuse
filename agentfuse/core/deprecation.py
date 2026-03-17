"""
ModelDeprecationChecker — warns when deprecated models are used.

OpenAI deprecated GPT-4, GPT-4o, GPT-4.1 in February 2026.
Claude Haiku 3 is being deprecated April 19, 2026.
Users may not be aware their code uses deprecated models.

This checker:
1. Warns at call time when a deprecated model is used
2. Suggests the replacement model
3. Can auto-redirect to the replacement (opt-in)

Usage:
    checker = ModelDeprecationChecker()
    model = checker.check_and_suggest("gpt-4o")
    # Logs warning: "gpt-4o is deprecated, use gpt-4.1 instead"
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


# Known deprecated models and their replacements
DEPRECATED_MODELS = {
    # OpenAI (deprecated Feb 13, 2026)
    "gpt-4": {"replacement": "gpt-5", "deprecated_date": "2026-02-13", "sunset_date": "2026-04-13"},
    "gpt-4-turbo": {"replacement": "gpt-5", "deprecated_date": "2026-02-13", "sunset_date": "2026-04-13"},
    "gpt-4o": {"replacement": "gpt-5", "deprecated_date": "2026-02-13", "sunset_date": "2026-04-13"},
    "gpt-4o-mini": {"replacement": "gpt-5", "deprecated_date": "2026-02-13", "sunset_date": "2026-04-13"},
    "gpt-4.1": {"replacement": "gpt-5", "deprecated_date": "2026-02-13", "sunset_date": "2026-04-13"},
    "gpt-4.1-mini": {"replacement": "gpt-5", "deprecated_date": "2026-02-13", "sunset_date": "2026-04-13"},
    "gpt-4.1-nano": {"replacement": "gpt-5", "deprecated_date": "2026-02-13", "sunset_date": "2026-04-13"},
    # Anthropic (deprecated Apr 19, 2026)
    "claude-3-haiku-20240307": {"replacement": "claude-haiku-4-5-20251001", "deprecated_date": "2026-03-01", "sunset_date": "2026-04-19"},
    "claude-3-sonnet-20240229": {"replacement": "claude-sonnet-4-6", "deprecated_date": "2025-10-01", "sunset_date": "2026-04-19"},
}


class ModelDeprecationChecker:
    """
    Checks for deprecated models and suggests replacements.
    """

    def __init__(self, auto_redirect: bool = False):
        """
        Args:
            auto_redirect: If True, automatically use replacement model
        """
        self._auto_redirect = auto_redirect
        self._warned: set[str] = set()  # only warn once per model

    def check_and_suggest(self, model: str) -> str:
        """
        Check if model is deprecated and return the model to use.

        If auto_redirect is True, returns the replacement model.
        Otherwise, returns the original model with a warning.
        """
        if model in DEPRECATED_MODELS:
            info = DEPRECATED_MODELS[model]
            replacement = info["replacement"]

            # Only warn once per model per session
            if model not in self._warned:
                self._warned.add(model)
                logger.warning(
                    "Model '%s' is DEPRECATED (since %s, sunset %s). "
                    "Recommended replacement: '%s'. "
                    "Set auto_redirect=True to automatically use the replacement.",
                    model, info["deprecated_date"], info["sunset_date"], replacement,
                )

            if self._auto_redirect:
                logger.info("Auto-redirecting: %s → %s", model, replacement)
                return replacement

        return model

    def is_deprecated(self, model: str) -> bool:
        """Check if a model is deprecated."""
        return model in DEPRECATED_MODELS

    def get_replacement(self, model: str) -> Optional[str]:
        """Get the replacement for a deprecated model."""
        if model in DEPRECATED_MODELS:
            return DEPRECATED_MODELS[model]["replacement"]
        return None

    def get_all_deprecated(self) -> dict:
        """Get all known deprecated models."""
        return dict(DEPRECATED_MODELS)
