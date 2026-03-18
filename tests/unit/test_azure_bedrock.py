"""Tests for Azure OpenAI and cloud provider routing."""

from agentfuse.providers.router import resolve_provider, OPENAI_COMPATIBLE_PROVIDERS
import os


def test_azure_provider_in_list():
    """Azure must be in the provider list."""
    assert "azure" in OPENAI_COMPATIBLE_PROVIDERS


def test_bedrock_provider_in_list():
    """Bedrock must be in the provider list."""
    assert "bedrock" in OPENAI_COMPATIBLE_PROVIDERS


def test_siliconflow_provider_in_list():
    """SiliconFlow must be in the provider list."""
    assert "siliconflow" in OPENAI_COMPATIBLE_PROVIDERS


def test_azure_routing():
    """azure/model must route to azure provider."""
    name, base_url = resolve_provider("azure/gpt-4o")
    assert name == "azure"


def test_siliconflow_routing():
    """siliconflow/model must route with correct base_url."""
    name, base_url = resolve_provider("siliconflow/gpt-oss-120b")
    assert name == "siliconflow"
    assert "siliconflow" in base_url


def test_provider_count():
    """Must have 13+ providers."""
    assert len(OPENAI_COMPATIBLE_PROVIDERS) >= 13
