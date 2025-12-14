"""Unit tests for provider-specific configuration classes."""

import pytest
from pydantic import ValidationError

from mcp_server_cheap_llm.core.models import (  # type: ignore[import-not-found]
    ProviderConfig,
    ProviderType,
)


class TestProviderConfig:
    """Test basic provider configuration functionality."""

    def test_provider_config_basic(self):
        """Test basic provider configuration."""
        config = ProviderConfig(
            name="test",
            provider_type=ProviderType.GEMINI,
            models=["gemini-pro"],
            rate_limit={"requests_per_minute": 60},
        )

        assert config.name == "test"
        assert config.provider_type == ProviderType.GEMINI
        assert config.models == ["gemini-pro"]
        assert config.rate_limit == {"requests_per_minute": 60}
        assert config.enabled is True
        assert config.timeout == 30

    def test_provider_config_validation(self):
        """Test provider configuration validation."""
        # Test invalid timeout validation
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                provider_type=ProviderType.GEMINI,
                models=["gemini-pro"],
                timeout=0,  # Invalid: must be >= 1
            )

    def test_provider_config_endpoint_validation(self):
        """Test endpoint URL validation."""
        # Valid endpoints
        valid_endpoints = [
            "https://api.openai.com/v1",
            "https://generativelanguage.googleapis.com/v1",
            "https://api.anthropic.com/v1",
        ]

        for endpoint in valid_endpoints:
            config = ProviderConfig(
                name="test",
                provider_type=ProviderType.OPENAI,
                models=["gpt-3.5-turbo"],
                base_url=endpoint,
            )
            assert config.base_url == endpoint

    def test_provider_config_model_settings(self):
        """Test model settings configuration."""
        provider_specific = {
            "gpt-4": {"max_tokens": 8192},
            "gpt-3.5-turbo": {"max_tokens": 4096},
        }

        config = ProviderConfig(
            name="test",
            provider_type=ProviderType.OPENAI,
            models=["gpt-4"],
            provider_specific=provider_specific,
        )

        assert config.provider_specific == provider_specific
        assert len(config.provider_specific) == 2
