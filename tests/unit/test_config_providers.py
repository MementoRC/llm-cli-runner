"""Unit tests for provider-specific configuration classes."""

import pytest
from pydantic import ValidationError

from mcp_server_cheap_llm.utils.config import (  # type: ignore[import-not-found]
    ProviderConfig,
)


class TestProviderConfig:
    """Test basic provider configuration functionality."""

    def test_provider_config_basic(self):
        """Test basic provider configuration."""
        config = ProviderConfig(name="test", rate_limit=60, quota_limit=100000)

        assert config.name == "test"
        assert config.rate_limit == 60
        assert config.quota_limit == 100000
        assert config.enabled is True
        assert config.timeout == 30

    def test_provider_config_validation(self):
        """Test provider configuration validation."""
        # Test rate limit validation
        with pytest.raises(ValidationError):
            ProviderConfig(name="test", rate_limit=0)

        # Test quota limit validation
        with pytest.raises(ValidationError):
            ProviderConfig(name="test", quota_limit=0)

    def test_provider_config_endpoint_validation(self):
        """Test endpoint URL validation."""
        # Valid endpoints
        valid_endpoints = [
            "https://api.openai.com/v1",
            "https://generativelanguage.googleapis.com/v1",
            "https://api.anthropic.com/v1",
        ]

        for endpoint in valid_endpoints:
            config = ProviderConfig(name="test", endpoint=endpoint)
            assert config.endpoint == endpoint

    def test_provider_config_model_settings(self):
        """Test model settings configuration."""
        model_settings = {
            "gpt-4": {"max_tokens": 8192},
            "gpt-3.5-turbo": {"max_tokens": 4096},
        }

        config = ProviderConfig(name="test", model_settings=model_settings)

        assert config.model_settings == model_settings
        assert len(config.model_settings) == 2
