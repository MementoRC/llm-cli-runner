"""Unit tests for configuration management system."""

import pytest
from pydantic import ValidationError

from mcp_server_cheap_llm.utils.config import APIKeyConfig, ConfigModel, ProviderConfig


class TestConfigModel:
    """Test suite for basic ConfigModel functionality."""

    def test_config_model_instantiation(self):
        """Test basic ConfigModel can be instantiated with required fields."""
        config = ConfigModel(
            server_name="test-server",
            log_level="INFO",
            enabled_providers=["openai", "google"],
            default_provider="openai",
        )
        assert config.server_name == "test-server"
        assert config.log_level == "INFO"
        assert config.enabled_providers == ["openai", "google"]
        assert config.default_provider == "openai"

    def test_config_model_validation_missing_required(self):
        """Test ConfigModel raises ValidationError for missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigModel(server_name="test-server")

        errors = exc_info.value.errors()
        assert (
            len(errors) >= 2
        )  # enabled_providers, default_provider missing (log_level has default)
        field_names = {error["loc"][0] for error in errors}
        assert "enabled_providers" in field_names
        assert "default_provider" in field_names

    def test_config_model_default_values(self):
        """Test ConfigModel applies appropriate default values."""
        config = ConfigModel(
            server_name="test-server",
            enabled_providers=["openai"],
            default_provider="openai",
        )
        assert config.log_level == "INFO"  # Should have default
        assert config.max_retries == 3  # Should have default
        assert config.timeout == 30  # Should have default in seconds

    def test_config_model_log_level_validation(self):
        """Test log_level field validates against allowed values."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = ConfigModel(
                server_name="test",
                log_level=level,
                enabled_providers=["openai"],
                default_provider="openai",
            )
            assert config.log_level == level

        # Invalid log level
        with pytest.raises(ValidationError) as exc_info:
            ConfigModel(
                server_name="test",
                log_level="INVALID",
                enabled_providers=["openai"],
                default_provider="openai",
            )

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "log_level" for error in errors)

    def test_config_model_provider_validation(self):
        """Test that default_provider must be in enabled_providers."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigModel(
                server_name="test",
                enabled_providers=["openai"],
                default_provider="google",  # Not in enabled_providers
            )

        errors = exc_info.value.errors()
        assert any(
            "default_provider must be one of enabled_providers" in str(error)
            for error in errors
        )


class TestAPIKeyConfig:
    """Test suite for APIKeyConfig model."""

    def test_api_key_config_instantiation(self):
        """Test APIKeyConfig can be instantiated with provider and key."""
        key_config = APIKeyConfig(
            provider="openai",
            api_key="sk-test123",
            is_encrypted=False,
        )
        assert key_config.provider == "openai"
        assert key_config.api_key == "sk-test123"
        assert key_config.is_encrypted is False

    def test_api_key_config_validation(self):
        """Test APIKeyConfig validates required fields."""
        with pytest.raises(ValidationError) as exc_info:
            APIKeyConfig(provider="openai")  # Missing api_key

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "api_key" for error in errors)

    def test_api_key_config_empty_key_validation(self):
        """Test APIKeyConfig rejects empty API keys."""
        with pytest.raises(ValidationError) as exc_info:
            APIKeyConfig(
                provider="openai",
                api_key="",  # Empty string should be invalid
                is_encrypted=False,
            )

        errors = exc_info.value.errors()
        # Check that we have a validation error for api_key
        assert any(error["loc"][0] == "api_key" for error in errors)
        # With min_length=1, Pydantic gives us a string_too_short error
        assert any(error["type"] == "string_too_short" for error in errors)

    def test_api_key_config_provider_validation(self):
        """Test APIKeyConfig validates provider against known providers."""
        # Valid providers
        for provider in ["openai", "google", "anthropic", "llama", "codex"]:
            key_config = APIKeyConfig(
                provider=provider,
                api_key="test-key",
                is_encrypted=False,
            )
            assert key_config.provider == provider

        # Invalid provider
        with pytest.raises(ValidationError) as exc_info:
            APIKeyConfig(
                provider="invalid-provider",
                api_key="test-key",
                is_encrypted=False,
            )

        errors = exc_info.value.errors()
        assert any("provider" in str(error) for error in errors)


class TestProviderConfig:
    """Test suite for ProviderConfig model."""

    def test_provider_config_instantiation(self):
        """Test ProviderConfig can be instantiated with basic fields."""
        provider_config = ProviderConfig(
            name="openai",
            endpoint="https://api.openai.com/v1",
            rate_limit=100,
            quota_limit=1000000,
            enabled=True,
        )
        assert provider_config.name == "openai"
        assert provider_config.endpoint == "https://api.openai.com/v1"
        assert provider_config.rate_limit == 100
        assert provider_config.quota_limit == 1000000
        assert provider_config.enabled is True

    def test_provider_config_defaults(self):
        """Test ProviderConfig applies sensible defaults."""
        provider_config = ProviderConfig(name="openai")
        assert provider_config.enabled is True  # Should default to True
        assert provider_config.rate_limit > 0  # Should have default rate limit
        assert provider_config.timeout == 30  # Should have default timeout

    def test_provider_config_endpoint_validation(self):
        """Test ProviderConfig validates endpoint URLs."""
        # Valid endpoints
        provider_config = ProviderConfig(
            name="openai",
            endpoint="https://api.openai.com/v1",
        )
        assert provider_config.endpoint == "https://api.openai.com/v1"

        # Invalid endpoint (not a URL)
        with pytest.raises(ValidationError) as exc_info:
            ProviderConfig(
                name="openai",
                endpoint="not-a-url",
            )

        errors = exc_info.value.errors()
        assert any("endpoint" in str(error) for error in errors)

    def test_provider_config_rate_limit_validation(self):
        """Test ProviderConfig validates rate limits are positive."""
        # Valid rate limit
        provider_config = ProviderConfig(
            name="openai",
            rate_limit=100,
        )
        assert provider_config.rate_limit == 100

        # Invalid rate limit (negative)
        with pytest.raises(ValidationError) as exc_info:
            ProviderConfig(
                name="openai",
                rate_limit=-1,
            )

        errors = exc_info.value.errors()
        # Check that we have a validation error for rate_limit
        assert any(error["loc"][0] == "rate_limit" for error in errors)
        # With Field(gt=0), Pydantic gives a different error message
        assert any("greater than 0" in str(error) for error in errors)

    def test_provider_config_model_specific_settings(self):
        """Test ProviderConfig can store model-specific settings."""
        provider_config = ProviderConfig(
            name="openai",
            model_settings={
                "gpt-4": {"max_tokens": 8192, "temperature": 0.7},
                "gpt-3.5-turbo": {"max_tokens": 4096, "temperature": 0.9},
            },
        )
        assert "gpt-4" in provider_config.model_settings
        assert provider_config.model_settings["gpt-4"]["max_tokens"] == 8192
