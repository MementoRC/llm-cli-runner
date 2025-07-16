"""Tests for environment variable loading and validation.

This module tests the EnvironmentLoader class to ensure proper
handling of environment variables with validation and type conversion.
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mcp_server_cheap_llm.utils.config import EnvironmentLoader
from mcp_server_cheap_llm.utils.errors import ConfigurationError, ValidationError


class TestEnvironmentLoaderTDD:
    """Test-driven development tests for EnvironmentLoader."""

    def test_environment_loader_import(self):
        """Test that EnvironmentLoader can be imported."""
        from mcp_server_cheap_llm.utils.config import EnvironmentLoader

        assert EnvironmentLoader is not None

    def test_get_api_key_method_exists(self):
        """Test that get_api_key method exists and is callable."""
        assert hasattr(EnvironmentLoader, "get_api_key")
        assert callable(EnvironmentLoader.get_api_key)

    def test_get_api_key_finds_key(self):
        """Test that get_api_key finds API key in environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            key = EnvironmentLoader.get_api_key("openai")
            assert key == "sk-test123"

    def test_get_api_key_returns_none_when_not_found(self):
        """Test that get_api_key returns None when key not found."""
        with patch.dict(os.environ, {}, clear=True):
            key = EnvironmentLoader.get_api_key("nonexistent")
            assert key is None

    def test_get_server_config_method_exists(self):
        """Test that get_server_config method exists and is callable."""
        assert hasattr(EnvironmentLoader, "get_server_config")
        assert callable(EnvironmentLoader.get_server_config)

    def test_get_server_config_returns_dict(self):
        """Test that get_server_config returns a dictionary."""
        config = EnvironmentLoader.get_server_config()
        assert isinstance(config, dict)


class TestEnvironmentVariableLoading:
    """Test environment variable loading with comprehensive scenarios."""

    def test_api_key_loading_multiple_formats(self):
        """Test API key loading with different environment variable formats."""
        test_cases = [
            ("OPENAI_API_KEY", "sk-test123"),
            ("OPENAI_KEY", "sk-test456"),
            ("MCP_OPENAI_API_KEY", "sk-test789"),
        ]

        for env_var, expected_key in test_cases:
            with patch.dict(os.environ, {env_var: expected_key}, clear=True):
                key = EnvironmentLoader.get_api_key("openai")
                assert key == expected_key

    def test_api_key_priority_order(self):
        """Test that API keys are loaded in the correct priority order."""
        # Set multiple keys, should prefer PROVIDER_API_KEY format
        env_vars = {
            "OPENAI_API_KEY": "sk-priority1",
            "OPENAI_KEY": "sk-priority2",
            "MCP_OPENAI_API_KEY": "sk-priority3",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            key = EnvironmentLoader.get_api_key("openai")
            assert key == "sk-priority1"

    def test_different_provider_keys(self):
        """Test API key loading for different providers."""
        env_vars = {
            "OPENAI_API_KEY": "sk-openai123",
            "GOOGLE_API_KEY": "AIzaSyGoogle123",
            "ANTHROPIC_API_KEY": "sk-ant-api03-anthropic123",
            "GEMINI_API_KEY": "AIzaSyGemini123",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            assert EnvironmentLoader.get_api_key("openai") == "sk-openai123"
            assert EnvironmentLoader.get_api_key("google") == "AIzaSyGoogle123"
            assert (
                EnvironmentLoader.get_api_key("anthropic")
                == "sk-ant-api03-anthropic123"
            )
            assert EnvironmentLoader.get_api_key("gemini") == "AIzaSyGemini123"

    def test_case_insensitive_provider_names(self):
        """Test that provider names are handled case-insensitively."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            assert EnvironmentLoader.get_api_key("openai") == "sk-test123"
            assert EnvironmentLoader.get_api_key("OPENAI") == "sk-test123"
            assert EnvironmentLoader.get_api_key("OpenAI") == "sk-test123"


class TestServerConfigurationLoading:
    """Test server configuration loading from environment variables."""

    def test_server_config_defaults(self):
        """Test that server config returns appropriate defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = EnvironmentLoader.get_server_config()

            assert config["default_provider"] == "gemini"
            assert config["max_concurrent_requests"] == 10
            assert config["request_timeout_seconds"] == 30
            assert config["enable_metrics"] is True
            assert config["log_level"] == "INFO"

    def test_server_config_from_environment(self):
        """Test server config loading from environment variables."""
        env_vars = {
            "MCP_DEFAULT_PROVIDER": "openai",
            "MCP_MAX_CONCURRENT": "20",
            "MCP_REQUEST_TIMEOUT": "60",
            "MCP_ENABLE_METRICS": "false",
            "MCP_LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = EnvironmentLoader.get_server_config()

            assert config["default_provider"] == "openai"
            assert config["max_concurrent_requests"] == 20
            assert config["request_timeout_seconds"] == 60
            assert config["enable_metrics"] is False
            assert config["log_level"] == "DEBUG"

    def test_boolean_conversion(self):
        """Test boolean value conversion from environment."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("", False),
            ("invalid", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"MCP_ENABLE_METRICS": env_value}, clear=True):
                config = EnvironmentLoader.get_server_config()
                assert config["enable_metrics"] is expected

    def test_integer_conversion(self):
        """Test integer value conversion from environment."""
        with patch.dict(os.environ, {"MCP_MAX_CONCURRENT": "25"}, clear=True):
            config = EnvironmentLoader.get_server_config()
            assert config["max_concurrent_requests"] == 25
            assert isinstance(config["max_concurrent_requests"], int)

    def test_integer_conversion_invalid(self):
        """Test handling of invalid integer values."""
        with patch.dict(os.environ, {"MCP_MAX_CONCURRENT": "invalid"}, clear=True):
            with pytest.raises(ValueError):
                EnvironmentLoader.get_server_config()


class TestTypeConversionAndValidation:
    """Test type conversion and validation functionality."""

    def test_convert_type_method_exists(self):
        """Test that _convert_type method exists."""
        loader = EnvironmentLoader()
        assert hasattr(loader, "_convert_type")

    def test_convert_type_string(self):
        """Test string type conversion."""
        loader = EnvironmentLoader()
        result = loader._convert_type("test_value", str)
        assert result == "test_value"
        assert isinstance(result, str)

    def test_convert_type_integer(self):
        """Test integer type conversion."""
        loader = EnvironmentLoader()
        result = loader._convert_type("123", int)
        assert result == 123
        assert isinstance(result, int)

    def test_convert_type_boolean(self):
        """Test boolean type conversion."""
        loader = EnvironmentLoader()

        # Test true values
        for value in ["true", "True", "TRUE", "1", "yes", "on"]:
            result = loader._convert_type(value, bool)
            assert result is True

        # Test false values
        for value in ["false", "False", "FALSE", "0", "no", "off", ""]:
            result = loader._convert_type(value, bool)
            assert result is False

    def test_convert_type_list(self):
        """Test list type conversion."""
        loader = EnvironmentLoader()
        result = loader._convert_type("item1,item2,item3", list)
        assert result == ["item1", "item2", "item3"]
        assert isinstance(result, list)

    def test_convert_type_invalid_integer(self):
        """Test invalid integer conversion raises ValueError."""
        loader = EnvironmentLoader()
        with pytest.raises(ValueError):
            loader._convert_type("invalid", int)

    def test_convert_type_unsupported_type(self):
        """Test unsupported type raises ValueError."""
        loader = EnvironmentLoader()
        with pytest.raises(ValueError):
            loader._convert_type("value", dict)


class TestValidationAndErrorHandling:
    """Test validation and error handling."""

    def test_validate_required_vars_method_exists(self):
        """Test that _validate_required_vars method exists."""
        loader = EnvironmentLoader()
        assert hasattr(loader, "_validate_required_vars")

    def test_validate_required_vars_success(self):
        """Test successful validation of required variables."""
        loader = EnvironmentLoader()
        config = {
            "default_provider": "openai",
            "max_concurrent_requests": 10,
            "log_level": "INFO",
        }

        # Should not raise any exception
        loader._validate_required_vars(config)

    def test_validate_required_vars_missing_required(self):
        """Test validation failure when required variables are missing."""
        loader = EnvironmentLoader()
        config = {
            "max_concurrent_requests": 10,
            # Missing default_provider
        }

        with pytest.raises(ValidationError):
            loader._validate_required_vars(config)

    def test_validate_required_vars_invalid_log_level(self):
        """Test validation failure for invalid log level."""
        loader = EnvironmentLoader()
        config = {
            "default_provider": "openai",
            "max_concurrent_requests": 10,
            "log_level": "INVALID_LEVEL",
        }

        with pytest.raises(ValidationError):
            loader._validate_required_vars(config)

    def test_validate_required_vars_invalid_concurrent_requests(self):
        """Test validation failure for invalid concurrent requests value."""
        loader = EnvironmentLoader()
        config = {
            "default_provider": "openai",
            "max_concurrent_requests": -5,  # Invalid negative value
            "log_level": "INFO",
        }

        with pytest.raises(ValidationError):
            loader._validate_required_vars(config)


class TestEnvironmentLoaderInstance:
    """Test EnvironmentLoader instance methods."""

    def test_environment_loader_instantiation(self):
        """Test EnvironmentLoader can be instantiated."""
        loader = EnvironmentLoader()
        assert loader is not None
        assert isinstance(loader, EnvironmentLoader)

    def test_get_api_key_with_validation(self):
        """Test get_api_key with validation."""
        loader = EnvironmentLoader()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            key = loader.get_api_key("openai")
            assert key == "sk-test123"

    def test_get_api_key_empty_value(self):
        """Test get_api_key with empty environment value."""
        loader = EnvironmentLoader()

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            key = loader.get_api_key("openai")
            assert key is None

    def test_get_api_key_whitespace_only(self):
        """Test get_api_key with whitespace-only value."""
        loader = EnvironmentLoader()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "   "}):
            key = loader.get_api_key("openai")
            assert key is None

    def test_get_server_config_with_validation(self):
        """Test get_server_config with validation."""
        loader = EnvironmentLoader()

        config = loader.get_server_config()
        assert isinstance(config, dict)

        # Should have all required keys
        required_keys = [
            "default_provider",
            "max_concurrent_requests",
            "request_timeout_seconds",
            "enable_metrics",
            "log_level",
        ]

        for key in required_keys:
            assert key in config

    def test_get_server_config_validated_types(self):
        """Test that get_server_config returns correct types."""
        loader = EnvironmentLoader()

        config = loader.get_server_config()

        assert isinstance(config["default_provider"], str)
        assert isinstance(config["max_concurrent_requests"], int)
        assert isinstance(config["request_timeout_seconds"], int)
        assert isinstance(config["enable_metrics"], bool)
        assert isinstance(config["log_level"], str)


class TestEnvironmentLoaderIntegration:
    """Integration tests for EnvironmentLoader."""

    def test_complete_configuration_loading(self):
        """Test complete configuration loading scenario."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test123",
            "GOOGLE_API_KEY": "AIzaSyTest456",
            "MCP_DEFAULT_PROVIDER": "openai",
            "MCP_MAX_CONCURRENT": "15",
            "MCP_REQUEST_TIMEOUT": "45",
            "MCP_ENABLE_METRICS": "true",
            "MCP_LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            loader = EnvironmentLoader()

            # Test API key loading
            openai_key = loader.get_api_key("openai")
            google_key = loader.get_api_key("google")

            assert openai_key == "sk-test123"
            assert google_key == "AIzaSyTest456"

            # Test server config loading
            config = loader.get_server_config()

            assert config["default_provider"] == "openai"
            assert config["max_concurrent_requests"] == 15
            assert config["request_timeout_seconds"] == 45
            assert config["enable_metrics"] is True
            assert config["log_level"] == "DEBUG"

    def test_mixed_environment_and_defaults(self):
        """Test configuration with mix of environment vars and defaults."""
        env_vars = {
            "MCP_DEFAULT_PROVIDER": "google",
            "MCP_LOG_LEVEL": "WARNING",
            # Other values should use defaults
        }

        with patch.dict(os.environ, env_vars, clear=True):
            loader = EnvironmentLoader()
            config = loader.get_server_config()

            # Should use environment values
            assert config["default_provider"] == "google"
            assert config["log_level"] == "WARNING"

            # Should use defaults
            assert config["max_concurrent_requests"] == 10
            assert config["request_timeout_seconds"] == 30
            assert config["enable_metrics"] is True
