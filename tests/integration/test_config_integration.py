"""Integration tests for complete configuration loading system.

This module tests the ConfigManager class integration with all components:
environment variables, file loading, API key management, and validation.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import tomli_w

from mcp_server_cheap_llm.utils.config import ConfigManager
from mcp_server_cheap_llm.utils.errors import ConfigurationError, ValidationError


class TestConfigurationIntegration:
    """Test suite for complete configuration integration."""

    def test_config_manager_instantiation(self):
        """Test ConfigManager can be instantiated."""
        manager = ConfigManager()
        assert manager is not None
        assert hasattr(manager, "load_configuration")
        assert hasattr(manager, "get_provider_config")
        assert hasattr(manager, "get_enabled_providers")

    def test_load_from_toml_file(self):
        """Test loading configuration from TOML file."""
        config_data = {
            "server": {"host": "localhost", "port": 8000, "log_level": "INFO"},
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_key": "sk-test-key-from-file",
                    "model": "gpt-3.5-turbo",
                    "max_tokens": 1000,
                },
                "google": {"enabled": False, "api_key": "AIzaSyTestKey"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(config_data, f)
            config_file = f.name

        try:
            manager = ConfigManager(config_file)
            manager.load_configuration()

            # Test server config
            assert manager.get_server_config()["host"] == "localhost"
            assert manager.get_server_config()["port"] == 8000

            # Test provider config
            openai_config = manager.get_provider_config("openai")
            assert openai_config["enabled"] is True
            assert openai_config["model"] == "gpt-3.5-turbo"

            # Test enabled providers
            enabled = manager.get_enabled_providers()
            assert "openai" in enabled
            assert "google" not in enabled

        finally:
            os.unlink(config_file)

    def test_load_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "server": {"host": "0.0.0.0", "port": 9000, "log_level": "DEBUG"},
            "providers": {
                "anthropic": {
                    "enabled": True,
                    "api_key": "sk-ant-test-key",
                    "model": "claude-3-sonnet",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            manager = ConfigManager(config_file)
            manager.load_configuration()

            assert manager.get_server_config()["host"] == "0.0.0.0"
            assert manager.get_server_config()["port"] == 9000

            anthropic_config = manager.get_provider_config("anthropic")
            assert anthropic_config["enabled"] is True
            assert anthropic_config["model"] == "claude-3-sonnet"

        finally:
            os.unlink(config_file)

    def test_environment_override_precedence(self):
        """Test environment variables override file configuration."""
        config_data = {
            "server": {"host": "localhost", "port": 8000},
            "providers": {"openai": {"enabled": True, "api_key": "sk-file-key"}},
        }

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(config_data, f)
            config_file = f.name

        env_vars = {
            "MCP_SERVER_HOST": "0.0.0.0",
            "MCP_SERVER_PORT": "9000",
            "OPENAI_API_KEY": "sk-env-override-key",
        }

        try:
            with patch.dict(os.environ, env_vars):
                manager = ConfigManager(config_file)
                manager.load_configuration()

                # Environment should override file
                assert manager.get_server_config()["host"] == "0.0.0.0"
                assert manager.get_server_config()["port"] == 9000

                openai_config = manager.get_provider_config("openai")
                assert openai_config["api_key"] == "sk-env-override-key"

        finally:
            os.unlink(config_file)

    def test_configuration_validation_pipeline(self):
        """Test complete configuration validation pipeline."""
        config_data = {
            "server": {
                "host": "localhost",
                "port": "invalid-port",  # Should fail validation
                "log_level": "INFO",
            },
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_key": "invalid-key",  # Should fail validation
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(config_data, f)
            config_file = f.name

        try:
            manager = ConfigManager(config_file)

            with pytest.raises(ValidationError):
                manager.load_configuration()

        finally:
            os.unlink(config_file)

    def test_missing_config_file_fallback(self):
        """Test behavior when config file is missing."""
        manager = ConfigManager("/non/existent/config.toml")

        # Should fall back to environment variables only
        with patch.dict(
            os.environ,
            {
                "MCP_SERVER_HOST": "localhost",
                "MCP_SERVER_PORT": "8000",
                "MCP_LOG_LEVEL": "INFO",
            },
        ):
            manager.load_configuration()

            config = manager.get_server_config()
            assert config["host"] == "localhost"
            assert config["port"] == 8000
            assert config["log_level"] == "INFO"

    def test_configuration_caching(self):
        """Test configuration caching functionality."""
        config_data = {
            "server": {"host": "localhost", "port": 8000},
            "providers": {"openai": {"enabled": True}},
        }

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(config_data, f)
            config_file = f.name

        try:
            manager = ConfigManager(config_file)

            # First load
            manager.load_configuration()
            config1 = manager.get_server_config()

            # Second load should use cache
            config2 = manager.get_server_config()
            assert config1 is config2  # Same object reference (cached)

        finally:
            os.unlink(config_file)

    def test_configuration_reload(self):
        """Test configuration reload functionality."""
        initial_config = {"server": {"host": "localhost", "port": 8000}}

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(initial_config, f)
            config_file = f.name

        try:
            manager = ConfigManager(config_file)
            manager.load_configuration()

            assert manager.get_server_config()["port"] == 8000

            # Update config file
            updated_config = {"server": {"host": "localhost", "port": 9000}}

            with open(config_file, "wb") as f:
                tomli_w.dump(updated_config, f)

            # Reload configuration
            manager.reload_configuration()

            assert manager.get_server_config()["port"] == 9000

        finally:
            os.unlink(config_file)

    @patch.dict(os.environ, {}, clear=True)  # Clear environment to avoid real API keys
    def test_encrypted_api_keys_integration(self):
        """Test integration with encrypted API key storage."""
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_key": "sk-testintegrationkey12345678",
                    "encrypt_keys": True,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(config_data, f)
            config_file = f.name

        try:
            manager = ConfigManager(config_file)
            manager.load_configuration()

            # API key should be stored encrypted
            assert manager.key_manager.key_exists("openai")

            # But should be retrievable in plaintext
            openai_config = manager.get_provider_config("openai")
            assert openai_config["api_key"] == "sk-testintegrationkey12345678"

        finally:
            os.unlink(config_file)

    def test_multiple_provider_configuration(self):
        """Test configuration with multiple providers."""
        config_data = {
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_key": "sk-openai-key",
                    "model": "gpt-4",
                    "max_tokens": 2000,
                },
                "google": {
                    "enabled": True,
                    "api_key": "AIzaSyGoogleKey",
                    "model": "gemini-pro",
                },
                "anthropic": {"enabled": False, "api_key": "sk-ant-key"},
                "llama": {"enabled": True, "model_path": "/path/to/llama/model.gguf"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(config_data, f)
            config_file = f.name

        try:
            manager = ConfigManager(config_file)
            manager.load_configuration()

            enabled_providers = manager.get_enabled_providers()
            assert set(enabled_providers) == {"openai", "google", "llama"}

            # Test individual provider configs
            openai_config = manager.get_provider_config("openai")
            assert openai_config["model"] == "gpt-4"
            assert openai_config["max_tokens"] == 2000

            google_config = manager.get_provider_config("google")
            assert google_config["model"] == "gemini-pro"

            llama_config = manager.get_provider_config("llama")
            assert llama_config["model_path"] == "/path/to/llama/model.gguf"

        finally:
            os.unlink(config_file)

    def test_configuration_error_handling(self):
        """Test comprehensive error handling in configuration loading."""
        # Test invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            invalid_json_file = f.name

        try:
            manager = ConfigManager(invalid_json_file)
            with pytest.raises(ConfigurationError):
                manager.load_configuration()
        finally:
            os.unlink(invalid_json_file)

        # Test unsupported file format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("test: value")
            yaml_file = f.name

        try:
            manager = ConfigManager(yaml_file)
            with pytest.raises(
                ConfigurationError,
                match="Unsupported configuration file format",
            ):
                manager.load_configuration()
        finally:
            os.unlink(yaml_file)

    def test_configuration_performance(self):
        """Test configuration loading performance."""
        import time

        config_data = {
            "server": {"host": "localhost", "port": 8000},
            "providers": {f"provider_{i}": {"enabled": True} for i in range(100)},
        }

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(config_data, f)
            config_file = f.name

        try:
            manager = ConfigManager(config_file)

            start_time = time.time()
            manager.load_configuration()
            load_time = time.time() - start_time

            # Should load quickly (under 1 second even with 100 providers)
            assert load_time < 1.0

            # Cached access should be very fast
            start_time = time.time()
            for _ in range(10):
                manager.get_server_config()
            cache_time = time.time() - start_time

            assert cache_time < 0.1  # 10 cached accesses under 100ms

        finally:
            os.unlink(config_file)
