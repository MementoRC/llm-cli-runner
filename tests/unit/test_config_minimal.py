"""Minimal unit tests for configuration management - TDD approach."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from mcp_server_cheap_llm.utils.errors import ConfigurationError


class TestConfigManagerTDD:
    """Test ConfigManager using TDD approach - start simple."""

    def test_config_manager_import(self):
        """Test that we can import ConfigManager."""
        from mcp_server_cheap_llm.utils.config import ConfigManager

        assert ConfigManager is not None

    def test_config_manager_instantiation_default(self):
        """Test ConfigManager can be instantiated with default config."""
        from mcp_server_cheap_llm.utils.config import ConfigManager

        # This should work and create default config
        with patch.object(ConfigManager, "_load_configuration") as mock_load:
            mock_config = Mock()
            mock_config.providers = []  # Empty list for len() to work
            mock_load.return_value = mock_config
            config_manager = ConfigManager()
            assert config_manager is not None
            mock_load.assert_called_once()

    def test_config_manager_has_required_methods(self):
        """Test ConfigManager has required methods."""
        from mcp_server_cheap_llm.utils.config import ConfigManager

        with patch.object(ConfigManager, "_load_configuration") as mock_load:
            mock_config = Mock()
            mock_config.providers = []  # Empty list for len() to work
            mock_load.return_value = mock_config
            config_manager = ConfigManager()

            # Should have these methods
            assert hasattr(config_manager, "get_enabled_providers")
            assert hasattr(config_manager, "get_provider_config")
            assert hasattr(config_manager, "get_default_provider")

    def test_get_enabled_providers_returns_list(self):
        """Test get_enabled_providers returns a list."""
        from mcp_server_cheap_llm.utils.config import ConfigManager

        # Mock providers with proper name attributes
        gemini_provider = Mock()
        gemini_provider.name = "gemini"
        gemini_provider.enabled = True

        codex_provider = Mock()
        codex_provider.name = "codex"
        codex_provider.enabled = False

        llama_provider = Mock()
        llama_provider.name = "llama"
        llama_provider.enabled = True

        # Mock the config to have some providers
        mock_config = Mock()
        mock_config.providers = [gemini_provider, codex_provider, llama_provider]

        with patch.object(ConfigManager, "_load_configuration") as mock_load:
            mock_load.return_value = mock_config
            config_manager = ConfigManager()

            enabled = config_manager.get_enabled_providers()
            assert isinstance(enabled, list)
            assert "gemini" in enabled
            assert "llama" in enabled
            assert "codex" not in enabled

    def test_get_default_provider_returns_string(self):
        """Test get_default_provider returns a string."""
        from mcp_server_cheap_llm.utils.config import ConfigManager

        # Mock the config to have a default provider
        mock_config = Mock()
        mock_config.default_provider = "gemini"
        mock_config.providers = []  # Empty list for len() to work

        with patch.object(ConfigManager, "_load_configuration") as mock_load:
            mock_load.return_value = mock_config
            config_manager = ConfigManager()

            default = config_manager.get_default_provider()
            assert isinstance(default, str)
            assert default == "gemini"

    def test_get_provider_config_returns_provider_or_none(self):
        """Test get_provider_config returns provider or None."""
        from mcp_server_cheap_llm.utils.config import ConfigManager

        # Mock provider config with proper name attribute
        gemini_provider = Mock()
        gemini_provider.name = "gemini"
        gemini_provider.enabled = True

        mock_config = Mock()
        mock_config.providers = [gemini_provider]

        with patch.object(ConfigManager, "_load_configuration") as mock_load:
            mock_load.return_value = mock_config
            config_manager = ConfigManager()

            # Should find existing provider
            found = config_manager.get_provider_config("gemini")
            assert found == gemini_provider

            # Should return None for non-existing provider
            not_found = config_manager.get_provider_config("nonexistent")
            assert not_found is None


class TestEnvironmentLoaderTDD:
    """Test EnvironmentLoader using TDD approach."""

    def test_environment_loader_import(self):
        """Test that we can import EnvironmentLoader."""
        from mcp_server_cheap_llm.utils.config import EnvironmentLoader

        assert EnvironmentLoader is not None

    def test_get_api_key_method_exists(self):
        """Test get_api_key method exists and is static."""
        from mcp_server_cheap_llm.utils.config import EnvironmentLoader

        assert hasattr(EnvironmentLoader, "get_api_key")
        assert callable(EnvironmentLoader.get_api_key)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-123"})
    def test_get_api_key_finds_key(self):
        """Test get_api_key finds API key from environment."""
        from mcp_server_cheap_llm.utils.config import EnvironmentLoader

        key = EnvironmentLoader.get_api_key("gemini")
        assert key == "test-key-123"

    @patch.dict("os.environ", {}, clear=True)
    def test_get_api_key_returns_none_when_not_found(self):
        """Test get_api_key returns None when no key found."""
        from mcp_server_cheap_llm.utils.config import EnvironmentLoader

        key = EnvironmentLoader.get_api_key("nonexistent")
        assert key is None

    def test_get_server_config_method_exists(self):
        """Test get_server_config method exists."""
        from mcp_server_cheap_llm.utils.config import EnvironmentLoader

        assert hasattr(EnvironmentLoader, "get_server_config")
        assert callable(EnvironmentLoader.get_server_config)

    def test_get_server_config_returns_dict(self):
        """Test get_server_config returns dictionary."""
        from mcp_server_cheap_llm.utils.config import EnvironmentLoader

        config = EnvironmentLoader.get_server_config()
        assert isinstance(config, dict)
        assert "default_provider" in config
        assert "log_level" in config
