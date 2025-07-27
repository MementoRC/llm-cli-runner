"""
Unit tests for ServerConfigurationManager.

Tests comprehensive configuration loading, validation, and management functionality
including multi-source configuration loading, validation error handling, and
runtime configuration updates.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from pydantic import ValidationError

try:
    import yaml
except ImportError:
    yaml = None

from mcp_server_git.configuration.server_config import GitServerConfig
from mcp_server_git.frameworks.server_configuration import (
    ConfigurationError,
    ConfigurationState,
    ServerConfigurationManager,
)


class TestConfigurationState:
    """Test ConfigurationState functionality."""

    def test_configuration_state_creation(self):
        """Test creation of configuration state."""
        from datetime import datetime

        config = GitServerConfig()
        state = ConfigurationState(
            config=config,
            source_precedence=["defaults", "environment"],
            last_loaded=datetime.now(),
            validation_errors=[],
        )

        assert state.component_id == "server_configuration"
        assert state.component_type == "ServerConfigurationManager"
        assert state.config == config
        assert state.source_precedence == ["defaults", "environment"]
        assert isinstance(state.state_data, dict)

    def test_configuration_state_data(self):
        """Test configuration state data export."""
        from datetime import datetime

        config = GitServerConfig(port=8080, host="localhost")
        state = ConfigurationState(
            config=config,
            source_precedence=["defaults"],
            last_loaded=datetime.now(),
            validation_errors=["Test error"],
        )

        state_data = state.state_data
        assert "current_config" in state_data
        assert "source_precedence" in state_data
        assert "last_loaded" in state_data
        assert "validation_errors" in state_data
        assert state_data["validation_errors"] == ["Test error"]
        assert state_data["config_sources_active"] == 1


class TestServerConfigurationManager:
    """Test ServerConfigurationManager functionality."""

    @pytest.fixture
    def config_manager(self):
        """Create a basic configuration manager."""
        return ServerConfigurationManager()

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary configuration file."""
        config_data = {
            "host": "testhost",
            "port": 9000,
            "max_concurrent_operations": 20,
            "enable_security_validation": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_configuration_manager_creation(self, config_manager):
        """Test basic configuration manager creation."""
        assert config_manager._current_config is None
        assert config_manager._config_sources == {}
        assert not config_manager._initialized

    @pytest.mark.asyncio
    async def test_initialize_with_defaults(self, config_manager):
        """Test initialization with default configuration."""
        await config_manager.initialize()

        assert config_manager._initialized
        assert config_manager._current_config is not None
        assert isinstance(config_manager._current_config, GitServerConfig)
        assert config_manager._current_config.port == 8080  # Default port

    @pytest.mark.asyncio
    async def test_initialize_with_config_file(self, temp_config_file):
        """Test initialization with configuration file."""
        config_manager = ServerConfigurationManager(config_file_path=temp_config_file)
        await config_manager.initialize()

        config = config_manager.get_current_config()
        assert config.host == "testhost"
        assert config.port == 9000
        assert config.max_concurrent_operations == 20
        assert config.enable_security_validation is False

    @pytest.mark.asyncio
    async def test_load_json_config_file(self, config_manager, temp_config_file):
        """Test loading JSON configuration file."""
        config_data = await config_manager._load_config_file(temp_config_file)

        assert config_data["host"] == "testhost"
        assert config_data["port"] == 9000
        assert config_data["max_concurrent_operations"] == 20

    @pytest.mark.asyncio
    async def test_load_yaml_config_file(self, config_manager):
        """Test loading YAML configuration file."""
        if yaml is None:
            pytest.skip("PyYAML not available")

        config_data = {"host": "yamlhost", "port": 8888, "log_level": "DEBUG"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            loaded_data = await config_manager._load_config_file(temp_path)
            assert loaded_data["host"] == "yamlhost"
            assert loaded_data["port"] == 8888
            assert loaded_data["log_level"] == "DEBUG"
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_load_invalid_config_file(self, config_manager):
        """Test handling of invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content {")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ConfigurationError, match="Failed to load config file"):
                await config_manager._load_config_file(temp_path)
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_load_environment_config(self, config_manager):
        """Test loading configuration from environment variables."""
        test_env = {
            "MCP_GIT_HOST": "envhost",
            "MCP_GIT_PORT": "7777",
            "MCP_GIT_ENABLE_SECURITY_VALIDATION": "false",
            "MCP_GIT_MAX_CONCURRENT_OPERATIONS": "15",
            "OTHER_VAR": "ignored",
        }

        with patch.dict(os.environ, test_env, clear=False):
            env_config = await config_manager._load_environment_config()

        assert env_config["host"] == "envhost"
        assert env_config["port"] == 7777
        assert env_config["enable_security_validation"] is False
        assert env_config["max_concurrent_operations"] == 15
        assert "other_var" not in env_config

    @pytest.mark.asyncio
    async def test_load_environment_config_with_dotenv(self, config_manager):
        """Test loading configuration from .env file."""
        env_content = """
MCP_GIT_HOST=dotenvhost
MCP_GIT_PORT=6666
MCP_GIT_LOG_LEVEL=WARNING
"""

        # Store original environment variables to restore later
        original_env_vars = {}
        env_var_names = ["MCP_GIT_HOST", "MCP_GIT_PORT", "MCP_GIT_LOG_LEVEL"]
        for var_name in env_var_names:
            if var_name in os.environ:
                original_env_vars[var_name] = os.environ[var_name]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, dir="."
        ) as f:
            f.write(env_content)
            env_path = Path(f.name)
            env_path.rename(".env")

        try:
            env_config = await config_manager._load_environment_config()
            assert env_config.get("host") == "dotenvhost"
            assert env_config.get("port") == 6666
            assert env_config.get("log_level") == "WARNING"
        finally:
            # Clean up .env file
            if Path(".env").exists():
                Path(".env").unlink()

            # Clean up environment variables set by load_dotenv
            for var_name in env_var_names:
                if var_name in os.environ:
                    if var_name in original_env_vars:
                        os.environ[var_name] = original_env_vars[var_name]
                    else:
                        del os.environ[var_name]

    @pytest.mark.asyncio
    async def test_merge_configuration_sources(self, config_manager):
        """Test merging configuration sources with precedence."""
        # Set up test sources
        config_manager._config_sources = {
            "defaults": {"host": "default", "port": 8080, "log_level": "INFO"},
            "file": {"host": "filehost", "port": 9000},
            "environment": {"port": 7777},
        }

        merged = await config_manager._merge_configuration_sources()

        # Environment should override file, file should override defaults
        assert merged["host"] == "filehost"  # From file
        assert merged["port"] == 7777  # From environment (highest precedence)
        assert merged["log_level"] == "INFO"  # From defaults

    @pytest.mark.asyncio
    async def test_validate_configuration_success(self, config_manager):
        """Test successful configuration validation."""
        valid_config = {
            "host": "testhost",
            "port": 8080,
            "max_concurrent_operations": 10,
        }

        validated = await config_manager._validate_configuration(valid_config)

        assert isinstance(validated, GitServerConfig)
        assert validated.host == "testhost"
        assert validated.port == 8080

    @pytest.mark.asyncio
    async def test_validate_configuration_failure_strict(self, config_manager):
        """Test configuration validation failure in strict mode."""
        invalid_config = {
            "host": "testhost",
            "port": 99999,  # Invalid port (too high)
            "max_concurrent_operations": -1,  # Invalid (negative)
        }

        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            await config_manager._validate_configuration(invalid_config)

    @pytest.mark.asyncio
    async def test_validate_configuration_non_strict(self):
        """Test configuration validation in non-strict mode."""
        config_manager = ServerConfigurationManager(validation_strict=False)

        invalid_config = {
            "host": "testhost",
            "port": 99999,  # Invalid port
            "valid_field": "INFO",  # This would be log_level
        }

        # Should not raise, but use defaults for invalid fields
        validated = await config_manager._validate_configuration(invalid_config)

        assert isinstance(validated, GitServerConfig)
        assert validated.host == "testhost"
        assert validated.port != 99999  # Should use default

    @pytest.mark.asyncio
    async def test_get_current_config_before_init(self, config_manager):
        """Test getting configuration before initialization."""
        with pytest.raises(ConfigurationError, match="Configuration not initialized"):
            config_manager.get_current_config()

    @pytest.mark.asyncio
    async def test_get_current_config_after_init(self, config_manager):
        """Test getting configuration after initialization."""
        await config_manager.initialize()

        config = config_manager.get_current_config()
        assert isinstance(config, GitServerConfig)

    @pytest.mark.asyncio
    async def test_update_config_success(self, config_manager):
        """Test successful configuration update."""
        await config_manager.initialize()

        updates = {"port": 9999, "host": "updatedhost"}
        await config_manager.update_config(updates)

        config = config_manager.get_current_config()
        assert config.port == 9999
        assert config.host == "updatedhost"

    @pytest.mark.asyncio
    async def test_update_config_validation_failure(self, config_manager):
        """Test configuration update with validation failure."""
        await config_manager.initialize()

        invalid_updates = {"port": -1}  # Invalid port

        with pytest.raises(ConfigurationError, match="Failed to update configuration"):
            await config_manager.update_config(invalid_updates)

    @pytest.mark.asyncio
    async def test_update_config_before_init(self, config_manager):
        """Test updating configuration before initialization."""
        with pytest.raises(ConfigurationError, match="Configuration not initialized"):
            await config_manager.update_config({"port": 9000})

    @pytest.mark.asyncio
    async def test_reload_configuration(self, config_manager, temp_config_file):
        """Test configuration reloading."""
        config_manager._config_file_path = temp_config_file
        await config_manager.initialize()

        # Verify initial configuration
        config = config_manager.get_current_config()
        assert config.host == "testhost"

        # Update the config file
        new_config = {"host": "reloadedhost", "port": 8888}
        with open(temp_config_file, "w") as f:
            json.dump(new_config, f)

        # Reload configuration
        await config_manager.reload_configuration()

        updated_config = config_manager.get_current_config()
        assert updated_config.host == "reloadedhost"
        assert updated_config.port == 8888

    def test_export_configuration_dict(self, config_manager):
        """Test exporting configuration as dictionary."""
        # Initialize with a known config
        config_manager._current_config = GitServerConfig(host="exporthost", port=8888)

        exported = config_manager.export_configuration("dict")

        assert isinstance(exported, dict)
        assert exported["host"] == "exporthost"
        assert exported["port"] == 8888

    def test_export_configuration_json(self, config_manager):
        """Test exporting configuration as JSON."""
        config_manager._current_config = GitServerConfig(host="exporthost", port=8888)

        exported = config_manager.export_configuration("json")

        assert isinstance(exported, str)
        parsed = json.loads(exported)
        assert parsed["host"] == "exporthost"
        assert parsed["port"] == 8888

    def test_export_configuration_yaml(self, config_manager):
        """Test exporting configuration as YAML."""
        if yaml is None:
            pytest.skip("PyYAML not available")

        config_manager._current_config = GitServerConfig(host="exporthost", port=8888)

        exported = config_manager.export_configuration("yaml")

        assert isinstance(exported, str)
        parsed = yaml.safe_load(exported)
        assert parsed["host"] == "exporthost"
        assert parsed["port"] == 8888

    def test_export_configuration_invalid_format(self, config_manager):
        """Test exporting configuration with invalid format."""
        config_manager._current_config = GitServerConfig()

        with pytest.raises(ValueError, match="Unsupported format type"):
            config_manager.export_configuration("xml")

    def test_export_configuration_before_init(self, config_manager):
        """Test exporting configuration before initialization."""
        with pytest.raises(ConfigurationError, match="Configuration not initialized"):
            config_manager.export_configuration("dict")

    @pytest.mark.asyncio
    async def test_debuggable_component_interface(self, config_manager):
        """Test DebuggableComponent interface implementation."""
        await config_manager.initialize()

        # Test get_component_state
        state = config_manager.get_component_state()
        assert state.component_id == "server_configuration"
        assert state.component_type == "ServerConfigurationManager"
        assert isinstance(state.state_data, dict)

        # Test validate_component
        validation = config_manager.validate_component()
        assert validation["is_valid"] is True
        assert "validation_errors" in validation
        assert "validation_warnings" in validation

    def test_debuggable_component_before_init(self, config_manager):
        """Test DebuggableComponent interface before initialization."""
        state = config_manager.get_component_state()
        assert state.component_id == "server_configuration"
        assert "Component not initialized" in state.validation_errors

        validation = config_manager.validate_component()
        assert validation["is_valid"] is False
        assert "Component not initialized" in validation["validation_errors"]


class TestConfigurationIntegration:
    """Integration tests for configuration management."""

    @pytest.mark.asyncio
    async def test_full_configuration_lifecycle(self):
        """Test complete configuration lifecycle with file, env, and updates."""
        # Create temporary config file
        file_config = {"host": "filehost", "port": 8000}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(file_config, f)
            temp_path = Path(f.name)

        try:
            # Set environment variables
            env_vars = {"MCP_GIT_PORT": "9000", "MCP_GIT_LOG_LEVEL": "DEBUG"}
            with patch.dict(os.environ, env_vars, clear=False):
                # Initialize configuration manager
                config_manager = ServerConfigurationManager(config_file_path=temp_path)
                await config_manager.initialize()

                # Verify precedence: env should override file
                config = config_manager.get_current_config()
                assert config.host == "filehost"  # From file
                assert config.port == 9000  # From environment (override)
                assert config.log_level == "DEBUG"  # From environment

                # Update configuration
                await config_manager.update_config({"max_concurrent_operations": 25})
                updated_config = config_manager.get_current_config()
                assert updated_config.max_concurrent_operations == 25

                # Export configuration
                exported = config_manager.export_configuration("dict")
                assert exported["host"] == "filehost"
                assert exported["port"] == 9000
                assert exported["max_concurrent_operations"] == 25

                # Test state inspection
                state = config_manager.get_component_state()
                assert "environment" in state.source_precedence
                assert "file" in state.source_precedence

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_configuration_error_recovery(self):
        """Test configuration error handling and recovery."""
        config_manager = ServerConfigurationManager(validation_strict=True)

        # Start with valid configuration
        await config_manager.initialize()
        original_port = config_manager.get_current_config().port

        # Try invalid update (should fail)
        with pytest.raises(ConfigurationError):
            await config_manager.update_config({"port": "invalid_port"})

        # Original configuration should be preserved
        current_config = config_manager.get_current_config()
        assert current_config.port == original_port

        # Valid update should still work
        await config_manager.update_config({"host": "recoveredhost"})
        updated_config = config_manager.get_current_config()
        assert updated_config.host == "recoveredhost"
        assert updated_config.port == original_port
