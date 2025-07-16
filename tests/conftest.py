"""Test configuration and shared fixtures for MCP Server Cheap LLM."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from mcp_server_cheap_llm.server.handlers import CheapLLMServer
from mcp_server_cheap_llm.utils.config import ConfigManager


@pytest.fixture
def mock_config_dict() -> dict[str, Any]:
    """Mock configuration dictionary."""
    return {
        "providers": {
            "gemini": {
                "enabled": True,
                "cli_path": "/usr/local/bin/gemini",
                "model": "gemini-1.5-flash",
                "timeout": 30,
            },
            "codex": {
                "enabled": False,
                "api_key": "sk-test-key",
                "model": "code-davinci-002",
                "timeout": 30,
            },
            "llama": {
                "enabled": False,
                "model_path": "/path/to/llama/model",
                "n_ctx": 2048,
                "n_threads": 4,
            },
        },
        "server": {"host": "localhost", "port": 8080, "debug": False},
        "logging": {"level": "INFO", "format": "structured"},
    }


@pytest.fixture
def temp_config_file(mock_config_dict: dict[str, Any]) -> Path:
    """Create temporary config file."""
    import toml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(mock_config_dict, f)
        return Path(f.name)


@pytest.fixture
def mock_config_manager(mock_config_dict: dict[str, Any]) -> ConfigManager:
    """Mock ConfigManager instance."""
    manager = Mock(spec=ConfigManager)
    manager.get_config.return_value = mock_config_dict
    manager.get_enabled_providers.return_value = ["gemini"]
    manager.get_provider_config.return_value = mock_config_dict["providers"]["gemini"]
    return manager


@pytest.fixture
def mock_cheap_llm_server(mock_config_manager: ConfigManager) -> CheapLLMServer:
    """Mock CheapLLMServer instance."""
    server = Mock(spec=CheapLLMServer)
    server.config_manager = mock_config_manager
    server.get_mcp_server.return_value = Mock()
    return server


@pytest.fixture
def mock_gemini_cli() -> Mock:
    """Mock Gemini CLI provider."""
    cli = Mock()
    cli.is_available.return_value = True
    cli.generate_response = AsyncMock(return_value="Mock response")
    return cli


@pytest.fixture
def mock_llama_provider() -> Mock:
    """Mock LLaMA provider."""
    provider = Mock()
    provider.is_available.return_value = True
    provider.generate_response = AsyncMock(return_value="Mock response")
    return provider


@pytest.fixture
def sample_prompt() -> str:
    """Sample prompt for testing."""
    return "Write a Python function that adds two numbers"


@pytest.fixture
def sample_response() -> str:
    """Sample response for testing."""
    return "def add_numbers(a, b):\n    return a + b"


@pytest.fixture
def mock_mcp_server() -> Mock:
    """Mock MCP Server instance."""
    server = Mock()
    server.list_tools = AsyncMock(return_value=[])
    server.call_tool = AsyncMock(return_value={"content": "test"})
    return server


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch) -> None:
    """Setup test environment variables."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MCP_SERVER_CHEAP_LLM_TEST", "true")
