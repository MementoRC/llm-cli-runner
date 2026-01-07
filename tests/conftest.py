"""Test configuration and shared fixtures for MCP Server Cheap LLM."""

import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from mcp_server_cheap_llm.server.handlers import CheapLLMServer
from mcp_server_cheap_llm.utils.config import ConfigManager


def _check_git_available():
    """Check if git is available and works properly.

    Returns:
        True if git init works in a temp directory, False otherwise
    """
    import os
    import shutil

    # First check if git binary exists
    if not shutil.which("git"):
        return False

    # Try to run git init in a temp directory to verify it works
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = os.environ.copy()
            # Remove shell wrapper environment variables that might interfere
            for var in ["CLAUDE_CODE_SHELL_PREFIX", "BASH_ENV"]:
                env.pop(var, None)

            result = subprocess.run(
                ["git", "init"],
                cwd=tmp_dir,
                capture_output=True,
                env=env,
                timeout=10,
            )
            return result.returncode == 0
    except Exception:
        return False


# Cache the git availability check
_git_available = None


def is_git_available():
    """Check if git is available (cached)."""
    global _git_available
    if _git_available is None:
        _git_available = _check_git_available()
    return _git_available


# Marker for tests that require git
requires_git = pytest.mark.skipif(
    not is_git_available(), reason="Git is not available or blocked in this environment"
)


def _run_git_isolated(cmd, *, cwd=None, check=False, capture_output=False, **kwargs):
    """Run git command in isolated environment (similar to subprocess.run).

    This function provides git execution in a clean environment for testing,
    avoiding interference from ClaudeCode redirectors or mocked environments.

    Args:
        cmd: Command list (e.g., ["git", "init"])
        cwd: Working directory for the command
        check: Raise exception if command fails
        capture_output: Capture stdout/stderr
        **kwargs: Additional subprocess arguments

    Returns:
        subprocess.CompletedProcess object
    """
    import os

    # Create clean environment for git execution
    env = os.environ.copy()

    # Remove any test-specific environment variables that could interfere
    test_vars_to_remove = [
        "MCP_TEST_MODE",
        "PYTEST_CURRENT_TEST",
        "GITHUB_TOKEN",  # Use system git, not mocked
        "CLAUDE_CODE_SHELL_PREFIX",  # Remove ClaudeCode shell wrapper
        "BASH_ENV",  # Remove custom bash environment
    ]

    for var in test_vars_to_remove:
        env.pop(var, None)

    # Ensure git has proper user configuration for test operations
    if "git" in cmd and any(op in cmd for op in ["commit", "merge", "rebase"]):
        # For git operations that require user info, ensure they're set
        user_name = env.get("GIT_AUTHOR_NAME", "Test User")
        user_email = env.get("GIT_AUTHOR_EMAIL", "test@example.com")
        env["GIT_AUTHOR_NAME"] = user_name
        env["GIT_AUTHOR_EMAIL"] = user_email
        env["GIT_COMMITTER_NAME"] = user_name
        env["GIT_COMMITTER_EMAIL"] = user_email

    # Run the command with clean environment
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            capture_output=capture_output,
            env=env,
            text=True,  # Handle text encoding properly
            **kwargs,
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            # Re-raise with more context for debugging
            raise subprocess.CalledProcessError(
                e.returncode, e.cmd, output=e.stdout, stderr=e.stderr
            ) from e
        return e
    except Exception as e:
        if check:
            raise RuntimeError(f"Git command failed: {cmd}") from e
        return subprocess.CompletedProcess(cmd, 1, "", str(e))


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
    """Mock ConfigManager instance with proper method signatures."""
    manager = Mock(spec=ConfigManager)

    # Configure the mock to return appropriate values for actual ConfigManager methods
    manager.get_enabled_providers.return_value = ["gemini"]
    manager.get_provider_config.return_value = mock_config_dict["providers"]["gemini"]
    manager.get_default_provider.return_value = "gemini"
    manager.get_server_config.return_value = mock_config_dict["server"]
    manager.get_debug_state.return_value = {
        "config_path": None,
        "server_config": mock_config_dict["server"],
        "provider_details": [
            {
                "name": "gemini",
                "type": "gemini",
                "enabled": True,
                "model": "gemini-1.5-flash",
                "has_api_key": False,
            }
        ],
    }

    # Add key_manager property mock
    key_manager_mock = Mock()
    key_manager_mock.list_stored_providers.return_value = []
    manager.key_manager = key_manager_mock

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
