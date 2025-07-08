"""Integration tests for validating complete infrastructure setup."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestInfrastructureIntegration:
    """Test complete infrastructure setup and development workflow."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    def test_package_structure_exists(self, project_root: Path):
        """Verify all required package directories exist."""
        required_dirs = [
            "src/mcp_server_cheap_llm",
            "src/mcp_server_cheap_llm/core",
            "src/mcp_server_cheap_llm/server",
            "src/mcp_server_cheap_llm/utils",
            "tests",
            "tests/unit",
            "tests/integration",
            "tests/e2e",
        ]

        for dir_path in required_dirs:
            assert (project_root / dir_path).exists(), f"Missing directory: {dir_path}"
            assert (project_root / dir_path / "__init__.py").exists(), (
                f"Missing __init__.py in {dir_path}"
            )

    def test_configuration_files_exist(self, project_root: Path):
        """Verify all configuration files are present."""
        required_files = [
            "pyproject.toml",
            ".gitignore",
            ".pre-commit-config.yaml",
            "README.md",
        ]

        for file_path in required_files:
            assert (project_root / file_path).exists(), f"Missing file: {file_path}"

    def test_dependencies_import(self):
        """Verify all production dependencies can be imported."""
        dependencies = [
            "mcp",
            "pydantic",
            "aiohttp",
            "httpx",
            "llama_cpp",
            "psutil",
            "dotenv",
            "structlog",
        ]

        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError as e:
                pytest.fail(f"Failed to import {dep}: {e}")

    def test_package_imports(self):
        """Verify package modules can be imported."""
        imports = [
            "mcp_server_cheap_llm",
            "mcp_server_cheap_llm.core.models",
            "mcp_server_cheap_llm.server.handlers",
            "mcp_server_cheap_llm.utils.config",
            "mcp_server_cheap_llm.utils.errors",
            "mcp_server_cheap_llm.utils.logging",
        ]

        for module in imports:
            try:
                __import__(module)
            except ImportError as e:
                pytest.fail(f"Failed to import {module}: {e}")

    def test_development_tools_available(self):
        """Verify development tools are accessible via pixi."""
        tools = [
            ["pixi", "run", "-e", "dev", "pytest", "--version"],
            ["pixi", "run", "-e", "dev", "ruff", "--version"],
            ["pixi", "run", "-e", "dev", "pyright", "--version"],
            ["pixi", "run", "-e", "dev", "python", "-m", "pre_commit", "--version"],
        ]

        for tool_cmd in tools:
            result = subprocess.run(tool_cmd, capture_output=True, text=True)
            assert result.returncode == 0, (
                f"Failed to run {' '.join(tool_cmd)}: {result.stderr}"
            )

    def test_quality_commands_work(self):
        """Verify all quality check commands execute successfully."""
        commands = [
            ["pixi", "run", "test", "--co"],  # collect only
            ["pixi", "run", "lint"],
            ["pixi", "run", "typecheck"],
        ]

        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, (
                f"Command {' '.join(cmd)} failed: {result.stderr}"
            )

    def test_git_ignore_patterns(self, project_root: Path):
        """Verify .gitignore contains essential patterns."""
        gitignore_path = project_root / ".gitignore"
        content = gitignore_path.read_text()

        essential_patterns = [
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".env",
            ".taskmaster/",
            ".mcp.json",
            "*.pyc",
        ]

        for pattern in essential_patterns:
            assert pattern in content, f"Missing pattern in .gitignore: {pattern}"

    def test_pre_commit_hooks_configured(self, project_root: Path):
        """Verify pre-commit hooks are properly configured."""
        config_path = project_root / ".pre-commit-config.yaml"
        content = config_path.read_text()

        required_hooks = [
            "bandit",
            "ruff",
            "trailing-whitespace",
            "check-yaml",
            "pytest",
        ]

        for hook in required_hooks:
            assert hook in content, f"Missing hook in pre-commit config: {hook}"
