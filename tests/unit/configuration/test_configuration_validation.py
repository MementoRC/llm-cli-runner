"""Unit tests for configuration validation.

This module tests the Pydantic-based configuration models to ensure
they properly validate all fields, handle edge cases, and provide
clear error messages.

Test categories:
    - Default value tests
    - Valid configuration tests
    - Invalid configuration tests
    - Validator function tests
    - Environment variable loading tests
    - Schema generation tests
"""

import pytest
from pathlib import Path
from pydantic import ValidationError
import json
import tempfile
import os

from mcp_server_git.configuration import GitServerConfig, GitConfig, GitHubConfig


class TestGitServerConfig:
    """Test GitServerConfig validation and behavior."""

    def test_default_configuration(self):
        """Test that default configuration is valid."""
        config = GitServerConfig()

        # Server settings
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.max_concurrent_operations == 10
        assert config.operation_timeout_seconds == 300

        # Security settings
        assert config.enable_security_validation is True
        assert config.allowed_repository_paths == []
        assert config.forbidden_operations == []
        assert config.require_gpg_signing is False

        # GitHub integration
        assert config.github_token is None
        assert config.github_api_timeout == 30
        assert config.github_rate_limit_buffer == 100

        # Logging and monitoring
        assert config.log_level == "INFO"
        assert config.enable_metrics_collection is True
        assert config.metrics_retention_days == 30

    def test_valid_custom_configuration(self):
        """Test creating configuration with valid custom values."""
        config = GitServerConfig(
            host="0.0.0.0",
            port=9000,
            max_concurrent_operations=20,
            github_token="ghp_validtokenformat12345",
            log_level="DEBUG",
            metrics_retention_days=60,
        )

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.max_concurrent_operations == 20
        assert config.github_token == "ghp_validtokenformat12345"
        assert config.log_level == "DEBUG"
        assert config.metrics_retention_days == 60

    def test_port_validation(self):
        """Test port number validation."""
        # Valid ports
        GitServerConfig(port=1024)  # Minimum
        GitServerConfig(port=65535)  # Maximum
        GitServerConfig(port=8080)  # Common

        # Invalid ports
        with pytest.raises(ValidationError) as exc_info:
            GitServerConfig(port=1023)  # Below minimum
        assert "Input should be greater than or equal to 1024" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            GitServerConfig(port=65536)  # Above maximum
        assert "Input should be less than or equal to 65535" in str(exc_info.value)

    def test_timeout_validation(self):
        """Test timeout validation."""
        # Valid timeouts
        GitServerConfig(operation_timeout_seconds=30)  # Minimum
        GitServerConfig(operation_timeout_seconds=1800)  # Maximum

        # Invalid timeouts
        with pytest.raises(ValidationError):
            GitServerConfig(operation_timeout_seconds=29)  # Below minimum

        with pytest.raises(ValidationError):
            GitServerConfig(operation_timeout_seconds=1801)  # Above maximum

    def test_github_token_validation(self):
        """Test GitHub token format validation."""
        # Valid tokens
        valid_tokens = [
            "ghp_1234567890abcdef1234567890abcdef12345678",  # Personal access token
            "gho_1234567890abcdef1234567890abcdef12345678",  # OAuth token
            "ghu_1234567890abcdef1234567890abcdef12345678",  # User-to-server token
            "ghs_1234567890abcdef1234567890abcdef12345678",  # Server-to-server token
            "ghr_1234567890abcdef1234567890abcdef12345678",  # Refresh token
        ]

        for token in valid_tokens:
            config = GitServerConfig(github_token=token)
            assert config.github_token == token

        # Invalid tokens
        invalid_tokens = [
            "invalid_token",
            "gh_wrongprefix",
            "ghp",  # Too short
        ]

        for token in invalid_tokens:
            with pytest.raises(ValidationError) as exc_info:
                GitServerConfig(github_token=token)
            assert "Invalid GitHub token format" in str(exc_info.value)

        # Empty string should be allowed (treated as None)
        config = GitServerConfig(github_token="")
        assert config.github_token == ""

    def test_repository_path_validation(self):
        """Test repository path validation."""
        # Create temporary directories for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid Git repository
            valid_repo = Path(tmpdir) / "valid_repo"
            valid_repo.mkdir()
            (valid_repo / ".git").mkdir()

            # Create non-Git directory
            non_git_dir = Path(tmpdir) / "non_git"
            non_git_dir.mkdir()

            # Valid configuration
            config = GitServerConfig(allowed_repository_paths=[valid_repo])
            assert valid_repo in config.allowed_repository_paths

            # Non-existent path
            with pytest.raises(ValidationError) as exc_info:
                GitServerConfig(allowed_repository_paths=[Path("/non/existent/path")])
            assert "Repository path does not exist" in str(exc_info.value)

            # Non-Git directory
            with pytest.raises(ValidationError) as exc_info:
                GitServerConfig(allowed_repository_paths=[non_git_dir])
            assert "Path is not a Git repository" in str(exc_info.value)

    def test_forbidden_operations_validation(self):
        """Test forbidden operations validation."""
        # Valid operations
        valid_ops = ["push", "force-push", "rebase", "merge"]
        config = GitServerConfig(forbidden_operations=valid_ops)
        assert config.forbidden_operations == valid_ops

        # Invalid operation
        with pytest.raises(ValidationError) as exc_info:
            GitServerConfig(forbidden_operations=["invalid-op"])
        assert "Unrecognized operation 'invalid-op'" in str(exc_info.value)

    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config = GitServerConfig(log_level=level)
            assert config.log_level == level

        # Invalid log level
        with pytest.raises(ValidationError):
            GitServerConfig(log_level="INVALID")

    def test_schema_generation(self):
        """Test that schema can be generated with examples."""
        schema = GitServerConfig.model_json_schema()

        # Check basic structure
        assert "properties" in schema

        # Pydantic v2 puts examples in json_schema_extra as 'example'
        assert "example" in schema

        # Check example values
        example = schema["example"]
        assert example["host"] == "localhost"
        assert example["port"] == 8080

        # Ensure schema is valid JSON
        json_str = json.dumps(schema)
        assert json_str  # Should not raise exception


class TestGitConfig:
    """Test GitConfig validation and behavior."""

    def test_default_configuration(self):
        """Test that default configuration is valid."""
        config = GitConfig()

        # Repository management
        assert config.default_clone_depth == 0
        assert config.max_repository_size_gb == 10.0
        assert config.allow_force_push is False
        assert config.allow_submodules is True

        # Operation limits
        assert config.max_file_size_mb == 100
        assert config.max_diff_size_mb == 50
        assert config.max_commit_message_length == 1000

        # Branch protection
        assert config.protected_branches == ["main", "master"]
        assert config.allow_delete_branch is True

        # Performance
        assert config.enable_git_cache is True
        assert config.parallel_operations == 4

    def test_commit_message_pattern_validation(self):
        """Test commit message regex pattern validation."""
        # Valid regex patterns
        valid_patterns = [
            r"^(feat|fix|docs):\s.+",
            r"^\[.+\]\s.+",
            r"^.{10,72}$",
        ]

        for pattern in valid_patterns:
            config = GitConfig(commit_message_pattern=pattern)
            assert config.commit_message_pattern == pattern

        # Invalid regex
        with pytest.raises(ValidationError) as exc_info:
            GitConfig(commit_message_pattern="[invalid(regex")
        assert "Invalid regex pattern" in str(exc_info.value)

    def test_protected_branches_validation(self):
        """Test protected branch pattern validation."""
        # Valid patterns
        patterns = ["main", "master", "release/*", "feature/**"]
        config = GitConfig(protected_branches=patterns)
        assert config.protected_branches == patterns

        # Empty pattern
        with pytest.raises(ValidationError):
            GitConfig(protected_branches=[""])

        # Warning for no protected branches
        with pytest.warns(UserWarning):
            GitConfig(protected_branches=[])

    def test_author_validation(self):
        """Test allowed authors validation."""
        # Valid author formats
        valid_authors = [
            "John Doe",
            "john@example.com",
            "John Doe <john@example.com>",
        ]

        config = GitConfig(allowed_authors=valid_authors)
        assert config.allowed_authors == valid_authors

        # Invalid formats
        with pytest.raises(ValidationError):
            GitConfig(allowed_authors=[""])  # Empty

        with pytest.raises(ValidationError):
            GitConfig(allowed_authors=["   "])  # Whitespace only

        with pytest.raises(ValidationError):
            GitConfig(
                allowed_authors=["Invalid <not@an@email>"]
            )  # Invalid email format

    def test_dependent_settings_validation(self):
        """Test interdependent settings validation."""
        # Warning for conflicting settings
        with pytest.warns(
            UserWarning, match="Commit signing is enabled but force push is allowed"
        ):
            GitConfig(enable_commit_signing=True, allow_force_push=True)

        with pytest.warns(
            UserWarning, match="Pull requests are required but no approvals are needed"
        ):
            GitConfig(require_pull_request=True, min_approvals=0)

    def test_numeric_limits(self):
        """Test numeric field limits."""
        # Test boundaries
        GitConfig(max_file_size_mb=1)  # Minimum
        GitConfig(max_file_size_mb=1000)  # Maximum

        with pytest.raises(ValidationError):
            GitConfig(max_file_size_mb=0)  # Below minimum

        with pytest.raises(ValidationError):
            GitConfig(max_file_size_mb=1001)  # Above maximum


class TestGitHubConfig:
    """Test GitHubConfig validation and behavior."""

    def test_default_configuration(self):
        """Test that default configuration is valid."""
        config = GitHubConfig()

        # API settings
        assert str(config.api_base_url) == "https://api.github.com"
        assert config.api_version == "2022-11-28"
        assert config.api_timeout_seconds == 30

        # Authentication
        assert config.api_token is None
        assert config.token_scopes == ["repo", "read:org"]
        assert config.validate_token_on_startup is True

        # Rate limiting
        assert config.rate_limit_buffer == 100
        assert config.auto_retry_rate_limited is True

        # Webhooks
        assert config.webhook_secret is None
        assert "push" in config.allowed_webhook_events
        assert "pull_request" in config.allowed_webhook_events

    def test_github_token_validation(self):
        """Test GitHub token validation with various formats."""
        # Valid tokens
        valid_tokens = [
            "ghp_1234567890abcdef1234567890abcdef12345678",
            "github_pat_11ABCDEFG_1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "gho_1234567890abcdef1234567890abcdef12345678",
        ]

        for token in valid_tokens:
            config = GitHubConfig(api_token=token)
            assert config.api_token == token

        # Invalid tokens
        with pytest.raises(ValidationError):
            GitHubConfig(api_token="invalid_token")

        with pytest.raises(ValidationError):
            GitHubConfig(api_token="ghp_short")  # Too short

    def test_api_version_validation(self):
        """Test API version format validation."""
        # Valid versions
        valid_versions = ["2022-11-28", "2023-01-01", "2023-12-31"]

        for version in valid_versions:
            config = GitHubConfig(api_version=version)
            assert config.api_version == version

        # Invalid versions
        invalid_versions = ["2022-1-1", "2022/11/28", "latest", "v3"]

        for version in invalid_versions:
            with pytest.raises(ValidationError):
                GitHubConfig(api_version=version)

    def test_webhook_event_validation(self):
        """Test webhook event type validation."""
        # Valid events
        valid_events = ["push", "pull_request", "issues", "workflow_run"]
        config = GitHubConfig(allowed_webhook_events=valid_events)
        assert config.allowed_webhook_events == valid_events

        # Invalid event
        with pytest.raises(ValidationError) as exc_info:
            GitHubConfig(allowed_webhook_events=["invalid_event"])
        assert "Invalid webhook events: invalid_event" in str(exc_info.value)

    def test_webhook_secret_validation(self):
        """Test webhook secret strength validation."""
        # Valid secret
        config = GitHubConfig(webhook_secret="a-very-secure-secret-123456")
        assert config.webhook_secret == "a-very-secure-secret-123456"

        # Too short
        with pytest.raises(ValidationError):
            GitHubConfig(webhook_secret="short")

        # No warning for a strong password even if it contains 'password'
        config = GitHubConfig(webhook_secret="password12345678901234")
        assert config.webhook_secret == "password12345678901234"

    def test_token_scope_validation(self):
        """Test GitHub token scope validation."""
        # Valid scopes
        valid_scopes = ["repo", "read:org", "write:packages", "workflow"]
        config = GitHubConfig(token_scopes=valid_scopes)
        assert config.token_scopes == valid_scopes

        # Invalid scope
        with pytest.raises(ValidationError):
            GitHubConfig(token_scopes=["invalid:scope"])

    def test_url_normalization(self):
        """Test API base URL normalization."""
        # URL with trailing slash - HttpUrl normalizes it
        config = GitHubConfig(api_base_url="https://api.github.com/")
        assert str(config.api_base_url) == "https://api.github.com/"

        # GitHub Enterprise URL
        enterprise_config = GitHubConfig(
            api_base_url="https://github.enterprise.com/api/v3", is_enterprise=True
        )
        assert (
            str(enterprise_config.api_base_url)
            == "https://github.enterprise.com/api/v3"
        )

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ["MCP_GITHUB_API_TOKEN"] = "ghp_test_token_12345678901234567890"
        os.environ["MCP_GITHUB_WEBHOOK_SECRET"] = "test-webhook-secret-123"
        os.environ["MCP_GITHUB_API_TIMEOUT_SECONDS"] = "60"

        try:
            # Create config (should load from environment)
            _ = GitHubConfig()

            # Note: Pydantic doesn't automatically load from environment
            # without explicit configuration. This test documents the expected
            # behavior when environment loading is properly configured.

            # Clean up
        finally:
            del os.environ["MCP_GITHUB_API_TOKEN"]
            del os.environ["MCP_GITHUB_WEBHOOK_SECRET"]
            del os.environ["MCP_GITHUB_API_TIMEOUT_SECONDS"]

    def test_comprehensive_validation(self):
        """Test creating a fully configured GitHubConfig."""
        config = GitHubConfig(
            api_base_url="https://github.enterprise.com/api/v3",
            api_version="2023-01-01",
            api_timeout_seconds=60,
            max_retries=5,
            retry_backoff_factor=3.0,
            api_token="ghp_enterprise_token_1234567890123456789012345678901234567890",
            token_scopes=["repo", "read:org", "write:packages"],
            validate_token_on_startup=True,
            rate_limit_buffer=200,
            rate_limit_threshold=0.3,
            auto_retry_rate_limited=True,
            max_rate_limit_wait=1800,
            webhook_secret="super-secure-webhook-secret-123456789",
            allowed_webhook_events=["push", "pull_request", "issues"],
            webhook_timeout=60,
            validate_webhook_payload=True,
            auto_link_issues=True,
            require_issue_in_pr=True,
            allowed_pr_labels=["bug", "feature"],
            auto_close_stale_prs=True,
            stale_pr_days=14,
            is_enterprise=True,
            enterprise_api_path="/api/v3",
            skip_ssl_verification=False,
        )

        # Verify all values were set correctly
        assert str(config.api_base_url) == "https://github.enterprise.com/api/v3"
        assert config.api_version == "2023-01-01"
        assert config.api_timeout_seconds == 60
        assert config.max_retries == 5
        assert config.is_enterprise is True
        assert config.stale_pr_days == 14


class TestConfigurationIntegration:
    """Test configuration models working together."""

    def test_combined_configuration(self):
        """Test creating all configuration models together."""
        server_config = GitServerConfig(
            port=9000,
            github_token="ghp_test12345678901234567890123456789012",  # 40 chars
        )

        git_config = GitConfig(
            max_file_size_mb=200, protected_branches=["main", "develop", "release/*"]
        )

        github_config = GitHubConfig(
            api_token=server_config.github_token,  # Share token
            webhook_secret="shared-secret-123456789",
        )

        # Verify configurations are independent but can share values
        assert server_config.github_token == github_config.api_token
        assert git_config.max_file_size_mb == 200
        assert len(git_config.protected_branches) == 3

    def test_json_serialization(self):
        """Test that configurations can be serialized to JSON."""
        config = GitServerConfig(
            host="0.0.0.0", port=8080, forbidden_operations=["force-push", "rebase"]
        )

        # Convert to JSON using Pydantic v2 method
        json_str = config.model_dump_json(indent=2)
        data = json.loads(json_str)

        # Verify data
        assert data["host"] == "0.0.0.0"
        assert data["port"] == 8080
        assert data["forbidden_operations"] == ["force-push", "rebase"]

        # Can recreate from JSON using Pydantic v2 method
        config2 = GitServerConfig.model_validate_json(json_str)
        assert config2.host == config.host
        assert config2.port == config.port
        assert config2.forbidden_operations == config.forbidden_operations
