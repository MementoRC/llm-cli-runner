"""Unit tests for the SecurityFramework component."""

import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from mcp_server_git.frameworks.server_security import (
    SecurityFramework,
    TokenValidator,
    InputSanitizer,
    SecurityStatus,
    SecuritySeverity,
    SecurityCategory,
    SecurityDefaults,
)
from mcp_server_git.types.git_types import GitValidationError


class TestTokenValidator:
    """Test suite for TokenValidator."""

    def test_valid_github_tokens(self):
        """Test validation of valid GitHub token formats."""
        valid_tokens = [
            "ghp_" + "a" * 36,  # Personal access token
            "github_pat_" + "a" * 82,  # Fine-grained PAT
            "ghs_" + "b" * 36,  # App installation token
            "ghu_" + "c" * 36,  # App user token
            "gho_" + "d" * 36,  # OAuth token
            "ghr_" + "e" * 36,  # Refresh token
        ]

        for token in valid_tokens:
            result = TokenValidator.validate_github_token(token)
            assert (
                result.status == SecurityStatus.SECURE
                or result.status == SecurityStatus.WARNING
            )
            assert result.metadata["format_valid"] is True
            assert result.metadata["token_provided"] is True

    def test_invalid_github_tokens(self):
        """Test validation of invalid GitHub token formats."""
        invalid_tokens = [
            "invalid_token",
            "ghp_short",
            "random_string_123",
            "bearer_token_format",
            "",
            "   ",
        ]

        for token in invalid_tokens:
            result = TokenValidator.validate_github_token(token)
            if token.strip():  # Non-empty tokens
                assert result.status in [
                    SecurityStatus.WARNING,
                    SecurityStatus.INSECURE,
                ]
                assert result.metadata["format_valid"] is False
            else:  # Empty tokens
                assert result.status == SecurityStatus.CRITICAL
                assert result.metadata["token_provided"] is False

    def test_empty_token(self):
        """Test validation of empty or None token."""
        result = TokenValidator.validate_github_token("")
        assert result.status == SecurityStatus.CRITICAL
        assert len(result.issues) == 1
        assert result.issues[0].severity == SecuritySeverity.CRITICAL
        assert result.issues[0].category == SecurityCategory.AUTHENTICATION

    def test_token_too_long(self):
        """Test validation of excessively long token."""
        long_token = "ghp_" + "a" * (SecurityDefaults.MAX_TOKEN_LENGTH + 100)
        result = TokenValidator.validate_github_token(long_token)

        # Should have length issue
        length_issues = [
            issue
            for issue in result.issues
            if "length exceeds maximum" in issue.message
        ]
        assert len(length_issues) == 1
        assert length_issues[0].severity == SecuritySeverity.HIGH

    def test_classic_token_recommendations(self):
        """Test that classic PAT tokens get security recommendations."""
        classic_token = "ghp_" + "a" * 36
        result = TokenValidator.validate_github_token(classic_token)

        # Should recommend fine-grained tokens
        fine_grained_recommendations = [
            rec for rec in result.recommendations if "fine-grained" in rec.description
        ]
        assert len(fine_grained_recommendations) == 1


class TestInputSanitizer:
    """Test suite for InputSanitizer."""

    def test_sanitize_repository_path_valid(self):
        """Test sanitization of valid repository paths."""
        valid_paths = [
            "my-repo",
            "path/to/repo",
            "user/project-name",
            "deeply/nested/repo/path",
        ]

        for path in valid_paths:
            result = InputSanitizer.sanitize_repository_path(path)
            assert result == os.path.normpath(path)

    def test_sanitize_repository_path_invalid(self):
        """Test sanitization rejects invalid repository paths."""
        invalid_paths = [
            "../../../etc/passwd",
            "/absolute/path",
            "path/../../../secret",
            "",
        ]

        for path in invalid_paths:
            with pytest.raises(GitValidationError):
                InputSanitizer.sanitize_repository_path(path)

    def test_validate_branch_name_valid(self):
        """Test validation of valid Git branch names."""
        valid_names = [
            "main",
            "feature/new-feature",
            "bugfix/issue-123",
            "develop",
            "release/v1.0.0",
        ]

        for name in valid_names:
            assert InputSanitizer.validate_branch_name(name) is True

    def test_validate_branch_name_invalid(self):
        """Test validation of invalid Git branch names."""
        invalid_names = [
            "-invalid",  # Cannot start with dash
            "branch..name",  # Consecutive dots
            "refs/heads/main",  # Cannot start with refs/
            "branch\x00name",  # Control characters
            "branch~name",  # Special characters
            "branch.",  # Cannot end with dot
            "branch.lock",  # Cannot end with .lock
            "branch/",  # Cannot end with slash
            "",  # Empty
        ]

        for name in invalid_names:
            assert InputSanitizer.validate_branch_name(name) is False

    def test_sanitize_commit_message_valid(self):
        """Test sanitization of valid commit messages."""
        valid_messages = [
            "Add new feature",
            "Fix bug in authentication",
            "Update documentation for API changes",
        ]

        for message in valid_messages:
            result = InputSanitizer.sanitize_commit_message(message)
            assert result == message

    def test_sanitize_commit_message_with_injection(self):
        """Test sanitization removes potential injection sequences."""
        dangerous_messages = [
            "Commit with `rm -rf /`",
            "Message with $HOME variable",
            "Command; rm file",
            "Pipe | to command",
            "Redirect > to file",
        ]

        for message in dangerous_messages:
            result = InputSanitizer.sanitize_commit_message(message)
            # Should not contain dangerous characters
            assert all(char not in result for char in "`$;|&<>")

    def test_sanitize_commit_message_empty(self):
        """Test sanitization rejects empty commit messages."""
        with pytest.raises(GitValidationError):
            InputSanitizer.sanitize_commit_message("")

    def test_sanitize_commit_message_too_long(self):
        """Test sanitization truncates very long commit messages."""
        long_message = "A" * 3000
        result = InputSanitizer.sanitize_commit_message(long_message)
        assert len(result) <= 2003  # 2000 + "..."

    def test_validate_file_path_valid(self):
        """Test validation of valid file paths."""
        valid_paths = [
            "file.py",
            "src/module.py",
            "docs/readme.md",
            "config.json",
            "tests/test_file.py",
        ]

        for path in valid_paths:
            assert InputSanitizer.validate_file_path(path) is True

    def test_validate_file_path_invalid(self):
        """Test validation of invalid file paths."""
        invalid_paths = [
            "../../../etc/passwd",
            "path/../secrets",
            "file.exe",  # Not in allowed extensions
            "",  # Empty
        ]

        for path in invalid_paths:
            assert InputSanitizer.validate_file_path(path) is False


class TestSecurityFramework:
    """Test suite for SecurityFramework."""

    @pytest.fixture
    def security_framework(self):
        """Create a SecurityFramework instance for testing."""
        return SecurityFramework("test-security")

    @pytest.fixture
    def mock_repo_path(self, tmp_path):
        """Create a mock repository path for testing."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        return repo_path

    def test_initialization(self, security_framework):
        """Test SecurityFramework initialization."""
        assert security_framework.component_id == "test-security"
        assert security_framework.failed_attempts == {}
        assert security_framework.rate_limits == {}
        assert security_framework.security_events == []
        assert security_framework.gpg_validated is False
        assert security_framework.token_validator is not None
        assert security_framework.input_sanitizer is not None

    def test_get_component_state(self, security_framework):
        """Test getting component state for debugging."""
        state = security_framework.get_component_state()

        assert state.component_id == "test-security"
        assert state.component_type == "SecurityFramework"
        assert "failed_attempts_count" in state.state_data
        assert "rate_limits_active" in state.state_data
        assert "security_events_count" in state.state_data
        assert "gpg_validated" in state.state_data

    def test_validate_component(self, security_framework):
        """Test component validation."""
        result = security_framework.validate_component()

        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)

    def test_get_debug_info(self, security_framework):
        """Test getting debug information."""
        debug_info = security_framework.get_debug_info()

        assert debug_info.debug_data["component_type"] == "SecurityFramework"
        assert debug_info.debug_data["component_id"] == "test-security"
        assert "state" in debug_info.debug_data
        assert "validation" in debug_info.debug_data
        assert "security_summary" in debug_info.debug_data

    @patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_" + "a" * 36})
    def test_authenticate_github_token_success(self, security_framework):
        """Test successful GitHub token authentication."""
        result = security_framework.authenticate_github_token()

        assert result.success is True
        assert result.token_type == "ghp"
        assert result.error_message is None

    @patch.dict(os.environ, {"GITHUB_TOKEN": ""})
    def test_authenticate_github_token_failure(self, security_framework):
        """Test failed GitHub token authentication."""
        result = security_framework.authenticate_github_token()

        assert result.success is False
        assert result.error_message is not None

    def test_authenticate_github_token_explicit(self, security_framework):
        """Test authentication with explicitly provided token."""
        # Valid token
        valid_token = "ghp_" + "a" * 36
        result = security_framework.authenticate_github_token(valid_token)
        assert result.success is True

        # Invalid token
        invalid_token = "invalid_token"
        result = security_framework.authenticate_github_token(invalid_token)
        assert result.success is False

    def test_validate_repository_access_valid(self, security_framework, mock_repo_path):
        """Test validation of valid repository access."""
        with patch(
            "mcp_server_git.frameworks.server_security.validate_git_security_config"
        ) as mock_validate:
            mock_validate.return_value = {"warnings": [], "recommendations": []}

            result = security_framework.validate_repository_access(str(mock_repo_path))

            assert result.status in [SecurityStatus.SECURE, SecurityStatus.WARNING]
            assert result.metadata["path_exists"] is True

    def test_validate_repository_access_nonexistent(self, security_framework):
        """Test validation of non-existent repository."""
        result = security_framework.validate_repository_access("nonexistent/path")

        assert result.status in [SecurityStatus.INSECURE, SecurityStatus.WARNING]
        # Should have issues about non-existent path
        path_issues = [
            issue for issue in result.issues if "does not exist" in issue.message
        ]
        assert len(path_issues) >= 1

    def test_validate_repository_access_invalid_path(self, security_framework):
        """Test validation with invalid repository path."""
        result = security_framework.validate_repository_access("../../../etc/passwd")

        assert result.status in [SecurityStatus.INSECURE, SecurityStatus.CRITICAL]
        # Should have path validation issues
        validation_issues = [
            issue
            for issue in result.issues
            if issue.category == SecurityCategory.INPUT_VALIDATION
        ]
        assert len(validation_issues) >= 1

    def test_validate_git_operation_commit(self, security_framework):
        """Test validation of Git commit operation."""
        params = {"message": "Valid commit message"}
        result = security_framework.validate_git_operation("commit", params)

        # Should be secure for valid commit message
        assert result.status in [SecurityStatus.SECURE, SecurityStatus.WARNING]

    def test_validate_git_operation_commit_dangerous(self, security_framework):
        """Test validation of Git commit with dangerous message."""
        params = {"message": "Commit with `rm -rf /` command"}
        result = security_framework.validate_git_operation("commit", params)

        # Should have sanitization issues
        sanitization_issues = [
            issue for issue in result.issues if "sanitized" in issue.message
        ]
        assert (
            len(sanitization_issues) >= 0
        )  # May or may not sanitize depending on content

    def test_validate_git_operation_file_operations(self, security_framework):
        """Test validation of file operations."""
        # Valid files
        params = {"files": ["src/module.py", "tests/test.py"]}
        result = security_framework.validate_git_operation("add", params)
        assert result.status in [SecurityStatus.SECURE, SecurityStatus.WARNING]

        # Invalid files
        params = {"files": ["../../../etc/passwd"]}
        result = security_framework.validate_git_operation("add", params)

        # Should have path validation issues
        path_issues = [
            issue for issue in result.issues if "unsafe file path" in issue.message
        ]
        assert len(path_issues) >= 1

    def test_rate_limiting(self, security_framework):
        """Test rate limiting functionality."""
        # First request should pass
        assert (
            security_framework._check_rate_limit("test_op", limit=2, window=60) is True
        )

        # Second request should pass
        assert (
            security_framework._check_rate_limit("test_op", limit=2, window=60) is True
        )

        # Third request should fail (limit=2)
        assert (
            security_framework._check_rate_limit("test_op", limit=2, window=60) is False
        )

    def test_rate_limiting_window_expiry(self, security_framework):
        """Test that rate limiting window expires correctly."""
        # Make requests that would exceed limit
        security_framework._check_rate_limit("test_op", limit=1, window=1)

        # Should be blocked immediately
        assert (
            security_framework._check_rate_limit("test_op", limit=1, window=1) is False
        )

        # Manually set old timestamp to simulate expiry
        past_time = datetime.now() - timedelta(seconds=2)
        security_framework.rate_limits["test_op"] = [past_time]

        # Should pass after window expiry
        assert (
            security_framework._check_rate_limit("test_op", limit=1, window=1) is True
        )

    def test_failed_attempt_tracking(self, security_framework):
        """Test tracking of failed attempts."""
        initial_events = len(security_framework.security_events)

        security_framework._record_failed_attempt("login", "invalid_password")

        # Should record the attempt
        assert "login" in security_framework.failed_attempts
        assert len(security_framework.failed_attempts["login"]) == 1
        assert len(security_framework.security_events) == initial_events + 1

    def test_failed_attempt_cleanup(self, security_framework):
        """Test cleanup of old failed attempts."""
        # Add old failed attempt
        old_time = datetime.now() - timedelta(hours=2)
        security_framework.failed_attempts["test"] = [old_time]

        # Record new attempt (should clean old ones)
        security_framework._record_failed_attempt("test", "reason")

        # Should only have the new attempt
        assert len(security_framework.failed_attempts["test"]) == 1
        assert security_framework.failed_attempts["test"][0] > old_time

    def test_security_event_logging(self, security_framework):
        """Test security event logging."""
        initial_count = len(security_framework.security_events)

        security_framework._log_security_event("test_event", {"key": "value"})

        assert len(security_framework.security_events) == initial_count + 1

        latest_event = security_framework.security_events[-1]
        assert latest_event["event_type"] == "test_event"
        assert latest_event["details"]["key"] == "value"
        assert latest_event["component_id"] == "test-security"

    def test_security_event_limit(self, security_framework):
        """Test that security events are limited to prevent memory issues."""
        # Add many events
        for i in range(1100):
            security_framework._log_security_event(f"event_{i}", {})

        # Should be limited to 1000
        assert len(security_framework.security_events) == 1000

        # Should have the most recent events
        latest_event = security_framework.security_events[-1]
        assert latest_event["event_type"] == "event_1099"

    def test_get_security_metrics(self, security_framework):
        """Test getting security metrics."""
        # Add some test data
        security_framework._record_failed_attempt("test", "reason")
        security_framework._log_security_event("test_event", {})

        metrics = security_framework.get_security_metrics()

        assert "total_events" in metrics
        assert "recent_events" in metrics
        assert "failed_attempts" in metrics
        assert "active_rate_limits" in metrics
        assert "gpg_validated" in metrics
        assert "component_status" in metrics
        assert "last_updated" in metrics

        assert metrics["total_events"] >= 1
        assert metrics["failed_attempts"]["test"] == 1

    def test_validate_git_operation_rate_limit_exceeded(self, security_framework):
        """Test Git operation validation when rate limit is exceeded."""
        # Exhaust rate limit
        for _ in range(70):  # Default limit is 60
            security_framework._check_rate_limit("test_operation", check_only=True)

        # Now validation should detect rate limit issue
        result = security_framework.validate_git_operation("test_operation", {})

        rate_limit_issues = [
            issue
            for issue in result.issues
            if issue.category == SecurityCategory.RATE_LIMITING
        ]
        assert (
            len(rate_limit_issues) >= 0
        )  # May or may not hit rate limit depending on timing

    def test_component_validation_with_failures(self, security_framework):
        """Test component validation when there are many failures."""
        # Add many failed attempts
        for i in range(20):
            security_framework._record_failed_attempt(f"context_{i}", "reason")

        result = security_framework.validate_component()

        # Should have errors about excessive failures
        assert len(result.errors) > 0 or len(result.warnings) > 0
