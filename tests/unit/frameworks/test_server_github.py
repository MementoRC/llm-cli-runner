"""
Unit tests for GitHub service implementation.

This module provides comprehensive unit tests for the GitHubService class,
following the established testing patterns and TDD compliance requirements.
Tests cover service lifecycle, GitHub API operations, error handling,
state management, and DebuggableComponent protocol implementation.

Test structure follows the established patterns:
    - Setup and teardown with proper mocking
    - Comprehensive coverage of public API
    - Error condition testing
    - State validation and debugging
    - Protocol compliance verification

Critical for TDD Compliance:
    These tests define the required interface and behavior for GitHubService.
    The implementation must satisfy these tests to prevent LLM compliance issues.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_server_git.configuration.github_config import GitHubConfig
from src.mcp_server_git.frameworks.server_github import (
    GitHubOperationResult,
    GitHubService,
    GitHubServiceConfig,
    GitHubServiceState,
)


class TestGitHubServiceConfig:
    """Test GitHub service configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = GitHubServiceConfig()

        assert isinstance(config.github_config, GitHubConfig)
        assert config.enable_webhooks is False
        assert config.enable_cli_operations is True
        assert config.enable_rate_limiting is True
        assert config.max_concurrent_operations == 10
        assert config.operation_timeout == 30.0
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0

    def test_custom_configuration(self):
        """Test custom configuration values."""
        github_config = GitHubConfig(
            api_token="ghp_" + "y" * 36
        )  # Valid length GitHub token
        config = GitHubServiceConfig(
            github_config=github_config,
            enable_webhooks=True,
            enable_cli_operations=False,
            max_concurrent_operations=5,
            operation_timeout=60.0,
        )

        assert config.github_config == github_config
        assert config.enable_webhooks is True
        assert config.enable_cli_operations is False
        assert config.max_concurrent_operations == 5
        assert config.operation_timeout == 60.0


class TestGitHubOperationResult:
    """Test GitHub operation result data structure."""

    def test_successful_result(self):
        """Test successful operation result."""
        result = GitHubOperationResult(
            success=True,
            operation="get_pr_checks",
            data={"checks": []},
            execution_time=1.5,
        )

        assert result.success is True
        assert result.operation == "get_pr_checks"
        assert result.data == {"checks": []}
        assert result.error_message is None
        assert result.status_code is None
        assert result.execution_time == 1.5
        assert result.retry_count == 0

    def test_failed_result(self):
        """Test failed operation result."""
        result = GitHubOperationResult(
            success=False,
            operation="get_pr_details",
            error_message="API rate limit exceeded",
            status_code=403,
            retry_count=2,
        )

        assert result.success is False
        assert result.operation == "get_pr_details"
        assert result.error_message == "API rate limit exceeded"
        assert result.status_code == 403
        assert result.retry_count == 2
        assert result.data == {}


class TestGitHubServiceState:
    """Test GitHub service state management."""

    def test_default_state(self):
        """Test default state values."""
        state = GitHubServiceState()

        assert state.is_initialized is False
        assert state.is_running is False
        assert state.operation_count == 0
        assert state.error_count == 0
        assert state.last_operation_time is None
        assert state.active_operations == 0
        assert state.rate_limit_status == {}
        assert state.configuration is None


class TestGitHubService:
    """Test main GitHub service functionality."""

    @pytest.fixture
    def github_config(self):
        """Provide GitHub configuration for testing."""
        return GitHubConfig(api_token="ghp_" + "x" * 36)  # Valid length GitHub token

    @pytest.fixture
    def service_config(self, github_config):
        """Provide service configuration for testing."""
        return GitHubServiceConfig(
            github_config=github_config,
            max_concurrent_operations=5,
            operation_timeout=15.0,
        )

    @pytest.fixture
    def github_service(self, service_config):
        """Provide GitHub service instance for testing."""
        return GitHubService(config=service_config)

    def test_initialization(self, service_config):
        """Test GitHub service initialization."""
        service = GitHubService(config=service_config)

        assert service._config == service_config
        assert isinstance(service._state, GitHubServiceState)
        assert service._state.configuration == service_config
        assert isinstance(service._executor, ThreadPoolExecutor)
        assert not service._state.is_initialized
        assert not service._state.is_running

    def test_initialization_default_config(self):
        """Test GitHub service initialization with default config."""
        service = GitHubService()

        assert isinstance(service._config, GitHubServiceConfig)
        assert isinstance(service._state, GitHubServiceState)
        assert not service._state.is_initialized
        assert not service._state.is_running

    @pytest.mark.asyncio
    async def test_start_service(self, github_service):
        """Test starting the GitHub service."""
        with (
            patch.object(
                github_service, "_validate_configuration", new_callable=AsyncMock
            ) as mock_validate,
            patch.object(
                github_service, "_initialize_rate_limiting", new_callable=AsyncMock
            ) as mock_rate_limit,
        ):
            await github_service.start()

            assert github_service._state.is_initialized
            assert github_service._state.is_running
            mock_validate.assert_called_once()
            mock_rate_limit.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_service_already_running(self, github_service):
        """Test starting service when already running."""
        github_service._state.is_running = True

        with patch.object(
            github_service, "_validate_configuration", new_callable=AsyncMock
        ) as mock_validate:
            await github_service.start()
            mock_validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_service_validation_failure(self, github_service):
        """Test service start with configuration validation failure."""
        with patch.object(
            github_service, "_validate_configuration", new_callable=AsyncMock
        ) as mock_validate:
            mock_validate.side_effect = Exception("Invalid token")

            with pytest.raises(Exception, match="Invalid token"):
                await github_service.start()

            assert not github_service._state.is_initialized
            assert not github_service._state.is_running

    @pytest.mark.asyncio
    async def test_stop_service(self, github_service):
        """Test stopping the GitHub service."""
        github_service._state.is_running = True
        github_service._state.active_operations = 0

        await github_service.stop()

        assert not github_service._state.is_running

    @pytest.mark.asyncio
    async def test_stop_service_with_active_operations(self, github_service):
        """Test stopping service with active operations."""
        github_service._state.is_running = True
        github_service._state.active_operations = 2

        # Mock the active operations to decrease over time
        async def mock_decrease_operations():
            await asyncio.sleep(0.1)
            github_service._state.active_operations = 0

        # Start the mock decrease task
        asyncio.create_task(mock_decrease_operations())

        await github_service.stop()

        assert not github_service._state.is_running
        assert github_service._state.active_operations == 0


class TestGitHubServiceOperations:
    """Test GitHub service API operations."""

    @pytest.fixture
    def github_service(self):
        """Provide GitHub service for operation testing."""
        return GitHubService()

    @pytest.mark.asyncio
    async def test_get_pr_checks_success(self, github_service):
        """Test successful PR checks retrieval."""
        mock_result = {"check_runs": [{"name": "test", "status": "completed"}]}

        with patch.object(
            github_service, "_execute_github_operation", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = GitHubOperationResult(
                success=True, operation="get_pr_checks", data=mock_result
            )

            result = await github_service.get_pr_checks("owner", "repo", 123)

            assert result.success
            assert result.operation == "get_pr_checks"
            assert result.data == mock_result
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_failing_jobs_success(self, github_service):
        """Test successful failing jobs retrieval."""
        mock_result = {"failing_jobs": [{"name": "test-job", "status": "failed"}]}

        with patch.object(
            github_service, "_execute_github_operation", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = GitHubOperationResult(
                success=True, operation="get_failing_jobs", data=mock_result
            )

            result = await github_service.get_failing_jobs("owner", "repo", 123)

            assert result.success
            assert result.operation == "get_failing_jobs"
            assert result.data == mock_result

    @pytest.mark.asyncio
    async def test_get_workflow_run_success(self, github_service):
        """Test successful workflow run retrieval."""
        mock_result = {"run": {"id": 123, "status": "completed"}}

        with patch.object(
            github_service, "_execute_github_operation", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = GitHubOperationResult(
                success=True, operation="get_workflow_run", data=mock_result
            )

            result = await github_service.get_workflow_run("owner", "repo", 123)

            assert result.success
            assert result.operation == "get_workflow_run"
            assert result.data == mock_result

    @pytest.mark.asyncio
    async def test_get_pr_details_success(self, github_service):
        """Test successful PR details retrieval."""
        mock_result = {"pr": {"number": 123, "title": "Test PR"}}

        with patch.object(
            github_service, "_execute_github_operation", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = GitHubOperationResult(
                success=True, operation="get_pr_details", data=mock_result
            )

            result = await github_service.get_pr_details("owner", "repo", 123)

            assert result.success
            assert result.operation == "get_pr_details"
            assert result.data == mock_result

    @pytest.mark.asyncio
    async def test_list_pull_requests_success(self, github_service):
        """Test successful pull requests listing."""
        mock_result = {"prs": [{"number": 123}, {"number": 124}]}

        with patch.object(
            github_service, "_execute_github_operation", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = GitHubOperationResult(
                success=True, operation="list_pull_requests", data=mock_result
            )

            result = await github_service.list_pull_requests("owner", "repo")

            assert result.success
            assert result.operation == "list_pull_requests"
            assert result.data == mock_result

    @pytest.mark.asyncio
    async def test_get_pr_status_success(self, github_service):
        """Test successful PR status retrieval."""
        mock_result = {"status": {"state": "success"}}

        with patch.object(
            github_service, "_execute_github_operation", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = GitHubOperationResult(
                success=True, operation="get_pr_status", data=mock_result
            )

            result = await github_service.get_pr_status("owner", "repo", 123)

            assert result.success
            assert result.operation == "get_pr_status"
            assert result.data == mock_result

    @pytest.mark.asyncio
    async def test_get_pr_files_success(self, github_service):
        """Test successful PR files retrieval."""
        mock_result = {"files": [{"filename": "test.py", "status": "modified"}]}

        with patch.object(
            github_service, "_execute_github_operation", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = GitHubOperationResult(
                success=True, operation="get_pr_files", data=mock_result
            )

            result = await github_service.get_pr_files("owner", "repo", 123)

            assert result.success
            assert result.operation == "get_pr_files"
            assert result.data == mock_result


class TestGitHubServiceExecutionFramework:
    """Test GitHub service operation execution framework."""

    @pytest.fixture
    def github_service(self):
        """Provide GitHub service for execution testing."""
        return GitHubService()

    @pytest.mark.asyncio
    async def test_execute_operation_success(self, github_service):
        """Test successful operation execution."""
        mock_func = AsyncMock(return_value={"result": "success"})

        with (
            patch.object(github_service, "_check_rate_limits", new_callable=AsyncMock),
            patch.object(github_service, "_record_operation_history") as mock_record,
        ):
            result = await github_service._execute_github_operation(
                "test_operation", mock_func, param1="value1"
            )

            assert result.success
            assert result.operation == "test_operation"
            assert result.data == {"result": "success"}
            assert result.execution_time > 0
            assert github_service._state.operation_count == 1
            assert github_service._state.active_operations == 0
            mock_record.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_operation_failure(self, github_service):
        """Test operation execution with failure."""
        mock_func = AsyncMock(side_effect=Exception("API Error"))

        with (
            patch.object(github_service, "_check_rate_limits", new_callable=AsyncMock),
            patch.object(github_service, "_record_operation_history") as mock_record,
        ):
            result = await github_service._execute_github_operation(
                "test_operation", mock_func, param1="value1"
            )

            assert not result.success
            assert result.operation == "test_operation"
            assert result.error_message == "API Error"
            assert result.execution_time > 0
            assert github_service._state.operation_count == 1
            assert github_service._state.error_count == 1
            assert github_service._state.active_operations == 0
            mock_record.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limiting_enabled(self, github_service):
        """Test operation execution with rate limiting enabled."""
        github_service._config.enable_rate_limiting = True
        mock_func = AsyncMock(return_value={})

        with patch.object(
            github_service, "_check_rate_limits", new_callable=AsyncMock
        ) as mock_rate_check:
            await github_service._execute_github_operation("test_op", mock_func)
            mock_rate_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limiting_disabled(self, github_service):
        """Test operation execution with rate limiting disabled."""
        github_service._config.enable_rate_limiting = False
        mock_func = AsyncMock(return_value={})

        with patch.object(
            github_service, "_check_rate_limits", new_callable=AsyncMock
        ) as mock_rate_check:
            await github_service._execute_github_operation("test_op", mock_func)
            mock_rate_check.assert_not_called()

    def test_record_operation_history(self, github_service):
        """Test operation history recording."""
        result = GitHubOperationResult(
            success=True, operation="test_op", execution_time=1.5
        )

        github_service._record_operation_history(result)

        assert len(github_service._operation_history) == 1
        history_entry = github_service._operation_history[0]
        assert history_entry["operation"] == "test_op"
        assert history_entry["success"] is True
        assert history_entry["execution_time"] == 1.5
        assert "timestamp" in history_entry

    def test_operation_history_limit(self, github_service):
        """Test operation history size limit."""
        # Add more than 100 operations
        for i in range(105):
            result = GitHubOperationResult(
                success=True, operation=f"test_op_{i}", execution_time=1.0
            )
            github_service._record_operation_history(result)

        # Should only keep last 100
        assert len(github_service._operation_history) == 100
        assert github_service._operation_history[0]["operation"] == "test_op_5"
        assert github_service._operation_history[-1]["operation"] == "test_op_104"


class TestGitHubServiceDebuggableComponent:
    """Test DebuggableComponent protocol implementation."""

    @pytest.fixture
    def github_service(self):
        """Provide GitHub service for debugging tests."""
        service = GitHubService()
        service._state.is_running = True
        service._state.operation_count = 10
        service._state.error_count = 2
        service._state.last_operation_time = datetime.now()
        return service

    def test_get_component_state(self, github_service):
        """Test component state retrieval."""
        state = github_service.get_component_state()

        # Check that it returns an object with the required protocol properties
        assert hasattr(state, "component_id")
        assert hasattr(state, "component_type")
        assert hasattr(state, "state_data")
        assert hasattr(state, "last_updated")

        assert state.component_type == "GitHubService"
        assert "github_service_" in state.component_id
        assert isinstance(state.state_data, dict)
        assert isinstance(state.last_updated, datetime)

    def test_get_component_state_unhealthy(self, github_service):
        """Test component state when unhealthy."""
        github_service._state.error_count = 15  # High error rate

        state = github_service.get_component_state()

        # Verify state data reflects the unhealthy condition
        assert state.state_data["error_count"] == 15
        assert state.state_data["operation_count"] == 10

    def test_validate_component_success(self, github_service):
        """Test successful component validation."""
        github_service._state.is_initialized = True

        result = github_service.validate_component()

        # Check that it returns an object with the required protocol properties
        assert hasattr(result, "is_valid")
        assert hasattr(result, "validation_errors")
        assert hasattr(result, "validation_warnings")
        assert hasattr(result, "validation_timestamp")

        assert result.is_valid is True
        assert len(result.validation_errors) == 0
        assert isinstance(result.validation_timestamp, datetime)

    def test_validate_component_issues(self, github_service):
        """Test component validation with issues."""
        github_service._state.is_initialized = False
        github_service._state.error_count = 8  # High error rate
        github_service._state.active_operations = 15  # Too many operations

        result = github_service.validate_component()

        assert result.is_valid is False
        assert len(result.validation_errors) == 3
        assert "not initialized" in result.validation_errors[0]
        assert "High error rate" in result.validation_errors[1]
        assert "Too many concurrent" in result.validation_errors[2]

    def test_get_debug_info_basic(self, github_service):
        """Test basic debug info retrieval."""
        debug_info = github_service.get_debug_info("INFO")

        # Check that it returns an object with the required protocol properties
        assert hasattr(debug_info, "debug_level")
        assert hasattr(debug_info, "debug_data")
        assert hasattr(debug_info, "stack_trace")
        assert hasattr(debug_info, "performance_metrics")

        assert debug_info.debug_level == "INFO"
        assert "service_state" in debug_info.debug_data
        assert "configuration" in debug_info.debug_data

    def test_get_debug_info_detailed(self, github_service):
        """Test detailed debug info retrieval."""
        # Add some operation history
        github_service._operation_history = [{"op": "test"}] * 5
        github_service._state.rate_limit_status = {"remaining": 100}

        debug_info = github_service.get_debug_info("DEBUG")

        assert "recent_operations" in debug_info.debug_data
        assert "rate_limit_status" in debug_info.debug_data
        assert debug_info.debug_data["rate_limit_status"]["remaining"] == 100

    def test_inspect_state_full(self, github_service):
        """Test full state inspection."""
        state_data = github_service.inspect_state()

        assert "service_id" in state_data
        assert "configuration" in state_data
        assert "metrics" in state_data
        assert state_data["operation_count"] == 10
        assert state_data["error_count"] == 2

    def test_inspect_state_path(self, github_service):
        """Test state inspection with specific path."""
        result = github_service.inspect_state("operation_count")
        assert result == {"operation_count": {"value": 10}}

        result = github_service.inspect_state("configuration.enable_webhooks")
        assert result == {"configuration.enable_webhooks": {"value": False}}

        result = github_service.inspect_state("nonexistent.path")
        assert "error" in result

    def test_get_component_dependencies(self, github_service):
        """Test component dependencies retrieval."""
        dependencies = github_service.get_component_dependencies()

        assert isinstance(dependencies, list)
        assert "GitHubConfig" in dependencies
        assert "ThreadPoolExecutor" in dependencies
        assert "GitHub API" in dependencies
        assert "Network connectivity" in dependencies

    def test_export_state_json(self, github_service):
        """Test state export as JSON."""
        json_str = github_service.export_state_json()

        assert isinstance(json_str, str)
        state_data = json.loads(json_str)
        assert "service_id" in state_data
        assert "configuration" in state_data
        assert "metrics" in state_data
        assert "operation_count" in state_data
        assert "error_count" in state_data

    def test_health_check(self, github_service):
        """Test health check functionality."""
        # Set service as initialized and running for health check
        github_service._state.is_initialized = True
        github_service._state.is_running = True

        health = github_service.health_check()

        assert isinstance(health, dict)
        assert "healthy" in health
        assert "initialized" in health
        assert "running" in health
        assert "operation_count" in health
        assert "error_count" in health
        assert "success_rate" in health
        assert "active_operations" in health

        assert (
            health["healthy"] is True
        )  # Success rate > 0.8 AND initialized AND running
        assert health["operation_count"] == 10
        assert health["error_count"] == 2
        assert health["success_rate"] == 0.8

    def test_health_check_unhealthy(self, github_service):
        """Test health check when service is unhealthy."""
        github_service._state.error_count = 9  # Low success rate

        health = github_service.health_check()
        assert health["healthy"] is False
        assert health["success_rate"] == 0.1


class TestGitHubServiceUtilities:
    """Test GitHub service utility functions."""

    @pytest.fixture
    def github_service(self):
        """Provide GitHub service for utility testing."""
        return GitHubService()

    def test_calculate_success_rate_no_operations(self, github_service):
        """Test success rate calculation with no operations."""
        rate = github_service._calculate_success_rate()
        assert rate == 1.0

    def test_calculate_success_rate_with_operations(self, github_service):
        """Test success rate calculation with operations."""
        github_service._state.operation_count = 10
        github_service._state.error_count = 3

        rate = github_service._calculate_success_rate()
        assert rate == 0.7  # 7 successes out of 10

    def test_calculate_success_rate_all_errors(self, github_service):
        """Test success rate calculation with all errors."""
        github_service._state.operation_count = 5
        github_service._state.error_count = 5

        rate = github_service._calculate_success_rate()
        assert rate == 0.0

    def test_calculate_success_rate_no_errors(self, github_service):
        """Test success rate calculation with no errors."""
        github_service._state.operation_count = 8
        github_service._state.error_count = 0

        rate = github_service._calculate_success_rate()
        assert rate == 1.0
