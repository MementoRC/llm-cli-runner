"""
Unit tests for GitService implementation.

This module provides comprehensive unit tests for the GitService class,
testing all public methods, error conditions, and integration points.

Critical for TDD Compliance:
    These tests define the interface that GitService must implement.
    DO NOT modify these tests to match implementation - the implementation
    must satisfy these test requirements to prevent LLM compliance issues.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import patch

import pytest

from mcp_server_git.operations.git_operations import (
    BranchResult,
    CommitResult,
    MergeResult,
)
from mcp_server_git.primitives.git_primitives import (
    GitValidationError,
)
from mcp_server_git.services.git_service import (
    GitOperationResult,
    GitService,
    GitServiceConfig,
    GitServiceState,
)


class TestGitServiceConfig:
    """Test GitServiceConfig dataclass."""

    def test_default_values(self):
        """Test GitServiceConfig default values."""
        config = GitServiceConfig()

        assert config.max_concurrent_operations == 10
        assert config.operation_timeout_seconds == 300
        assert config.enable_security_validation is True
        assert config.enable_performance_monitoring is True
        assert config.enable_state_history is True
        assert config.max_state_history_entries == 100
        assert config.default_remote == "origin"
        assert config.auto_push_after_commit is False
        assert config.gpg_signing_enabled is False
        assert config.gpg_key_id is None

    def test_custom_values(self):
        """Test GitServiceConfig with custom values."""
        config = GitServiceConfig(
            max_concurrent_operations=5,
            operation_timeout_seconds=600,
            enable_security_validation=False,
            default_remote="upstream",
            gpg_signing_enabled=True,
            gpg_key_id="TEST123",
        )

        assert config.max_concurrent_operations == 5
        assert config.operation_timeout_seconds == 600
        assert config.enable_security_validation is False
        assert config.default_remote == "upstream"
        assert config.gpg_signing_enabled is True
        assert config.gpg_key_id == "TEST123"


class TestGitOperationResult:
    """Test GitOperationResult dataclass."""

    def test_default_values(self):
        """Test GitOperationResult with required values."""
        result = GitOperationResult(
            success=True,
            operation_type="test_operation",
            repository_path="/test/repo",
        )

        assert result.success is True
        assert result.operation_type == "test_operation"
        assert result.repository_path == "/test/repo"
        assert result.result_data is None
        assert result.error_message is None
        assert result.duration_seconds == 0.0
        assert isinstance(result.timestamp, datetime)

    def test_custom_values(self):
        """Test GitOperationResult with custom values."""
        test_data = {"key": "value"}
        result = GitOperationResult(
            success=False,
            operation_type="commit",
            repository_path="/custom/repo",
            result_data=test_data,
            error_message="Test error",
            duration_seconds=2.5,
        )

        assert result.success is False
        assert result.operation_type == "commit"
        assert result.repository_path == "/custom/repo"
        assert result.result_data == test_data
        assert result.error_message == "Test error"
        assert result.duration_seconds == 2.5


class TestGitServiceState:
    """Test GitServiceState dataclass."""

    def test_required_fields(self):
        """Test GitServiceState with required fields."""
        start_time = datetime.now()
        state = GitServiceState(
            service_id="test_service_123",
            started_at=start_time,
        )

        assert state.service_id == "test_service_123"
        assert state.started_at == start_time
        assert state.operation_count == 0
        assert state.error_count == 0
        assert state.last_operation is None
        assert state.active_operations == 0
        assert isinstance(state.configuration, GitServiceConfig)
        assert isinstance(state.performance_metrics, dict)

    def test_custom_values(self):
        """Test GitServiceState with custom values."""
        start_time = datetime.now()
        config = GitServiceConfig(max_concurrent_operations=5)
        metrics = {"ops_per_sec": 10.5}

        state = GitServiceState(
            service_id="custom_service",
            started_at=start_time,
            operation_count=50,
            error_count=5,
            active_operations=3,
            configuration=config,
            performance_metrics=metrics,
        )

        assert state.operation_count == 50
        assert state.error_count == 5
        assert state.active_operations == 3
        assert state.configuration == config
        assert state.performance_metrics == metrics


class TestGitService:
    """Test GitService implementation."""

    def test_init_default_config(self):
        """Test GitService initialization with default config."""
        service = GitService()

        assert service._config is not None
        assert isinstance(service._config, GitServiceConfig)
        assert service._service_id.startswith("git_service_")
        assert isinstance(service._state, GitServiceState)
        assert service._state.service_id == service._service_id
        assert service._is_started is False

    def test_init_custom_config(self):
        """Test GitService initialization with custom config."""
        config = GitServiceConfig(max_concurrent_operations=5)
        service = GitService(config)

        assert service._config == config
        assert service._state.configuration == config

    @pytest.mark.asyncio
    async def test_start_service(self):
        """Test starting the GitService."""
        service = GitService()

        with patch.object(service, "_validate_configuration") as mock_validate:
            await service.start()

            assert service._is_started is True
            assert "operations_per_second" in service._state.performance_metrics
            assert "average_operation_duration" in service._state.performance_metrics
            assert "success_rate" in service._state.performance_metrics
            mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_service_already_started(self):
        """Test starting an already started service."""
        service = GitService()

        with patch.object(service, "_validate_configuration"):
            await service.start()
            # Start again - should not raise error
            await service.start()

            assert service._is_started is True

    @pytest.mark.asyncio
    async def test_stop_service(self):
        """Test stopping the GitService."""
        service = GitService()

        with patch.object(service, "_validate_configuration"):
            await service.start()
            await service.stop()

            assert service._is_started is False

    @pytest.mark.asyncio
    async def test_stop_service_not_started(self):
        """Test stopping a service that wasn't started."""
        service = GitService()

        # Should not raise error
        await service.stop()
        assert service._is_started is False

    @pytest.mark.asyncio
    async def test_commit_changes_success(self):
        """Test successful commit operation."""
        service = GitService()

        mock_commit_result = CommitResult(
            success=True,
            commit_hash="abc123",
            message="Test commit",
            files_committed=["test.py"],
        )

        with (
            patch.object(service, "_validate_configuration"),
            patch(
                "mcp_server_git.services.git_service.commit_changes_with_validation"
            ) as mock_commit,
        ):
            mock_commit.return_value = mock_commit_result
            await service.start()

            result = await service.commit_changes(
                repository_path="/test/repo",
                message="Test commit message",
                files=["test.py"],
            )

            assert result.success is True
            assert result.operation_type == "commit_changes"
            assert result.repository_path == "/test/repo"
            assert result.result_data["commit_hash"] == "abc123"
            assert result.result_data["files_committed"] == ["test.py"]

    @pytest.mark.asyncio
    async def test_commit_changes_failure(self):
        """Test failed commit operation."""
        service = GitService()

        mock_commit_result = CommitResult(
            success=False,
            error="Commit failed",
        )

        with (
            patch.object(service, "_validate_configuration"),
            patch(
                "mcp_server_git.services.git_service.commit_changes_with_validation"
            ) as mock_commit,
        ):
            mock_commit.return_value = mock_commit_result
            await service.start()

            result = await service.commit_changes(
                repository_path="/test/repo",
                message="Test commit message",
            )

            assert result.success is False
            assert result.error_message == "Commit failed"

    @pytest.mark.asyncio
    async def test_commit_changes_not_started(self):
        """Test commit operation when service not started."""
        service = GitService()

        result = await service.commit_changes(
            repository_path="/test/repo",
            message="Test commit message",
        )

        assert result.success is False
        assert "not started" in result.error_message

    @pytest.mark.asyncio
    async def test_create_branch_success(self):
        """Test successful branch creation."""
        service = GitService()

        mock_branch_result = BranchResult(
            success=True,
            branch_name="feature/test",
            message="Branch created",
            previous_branch="main",
        )

        with (
            patch.object(service, "_validate_configuration"),
            patch(
                "mcp_server_git.services.git_service.create_branch_with_checkout"
            ) as mock_branch,
        ):
            mock_branch.return_value = mock_branch_result
            await service.start()

            result = await service.create_branch(
                repository_path="/test/repo",
                branch_name="feature/test",
                base_branch="main",
            )

            assert result.success is True
            assert result.operation_type == "create_branch"
            assert result.result_data["branch_name"] == "feature/test"
            assert result.result_data["previous_branch"] == "main"

    @pytest.mark.asyncio
    async def test_merge_branches_success(self):
        """Test successful branch merge."""
        service = GitService()

        mock_merge_result = MergeResult(
            success=True,
            merge_commit_hash="def456",
            message="Merge successful",
        )

        with (
            patch.object(service, "_validate_configuration"),
            patch(
                "mcp_server_git.services.git_service.merge_branches_with_conflict_detection"
            ) as mock_merge,
        ):
            mock_merge.return_value = mock_merge_result
            await service.start()

            result = await service.merge_branches(
                repository_path="/test/repo",
                source_branch="feature/test",
                target_branch="main",
            )

            assert result.success is True
            assert result.operation_type == "merge_branches"
            assert result.result_data["merge_commit_hash"] == "def456"

    @pytest.mark.asyncio
    async def test_get_repository_status_success(self):
        """Test successful repository status retrieval."""
        service = GitService()

        mock_status = {"branch": "main", "modified": ["file1.py"]}

        with (
            patch.object(service, "_validate_configuration"),
            patch(
                "mcp_server_git.services.git_service.get_repository_status"
            ) as mock_status_func,
        ):
            mock_status_func.return_value = mock_status
            await service.start()

            result = await service.get_repository_status("/test/repo")

            assert result.success is True
            assert result.operation_type == "get_repository_status"
            assert result.result_data["status"] == mock_status

    @pytest.mark.asyncio
    async def test_concurrent_operations_limit(self):
        """Test concurrent operations limit enforcement."""
        config = GitServiceConfig(max_concurrent_operations=1)
        service = GitService(config)

        with (
            patch.object(service, "_validate_configuration"),
            patch(
                "mcp_server_git.services.git_service.get_repository_status"
            ) as mock_status,
        ):
            mock_status.return_value = {"branch": "main"}
            await service.start()

            # Start two operations concurrently
            task1 = asyncio.create_task(service.get_repository_status("/test/repo1"))
            task2 = asyncio.create_task(service.get_repository_status("/test/repo2"))

            # Wait for completion
            results = await asyncio.gather(task1, task2)

            assert len(results) == 2
            assert all(result.success for result in results)

    def test_get_component_state(self):
        """Test DebuggableComponent.get_component_state method."""
        service = GitService()

        state = service.get_component_state()

        assert state.component_id == service._service_id
        assert state.component_type == "GitService"
        assert isinstance(state.state_data, dict)
        assert isinstance(state.last_updated, datetime)

    def test_validate_component_valid(self):
        """Test component validation with valid configuration."""
        service = GitService()

        result = service.validate_component()

        assert result.is_valid is True
        assert len(result.validation_errors) == 0
        assert isinstance(result.validation_timestamp, datetime)

    def test_validate_component_invalid_config(self):
        """Test component validation with invalid configuration."""
        # Initialize with valid config, then manually set invalid values
        service = GitService()
        service._config.max_concurrent_operations = 0

        result = service.validate_component()

        assert result.is_valid is False
        assert len(result.validation_errors) > 0
        assert "max_concurrent_operations must be positive" in result.validation_errors

    def test_get_debug_info(self):
        """Test debug information retrieval."""
        service = GitService()

        debug_info = service.get_debug_info("DEBUG")

        assert debug_info.debug_level == "DEBUG"
        assert isinstance(debug_info.debug_data, dict)
        assert "service_id" in debug_info.debug_data
        assert "configuration" in debug_info.debug_data
        assert debug_info.stack_trace is None
        assert isinstance(debug_info.performance_metrics, dict)

    def test_inspect_state_full(self):
        """Test full state inspection."""
        service = GitService()

        state = service.inspect_state()

        assert isinstance(state, dict)
        assert "service_id" in state
        assert "started_at" in state
        assert "is_started" in state
        assert "configuration" in state

    def test_inspect_state_path(self):
        """Test state inspection with specific path."""
        service = GitService()

        state = service.inspect_state("service_id")

        assert "service_id" in state
        assert state["service_id"]["value"] == service._service_id

    def test_inspect_state_invalid_path(self):
        """Test state inspection with invalid path."""
        service = GitService()

        state = service.inspect_state("invalid.path.here")

        assert "error" in state
        assert "not found" in state["error"]

    def test_get_component_dependencies(self):
        """Test component dependencies listing."""
        service = GitService()

        dependencies = service.get_component_dependencies()

        assert isinstance(dependencies, list)
        assert "git_operations" in dependencies
        assert "git_primitives" in dependencies
        assert "thread_pool_executor" in dependencies
        assert "asyncio_event_loop" in dependencies

    def test_export_state_json(self):
        """Test state export as JSON."""
        service = GitService()

        json_state = service.export_state_json()

        assert isinstance(json_state, str)
        parsed_state = json.loads(json_state)
        assert isinstance(parsed_state, dict)
        assert "service_id" in parsed_state
        assert "started_at" in parsed_state

    def test_health_check_not_started(self):
        """Test health check for non-started service."""
        service = GitService()

        health = service.health_check()

        assert health["healthy"] is False
        assert health["status"] == "not_started"
        assert "uptime" in health
        assert "error_count" in health

    @pytest.mark.asyncio
    async def test_health_check_started(self):
        """Test health check for started service."""
        service = GitService()

        with patch.object(service, "_validate_configuration"):
            await service.start()

            health = service.health_check()

            assert health["healthy"] is True
            assert health["status"] == "healthy"
            assert health["uptime"] >= 0
            assert health["error_count"] == 0

    def test_validate_configuration_invalid_max_operations(self):
        """Test configuration validation with invalid max operations."""
        # Initialize with valid config, then manually set invalid values for validation test
        service = GitService()
        service._config.max_concurrent_operations = 0

        with pytest.raises(
            GitValidationError, match="max_concurrent_operations must be positive"
        ):
            service._validate_configuration()

    def test_validate_configuration_invalid_timeout(self):
        """Test configuration validation with invalid timeout."""
        service = GitService()
        service._config.operation_timeout_seconds = 0

        with pytest.raises(
            GitValidationError, match="operation_timeout_seconds must be positive"
        ):
            service._validate_configuration()

    def test_validate_configuration_gpg_warning(self):
        """Test configuration validation with GPG enabled but no key."""
        config = GitServiceConfig(gpg_signing_enabled=True, gpg_key_id=None)
        service = GitService(config)

        # Should not raise exception, but should log warning
        service._validate_configuration()  # No exception expected

    @pytest.mark.asyncio
    async def test_auto_push_after_commit(self):
        """Test auto-push functionality after commit."""
        config = GitServiceConfig(auto_push_after_commit=True)
        service = GitService(config)

        mock_commit_result = CommitResult(success=True, commit_hash="abc123")
        mock_push_result = {"success": True, "message": "Push successful"}

        with (
            patch.object(service, "_validate_configuration"),
            patch(
                "mcp_server_git.services.git_service.commit_changes_with_validation"
            ) as mock_commit,
            patch.object(service, "_push_changes") as mock_push,
        ):
            mock_commit.return_value = mock_commit_result
            mock_push.return_value = mock_push_result
            await service.start()

            result = await service.commit_changes(
                repository_path="/test/repo",
                message="Test commit",
            )

            assert result.success is True
            mock_push.assert_called_once_with("/test/repo")

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test exception handling in operations."""
        service = GitService()

        with (
            patch.object(service, "_validate_configuration"),
            patch(
                "mcp_server_git.services.git_service.commit_changes_with_validation"
            ) as mock_commit,
        ):
            mock_commit.side_effect = Exception("Unexpected error")
            await service.start()

            result = await service.commit_changes(
                repository_path="/test/repo",
                message="Test commit",
            )

            assert result.success is False
            assert "Unexpected error" in result.error_message

    @pytest.mark.asyncio
    async def test_operation_metrics_tracking(self):
        """Test that operation metrics are properly tracked."""
        service = GitService()

        mock_commit_result = CommitResult(success=True, commit_hash="abc123")

        with (
            patch.object(service, "_validate_configuration"),
            patch(
                "mcp_server_git.services.git_service.commit_changes_with_validation"
            ) as mock_commit,
        ):
            mock_commit.return_value = mock_commit_result
            await service.start()

            initial_count = service._state.operation_count

            await service.commit_changes("/test/repo", "Test commit")

            assert service._state.operation_count == initial_count + 1
            assert service._state.last_operation is not None
            assert service._state.last_operation.operation_type == "commit_changes"

    @pytest.mark.asyncio
    async def test_error_count_tracking(self):
        """Test that error counts are properly tracked."""
        service = GitService()

        mock_commit_result = CommitResult(success=False, error="Test error")

        with (
            patch.object(service, "_validate_configuration"),
            patch(
                "mcp_server_git.services.git_service.commit_changes_with_validation"
            ) as mock_commit,
        ):
            mock_commit.return_value = mock_commit_result
            await service.start()

            initial_error_count = service._state.error_count

            await service.commit_changes("/test/repo", "Test commit")

            assert service._state.error_count == initial_error_count + 1

    @pytest.mark.asyncio
    async def test_performance_metrics_update(self):
        """Test that performance metrics are updated correctly."""
        config = GitServiceConfig(enable_performance_monitoring=True)
        service = GitService(config)

        mock_commit_result = CommitResult(success=True, commit_hash="abc123")

        with (
            patch.object(service, "_validate_configuration"),
            patch(
                "mcp_server_git.services.git_service.commit_changes_with_validation"
            ) as mock_commit,
        ):
            mock_commit.return_value = mock_commit_result
            await service.start()

            await service.commit_changes("/test/repo", "Test commit")

            metrics = service._state.performance_metrics
            assert "operations_per_second" in metrics
            assert "average_operation_duration" in metrics
            assert "success_rate" in metrics
            assert metrics["success_rate"] == 100.0

    @pytest.mark.asyncio
    async def test_state_history_tracking(self):
        """Test that state history is tracked when enabled."""
        config = GitServiceConfig(
            enable_state_history=True, max_state_history_entries=5
        )
        service = GitService(config)

        with patch.object(service, "_validate_configuration"):
            await service.start()

            initial_history_length = len(service._state_history)
            assert initial_history_length >= 1  # Start creates initial snapshot

            # Simulate some operations to trigger state snapshots
            mock_result = GitOperationResult(
                success=True,
                operation_type="test",
                repository_path="/test",
            )
            await service._record_operation_result(mock_result)

            assert len(service._state_history) > initial_history_length
