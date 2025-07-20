"""
Unit tests for server session service.

Tests the SessionService class functionality including lifecycle management,
session validation, client capability checking, and debugging capabilities.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_git.services.server_session import (
    SessionService,
    SessionServiceDebugInfo,
    SessionServiceState,
    SessionServiceValidationResult,
)


class TestSessionService:
    """Test SessionService functionality."""

    def test_initialization(self):
        """Test service initialization."""
        service = SessionService(
            idle_timeout=600.0,
            heartbeat_timeout=30.0,
            service_id="test_service"
        )

        assert service.service_id == "test_service"
        assert not service.is_running
        assert service.start_time is None
        assert service.error_count == 0
        assert service.operation_count == 0
        assert service.session_manager is not None

    @pytest.mark.asyncio
    async def test_start_service(self):
        """Test service startup."""
        service = SessionService(service_id="test_service")

        with patch.object(service.session_manager, 'restore_sessions', new_callable=AsyncMock):
            await service.start()

        assert service.is_running
        assert service.start_time is not None
        assert isinstance(service.start_time, datetime)

    @pytest.mark.asyncio
    async def test_start_service_already_running(self):
        """Test starting service when already running."""
        service = SessionService(service_id="test_service")
        service.is_running = True

        with patch.object(service.session_manager, 'restore_sessions', new_callable=AsyncMock):
            await service.start()

        # Should not change start_time if already running
        assert service.is_running

    @pytest.mark.asyncio
    async def test_stop_service(self):
        """Test service shutdown."""
        service = SessionService(service_id="test_service")
        service.is_running = True

        with patch.object(service.session_manager, 'shutdown', new_callable=AsyncMock):
            await service.stop()

        assert not service.is_running

    @pytest.mark.asyncio
    async def test_stop_service_not_running(self):
        """Test stopping service when not running."""
        service = SessionService(service_id="test_service")
        assert not service.is_running

        with patch.object(service.session_manager, 'shutdown', new_callable=AsyncMock):
            await service.stop()

        assert not service.is_running

    @pytest.mark.asyncio
    async def test_create_session_success(self):
        """Test successful session creation."""
        service = SessionService(service_id="test_service")
        service.is_running = True

        mock_session = MagicMock()
        with patch.object(
            service.session_manager, 'create_session', new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_session

            result = await service.create_session(
                "test_session", user="test_user", repository=Path("/test/repo")
            )

        assert result is mock_session
        assert service.operation_count == 1
        mock_create.assert_called_once_with("test_session", "test_user", Path("/test/repo"))

    @pytest.mark.asyncio
    async def test_create_session_service_not_running(self):
        """Test session creation when service not running."""
        service = SessionService(service_id="test_service")
        assert not service.is_running

        with pytest.raises(RuntimeError, match="SessionService is not running"):
            await service.create_session("test_session")

    @pytest.mark.asyncio
    async def test_create_session_failure(self):
        """Test session creation failure handling."""
        service = SessionService(service_id="test_service")
        service.is_running = True

        with patch.object(
            service.session_manager, 'create_session', new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = Exception("Creation failed")

            with pytest.raises(Exception, match="Creation failed"):
                await service.create_session("test_session")

        assert service.error_count == 1
        assert service.last_error == "Creation failed"

    @pytest.mark.asyncio
    async def test_get_session(self):
        """Test getting existing session."""
        service = SessionService(service_id="test_service")
        service.is_running = True

        mock_session = MagicMock()
        with patch.object(
            service.session_manager, 'get_session', new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_session

            result = await service.get_session("test_session")

        assert result is mock_session
        mock_get.assert_called_once_with("test_session")

    @pytest.mark.asyncio
    async def test_get_session_service_not_running(self):
        """Test getting session when service not running."""
        service = SessionService(service_id="test_service")
        assert not service.is_running

        with pytest.raises(RuntimeError, match="SessionService is not running"):
            await service.get_session("test_session")

    @pytest.mark.asyncio
    async def test_close_session_success(self):
        """Test successful session closure."""
        service = SessionService(service_id="test_service")
        service.is_running = True

        with patch.object(
            service.session_manager, 'close_session', new_callable=AsyncMock
        ) as mock_close:
            await service.close_session("test_session")

        mock_close.assert_called_once_with("test_session")

    @pytest.mark.asyncio
    async def test_close_session_failure(self):
        """Test session closure failure handling."""
        service = SessionService(service_id="test_service")
        service.is_running = True

        with patch.object(
            service.session_manager, 'close_session', new_callable=AsyncMock
        ) as mock_close:
            mock_close.side_effect = Exception("Close failed")

            with pytest.raises(Exception, match="Close failed"):
                await service.close_session("test_session")

        assert service.error_count == 1
        assert service.last_error == "Close failed"

    @pytest.mark.asyncio
    async def test_validate_server_session_valid(self):
        """Test server session validation with valid session."""
        service = SessionService(service_id="test_service")

        from mcp.server.session import ServerSession
        mock_session = MagicMock(spec=ServerSession)

        result = await service.validate_server_session(mock_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_server_session_invalid(self):
        """Test server session validation with invalid session."""
        service = SessionService(service_id="test_service")

        # Test with None
        result = await service.validate_server_session(None)
        assert result is False

        # Test with wrong type
        result = await service.validate_server_session("not_a_session")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_client_capability_roots(self):
        """Test client capability checking for roots."""
        service = SessionService(service_id="test_service")

        mock_session = MagicMock()
        mock_session.check_client_capability.return_value = True

        result = await service.check_client_capability(mock_session, "roots")
        assert result is True

        mock_session.check_client_capability.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_client_capability_unknown(self):
        """Test client capability checking for unknown capability."""
        service = SessionService(service_id="test_service")

        mock_session = MagicMock()

        result = await service.check_client_capability(mock_session, "unknown")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_client_capability_error(self):
        """Test client capability checking with error."""
        service = SessionService(service_id="test_service")

        mock_session = MagicMock()
        mock_session.check_client_capability.side_effect = Exception("Check failed")

        result = await service.check_client_capability(mock_session, "roots")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_repository_roots_success(self):
        """Test successful repository roots listing."""
        service = SessionService(service_id="test_service")

        # Mock ServerSession with proper spec
        from mcp.server.session import ServerSession
        mock_session = MagicMock(spec=ServerSession)
        mock_session.check_client_capability.return_value = True

        # Mock roots result
        mock_root = MagicMock()
        mock_root.uri.path = "/test/repo"

        mock_roots_result = MagicMock()
        mock_roots_result.roots = [mock_root]
        mock_session.list_roots.return_value = mock_roots_result

        # Mock the git module import and Repo class
        mock_git_module = MagicMock()
        mock_repo_class = MagicMock()
        mock_git_module.Repo = mock_repo_class
        mock_repo_class.return_value = MagicMock()  # Valid repo

        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'git':
                    return mock_git_module
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = await service.list_repository_roots(mock_session)

        assert result == ["/test/repo"]

    @pytest.mark.asyncio
    async def test_list_repository_roots_invalid_session(self):
        """Test repository roots listing with invalid session."""
        service = SessionService(service_id="test_service")

        result = await service.list_repository_roots(None)
        assert result == []

    @pytest.mark.asyncio
    async def test_list_repository_roots_no_capability(self):
        """Test repository roots listing without client capability."""
        service = SessionService(service_id="test_service")

        mock_session = MagicMock()
        mock_session.check_client_capability.return_value = False

        result = await service.list_repository_roots(mock_session)
        assert result == []

    @pytest.mark.asyncio
    async def test_list_repository_roots_invalid_git_repo(self):
        """Test repository roots listing with invalid git repository."""
        service = SessionService(service_id="test_service")

        mock_session = MagicMock()
        mock_session.check_client_capability.return_value = True

        mock_root = MagicMock()
        mock_root.uri.path = "/not/a/repo"

        mock_roots_result = MagicMock()
        mock_roots_result.roots = [mock_root]
        mock_session.list_roots.return_value = mock_roots_result

        # Mock the git module import with exception raising
        class MockInvalidGitRepositoryError(Exception):
            pass

        mock_git_module = MagicMock()
        mock_repo_class = MagicMock()
        mock_git_module.Repo = mock_repo_class
        mock_git_module.InvalidGitRepositoryError = MockInvalidGitRepositoryError
        mock_repo_class.side_effect = MockInvalidGitRepositoryError("Not a git repo")

        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'git':
                    return mock_git_module
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = await service.list_repository_roots(mock_session)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_session_metrics(self):
        """Test getting session metrics."""
        service = SessionService(service_id="test_service")
        service.is_running = True
        service.start_time = datetime.now()
        service.operation_count = 5
        service.error_count = 1

        mock_metrics = {"session1": {"status": "active"}}
        with patch.object(
            service.session_manager, 'get_metrics', new_callable=AsyncMock
        ) as mock_get_metrics:
            mock_get_metrics.return_value = mock_metrics

            result = await service.get_session_metrics()

        assert result["service_id"] == "test_service"
        assert result["service_running"] is True
        assert result["operation_count"] == 5
        assert result["error_count"] == 1
        assert result["session_count"] == 1
        assert result["session_metrics"] == mock_metrics

    @pytest.mark.asyncio
    async def test_cleanup_idle_sessions(self):
        """Test idle session cleanup."""
        service = SessionService(service_id="test_service")
        service.is_running = True

        with patch.object(
            service.session_manager, 'cleanup_idle_sessions', new_callable=AsyncMock
        ) as mock_cleanup:
            await service.cleanup_idle_sessions()

        mock_cleanup.assert_called_once()

    def test_get_component_state(self):
        """Test getting component state."""
        service = SessionService(service_id="test_service")
        service.is_running = True
        service.start_time = datetime.now()
        service.operation_count = 10
        service.error_count = 2

        state = service.get_component_state()

        assert isinstance(state, SessionServiceState)
        assert state.component_id == "test_service"
        assert state.component_type == "SessionService"
        assert state.state_data["is_running"] is True
        assert state.state_data["operation_count"] == 10
        assert state.state_data["error_count"] == 2

    def test_validate_component_valid(self):
        """Test component validation with valid state."""
        service = SessionService(service_id="test_service")

        result = service.validate_component()

        assert isinstance(result, SessionServiceValidationResult)
        assert result.is_valid is True
        assert len(result.validation_errors) == 0

    def test_validate_component_invalid(self):
        """Test component validation with invalid state."""
        service = SessionService(service_id="test_service")
        service.session_manager = None  # Invalid state

        result = service.validate_component()

        assert result.is_valid is False
        assert "SessionManager is not initialized" in result.validation_errors

    def test_validate_component_warnings(self):
        """Test component validation with warnings."""
        service = SessionService(service_id="test_service")
        service.is_running = True
        service.start_time = None  # Warning condition
        service.error_count = 15  # High error count

        result = service.validate_component()

        assert result.is_valid is True  # No errors, just warnings
        assert len(result.validation_warnings) == 2
        assert any("start_time is not set" in w for w in result.validation_warnings)
        assert any("High error count" in w for w in result.validation_warnings)

    def test_get_debug_info(self):
        """Test getting debug information."""
        service = SessionService(service_id="test_service")
        service.is_running = True
        service.start_time = datetime.now()
        service.operation_count = 20
        service.error_count = 3

        debug_info = service.get_debug_info()

        assert isinstance(debug_info, SessionServiceDebugInfo)
        assert debug_info.debug_level == "detailed"
        assert debug_info.debug_data["service_id"] == "test_service"
        assert debug_info.debug_data["operation_count"] == 20
        assert debug_info.debug_data["error_count"] == 3
        assert "uptime_seconds" in debug_info.performance_metrics
        assert "error_rate" in debug_info.performance_metrics

    def test_inspect_state_full(self):
        """Test full state inspection."""
        service = SessionService(service_id="test_service")

        result = service.inspect_state()

        assert "component_state" in result
        assert "validation_result" in result
        assert "debug_info" in result

    def test_inspect_state_specific_paths(self):
        """Test specific path state inspection."""
        service = SessionService(service_id="test_service")

        # Test state path
        state_result = service.inspect_state("state")
        assert isinstance(state_result, dict)
        assert "service_id" in state_result

        # Test validation path
        validation_result = service.inspect_state("validation")
        assert "is_valid" in validation_result
        assert "errors" in validation_result
        assert "warnings" in validation_result

        # Test debug path
        debug_result = service.inspect_state("debug")
        assert isinstance(debug_result, dict)

        # Test metrics path
        metrics_result = service.inspect_state("metrics")
        assert isinstance(metrics_result, dict)

        # Test unknown path
        unknown_result = service.inspect_state("unknown")
        assert "error" in unknown_result

    def test_get_component_dependencies(self):
        """Test getting component dependencies."""
        service = SessionService(service_id="test_service")

        dependencies = service.get_component_dependencies()

        assert isinstance(dependencies, list)
        assert len(dependencies) > 0
        assert "session.SessionManager" in dependencies
        assert "protocols.debugging_protocol.DebuggableComponent" in dependencies

    def test_export_state_json(self):
        """Test exporting state as JSON."""
        service = SessionService(service_id="test_service")

        json_str = service.export_state_json()

        assert isinstance(json_str, str)

        import json
        state_data = json.loads(json_str)

        assert "component_state" in state_data
        assert "validation_result" in state_data
        assert "debug_info" in state_data

    def test_health_check_healthy(self):
        """Test health check with healthy service."""
        service = SessionService(service_id="test_service")
        service.is_running = True
        service.start_time = datetime.now()

        health = service.health_check()

        assert health["service_id"] == "test_service"
        assert health["status"] == "healthy"
        assert health["is_running"] is True
        assert health["error_rate"] == 0

    def test_health_check_degraded(self):
        """Test health check with warnings."""
        service = SessionService(service_id="test_service")
        service.is_running = True
        service.start_time = None  # This causes a warning
        service.error_count = 15  # High error count

        health = service.health_check()

        assert health["status"] == "degraded"
        assert len(health["validation_warnings"]) > 0

    def test_health_check_unhealthy(self):
        """Test health check with errors."""
        service = SessionService(service_id="test_service")
        service.session_manager = None  # This causes an error

        health = service.health_check()

        assert health["status"] == "unhealthy"
        assert len(health["validation_errors"]) > 0


class TestSessionServiceState:
    """Test SessionServiceState dataclass."""

    def test_initialization(self):
        """Test state initialization."""
        state = SessionServiceState(
            component_id="test_id",
            component_type="SessionService",
            state_data={"key": "value"}
        )

        assert state.component_id == "test_id"
        assert state.component_type == "SessionService"
        assert state.state_data == {"key": "value"}
        assert isinstance(state.last_updated, datetime)


class TestSessionServiceValidationResult:
    """Test SessionServiceValidationResult dataclass."""

    def test_initialization(self):
        """Test validation result initialization."""
        result = SessionServiceValidationResult(
            is_valid=True,
            validation_errors=["error1"],
            validation_warnings=["warning1"]
        )

        assert result.is_valid is True
        assert result.validation_errors == ["error1"]
        assert result.validation_warnings == ["warning1"]
        assert isinstance(result.validation_timestamp, datetime)


class TestSessionServiceDebugInfo:
    """Test SessionServiceDebugInfo dataclass."""

    def test_initialization(self):
        """Test debug info initialization."""
        debug_info = SessionServiceDebugInfo(
            debug_level="detailed",
            debug_data={"key": "value"},
            performance_metrics={"metric": 1.0}
        )

        assert debug_info.debug_level == "detailed"
        assert debug_info.debug_data == {"key": "value"}
        assert debug_info.performance_metrics == {"metric": 1.0}
        assert debug_info.stack_trace is None
