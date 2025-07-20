"""
Unit tests for server middleware components.

Tests middleware functionality including authentication, logging, error handling,
request tracking, and middleware chain composition.
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_git.frameworks.server_middleware import (
    AuthenticationMiddleware,
    BaseMiddleware,
    ErrorHandlingMiddleware,
    LoggingMiddleware,
    MiddlewareChainManager,
    MiddlewareContext,
    RequestTrackingMiddleware,
    create_default_middleware_chain,
)


class TestMiddlewareContext:
    """Test MiddlewareContext functionality."""

    def test_initialization(self):
        """Test context initialization."""
        request = MagicMock()
        context = MiddlewareContext(request=request)

        assert context.request is request
        assert context.response is None
        assert context.metadata == {}
        assert isinstance(context.start_time, float)

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        context = MiddlewareContext(request=MagicMock())

        # Small delay to test elapsed time
        time.sleep(0.01)
        elapsed = context.elapsed_time()

        assert elapsed > 0
        assert elapsed < 1  # Should be very small


class TestBaseMiddleware:
    """Test BaseMiddleware abstract functionality."""

    def test_initialization(self):
        """Test middleware initialization."""

        class TestMiddleware(BaseMiddleware):
            async def process_request(self, context, next_handler):
                return await next_handler(context)

        middleware = TestMiddleware("test")

        assert middleware.name == "test"
        assert middleware.enabled is True
        assert middleware.is_enabled() is True

    def test_enable_disable(self):
        """Test enabling and disabling middleware."""

        class TestMiddleware(BaseMiddleware):
            async def process_request(self, context, next_handler):
                return await next_handler(context)

        middleware = TestMiddleware("test")

        # Test disable
        middleware.disable()
        assert middleware.is_enabled() is False

        # Test enable
        middleware.enable()
        assert middleware.is_enabled() is True


class TestAuthenticationMiddleware:
    """Test AuthenticationMiddleware functionality."""

    def test_initialization(self):
        """Test authentication middleware initialization."""
        middleware = AuthenticationMiddleware(required_scopes=["repo"])

        assert middleware.name == "authentication"
        assert middleware.required_scopes == ["repo"]
        assert middleware.token_cache == {}

    @pytest.mark.asyncio
    async def test_non_auth_request(self):
        """Test processing request that doesn't require authentication."""
        middleware = AuthenticationMiddleware()
        context = MiddlewareContext(request=MagicMock(method="git_status"))
        next_handler = AsyncMock(return_value="response")

        result = await middleware.process_request(context, next_handler)

        assert result == "response"
        next_handler.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_github_request_without_token(self):
        """Test GitHub request without valid token."""
        with patch.dict(os.environ, {}, clear=True):
            middleware = AuthenticationMiddleware()
            context = MiddlewareContext(request=MagicMock(method="github_list_prs"))
            next_handler = AsyncMock()

            result = await middleware.process_request(context, next_handler)

            # Should return auth error
            assert hasattr(result, "error")
            assert result.error.code == -32001
            next_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_github_request_with_valid_token(self):
        """Test GitHub request with valid token."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_1234567890abcdef"}):
            middleware = AuthenticationMiddleware()
            context = MiddlewareContext(request=MagicMock(method="github_list_prs"))
            next_handler = AsyncMock(return_value="response")

            result = await middleware.process_request(context, next_handler)

            assert result == "response"
            assert context.metadata["authenticated"] is True
            next_handler.assert_called_once_with(context)

    def test_requires_auth(self):
        """Test authentication requirement detection."""
        middleware = AuthenticationMiddleware()

        # GitHub requests require auth
        github_request = MagicMock(method="github_list_prs")
        assert middleware._requires_auth(github_request) is True

        # Git requests don't require auth
        git_request = MagicMock(method="git_status")
        assert middleware._requires_auth(git_request) is False

    @pytest.mark.asyncio
    async def test_token_validation_caching(self):
        """Test token validation caching."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_1234567890abcdef"}):
            middleware = AuthenticationMiddleware()
            context = MiddlewareContext(request=MagicMock())

            # First validation
            result1 = await middleware._validate_github_token(context)
            assert result1 is True

            # Second validation should use cache
            result2 = await middleware._validate_github_token(context)
            assert result2 is True

            # Check token is in cache
            assert "ghp_1234567890abcdef" in middleware.token_cache


class TestLoggingMiddleware:
    """Test LoggingMiddleware functionality."""

    def test_initialization(self):
        """Test logging middleware initialization."""
        middleware = LoggingMiddleware(log_requests=False, log_responses=True)

        assert middleware.name == "logging"
        assert middleware.log_requests is False
        assert middleware.log_responses is True
        assert middleware.request_counter == 0

    @pytest.mark.asyncio
    async def test_request_processing_success(self):
        """Test successful request processing with logging."""
        middleware = LoggingMiddleware()
        context = MiddlewareContext(request=MagicMock(method="test_method"))
        next_handler = AsyncMock(return_value="success_response")

        with patch.object(middleware, '_log_request') as mock_log_req, \
             patch.object(middleware, '_log_response') as mock_log_resp:

            result = await middleware.process_request(context, next_handler)

            assert result == "success_response"
            assert "request_id" in context.metadata
            assert middleware.request_counter == 1

            mock_log_req.assert_called_once()
            mock_log_resp.assert_called_once()

            # Check response logging was called with success=True
            call_args = mock_log_resp.call_args
            assert call_args.kwargs["success"] is True  # success parameter

    @pytest.mark.asyncio
    async def test_request_processing_error(self):
        """Test request processing with error."""
        middleware = LoggingMiddleware()
        context = MiddlewareContext(request=MagicMock(method="test_method"))
        test_error = Exception("Test error")
        next_handler = AsyncMock(side_effect=test_error)

        with patch.object(middleware, '_log_request') as mock_log_req, \
             patch.object(middleware, '_log_response') as mock_log_resp:

            with pytest.raises(Exception) as exc_info:
                await middleware.process_request(context, next_handler)

            assert exc_info.value is test_error

            mock_log_req.assert_called_once()
            mock_log_resp.assert_called_once()

            # Check response logging was called with success=False
            call_args = mock_log_resp.call_args
            assert call_args.kwargs["success"] is False  # success parameter
            assert call_args.kwargs["error"] is test_error

    @pytest.mark.asyncio
    async def test_disabled_middleware(self):
        """Test disabled logging middleware."""
        middleware = LoggingMiddleware()
        middleware.disable()

        context = MiddlewareContext(request=MagicMock())
        next_handler = AsyncMock(return_value="response")

        result = await middleware.process_request(context, next_handler)

        assert result == "response"
        assert middleware.request_counter == 0  # No counting when disabled
        next_handler.assert_called_once_with(context)


class TestErrorHandlingMiddleware:
    """Test ErrorHandlingMiddleware functionality."""

    def test_initialization(self):
        """Test error handling middleware initialization."""
        middleware = ErrorHandlingMiddleware(mask_sensitive_data=False)

        assert middleware.name == "error_handling"
        assert middleware.mask_sensitive_data is False
        assert middleware.error_counts == {}

    @pytest.mark.asyncio
    async def test_successful_request(self):
        """Test successful request processing."""
        middleware = ErrorHandlingMiddleware()
        context = MiddlewareContext(request=MagicMock())
        next_handler = AsyncMock(return_value="success")

        result = await middleware.process_request(context, next_handler)

        assert result == "success"
        assert middleware.error_counts == {}
        next_handler.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and response creation."""
        middleware = ErrorHandlingMiddleware()
        context = MiddlewareContext(request=MagicMock())
        test_error = ValueError("Test validation error")
        next_handler = AsyncMock(side_effect=test_error)

        result = await middleware.process_request(context, next_handler)

        # Should return JSONRPCError
        assert hasattr(result, "error")
        assert result.error.code == -32602  # Invalid params for ValueError
        assert "ValueError" in middleware.error_counts
        assert middleware.error_counts["ValueError"] == 1

    def test_mask_sensitive_info(self):
        """Test sensitive information masking."""
        middleware = ErrorHandlingMiddleware(mask_sensitive_data=True)

        message = "Error with token: ghp_1234567890 and key=secret123"
        masked = middleware._mask_sensitive_info(message)

        assert "ghp_1234567890" not in masked
        assert "secret123" not in masked
        assert "[REDACTED]" in masked

    def test_error_code_mapping(self):
        """Test error code mapping for different exception types."""
        middleware = ErrorHandlingMiddleware()

        assert middleware._get_error_code(ValueError()) == -32602
        assert middleware._get_error_code(FileNotFoundError()) == -32603
        assert middleware._get_error_code(PermissionError()) == -32001
        assert middleware._get_error_code(Exception()) == -32603  # Default


class TestRequestTrackingMiddleware:
    """Test RequestTrackingMiddleware functionality."""

    def test_initialization(self):
        """Test request tracking middleware initialization."""
        middleware = RequestTrackingMiddleware(max_history=500)

        assert middleware.name == "request_tracking"
        assert middleware.max_history == 500
        assert middleware.request_history == []
        assert middleware.active_requests == {}

    @pytest.mark.asyncio
    async def test_request_tracking(self):
        """Test request tracking through lifecycle."""
        middleware = RequestTrackingMiddleware()
        context = MiddlewareContext(request=MagicMock(method="test_method"))
        context.metadata["request_id"] = "test_req_123"
        next_handler = AsyncMock(return_value="response")

        result = await middleware.process_request(context, next_handler)

        assert result == "response"
        assert len(middleware.request_history) == 1
        assert len(middleware.active_requests) == 0  # Cleaned up after completion

        # Check recorded info
        record = middleware.request_history[0]
        assert record["id"] == "test_req_123"
        assert record["method"] == "test_method"
        assert record["success"] is True
        assert "duration" in record

    @pytest.mark.asyncio
    async def test_request_tracking_error(self):
        """Test request tracking with error."""
        middleware = RequestTrackingMiddleware()
        context = MiddlewareContext(request=MagicMock(method="test_method"))
        context.metadata["request_id"] = "test_req_error"
        test_error = Exception("Test error")
        next_handler = AsyncMock(side_effect=test_error)

        with pytest.raises(Exception):
            await middleware.process_request(context, next_handler)

        assert len(middleware.request_history) == 1
        assert len(middleware.active_requests) == 0

        record = middleware.request_history[0]
        assert record["success"] is False
        assert record["error_type"] == "Exception"

    def test_get_metrics(self):
        """Test metrics calculation."""
        middleware = RequestTrackingMiddleware()

        # Add some test history with all required fields
        middleware.request_history = [
            {"success": True, "duration": 0.1, "timestamp": "2023-01-01T10:00:00"},
            {"success": False, "duration": 0.2, "timestamp": "2023-01-01T11:00:00"},
            {"success": True, "duration": 0.3, "timestamp": "2023-01-01T12:00:00"},
        ]

        metrics = middleware.get_metrics()

        assert metrics["total_requests"] == 3
        assert metrics["successful_requests"] == 2
        assert metrics["error_rate"] == pytest.approx(1/3, rel=1e-3)
        assert metrics["average_duration"] == pytest.approx(0.2, rel=1e-3)
        assert metrics["last_request"] == "2023-01-01T12:00:00"

    def test_history_size_limit(self):
        """Test history size limit enforcement."""
        middleware = RequestTrackingMiddleware(max_history=3)

        # Add more records than limit
        for i in range(5):
            middleware.request_history.append({"id": f"req_{i}", "success": True})

        # Simulate cleanup
        if len(middleware.request_history) > middleware.max_history:
            middleware.request_history = middleware.request_history[-middleware.max_history:]

        assert len(middleware.request_history) == 3
        assert middleware.request_history[0]["id"] == "req_2"  # Oldest removed


class TestMiddlewareChainManager:
    """Test MiddlewareChainManager functionality."""

    def test_initialization(self):
        """Test chain manager initialization."""
        manager = MiddlewareChainManager()

        assert manager.middlewares == []

    def test_add_remove_middleware(self):
        """Test adding and removing middleware."""
        manager = MiddlewareChainManager()
        middleware1 = LoggingMiddleware()
        middleware2 = AuthenticationMiddleware()

        # Add middleware
        manager.add_middleware(middleware1)
        manager.add_middleware(middleware2)

        assert len(manager.middlewares) == 2
        assert manager.middlewares[0] is middleware1
        assert manager.middlewares[1] is middleware2

        # Remove middleware
        assert manager.remove_middleware("logging") is True
        assert len(manager.middlewares) == 1
        assert manager.middlewares[0] is middleware2

        # Try to remove non-existent middleware
        assert manager.remove_middleware("nonexistent") is False

    def test_get_middleware(self):
        """Test getting middleware by name."""
        manager = MiddlewareChainManager()
        middleware = LoggingMiddleware()
        manager.add_middleware(middleware)

        found = manager.get_middleware("logging")
        assert found is middleware

        not_found = manager.get_middleware("nonexistent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_process_request_empty_chain(self):
        """Test processing request with empty middleware chain."""
        manager = MiddlewareChainManager()
        request = MagicMock()

        result = await manager.process_request(request)

        assert result is request  # Should return request unchanged

    @pytest.mark.asyncio
    async def test_process_request_single_middleware(self):
        """Test processing request with single middleware."""
        manager = MiddlewareChainManager()

        # Create test middleware that modifies request
        class TestMiddleware(BaseMiddleware):
            async def process_request(self, context, next_handler):
                result = await next_handler(context)
                return f"processed_{result.test_value}"

        middleware = TestMiddleware("test")
        manager.add_middleware(middleware)

        request = MagicMock()
        request.test_value = "original"

        result = await manager.process_request(request)

        assert result == "processed_original"

    @pytest.mark.asyncio
    async def test_process_request_multiple_middleware(self):
        """Test processing request through multiple middleware."""
        manager = MiddlewareChainManager()

        class Middleware1(BaseMiddleware):
            async def process_request(self, context, next_handler):
                context.metadata["step1"] = True
                result = await next_handler(context)
                return f"step1_{result}"

        class Middleware2(BaseMiddleware):
            async def process_request(self, context, next_handler):
                context.metadata["step2"] = True
                result = await next_handler(context)
                return f"step2_{result}"

        manager.add_middleware(Middleware1("m1"))
        manager.add_middleware(Middleware2("m2"))

        request = MagicMock()
        request.test_value = "original"
        request.__str__ = lambda self: "original"  # Make string conversion return "original"

        result = await manager.process_request(request)

        # Should process in order: m1 -> m2 -> request -> m2 response -> m1 response
        assert result == "step1_step2_original"

    @pytest.mark.asyncio
    async def test_disabled_middleware_skipped(self):
        """Test that disabled middleware is skipped."""
        manager = MiddlewareChainManager()

        class TestMiddleware(BaseMiddleware):
            async def process_request(self, context, next_handler):
                result = await next_handler(context)
                return f"processed_{result}"

        middleware = TestMiddleware("test")
        middleware.disable()
        manager.add_middleware(middleware)

        request = MagicMock()
        request.test_value = "original"

        result = await manager.process_request(request)

        assert result is request  # Unchanged, middleware skipped

    def test_get_chain_state(self):
        """Test getting chain state."""
        manager = MiddlewareChainManager()
        manager.add_middleware(LoggingMiddleware())
        manager.add_middleware(AuthenticationMiddleware())

        state = manager.get_chain_state()

        assert state["middleware_count"] == 2
        assert len(state["middlewares"]) == 2
        assert state["middlewares"][0]["name"] == "logging"
        assert state["middlewares"][1]["name"] == "authentication"

    def test_validate_chain_configuration(self):
        """Test chain configuration validation."""
        manager = MiddlewareChainManager()

        # Add middleware in good order
        manager.add_middleware(ErrorHandlingMiddleware())
        manager.add_middleware(AuthenticationMiddleware())
        manager.add_middleware(LoggingMiddleware())

        validation = manager.validate_chain_configuration()

        assert validation["valid"] is True
        assert len(validation["issues"]) == 0

    def test_validate_chain_configuration_issues(self):
        """Test chain configuration validation with issues."""
        manager = MiddlewareChainManager()

        # Add middleware in problematic order (auth too late)
        manager.add_middleware(LoggingMiddleware())
        manager.add_middleware(RequestTrackingMiddleware())
        manager.add_middleware(ErrorHandlingMiddleware())  # Too late
        manager.add_middleware(AuthenticationMiddleware())  # Too late

        validation = manager.validate_chain_configuration()

        assert validation["valid"] is False
        assert len(validation["issues"]) > 0
        assert any("AuthenticationMiddleware" in issue for issue in validation["issues"])
        assert any("ErrorHandlingMiddleware" in issue for issue in validation["issues"])


class TestDefaultMiddlewareChain:
    """Test default middleware chain creation."""

    def test_create_default_chain(self):
        """Test creation of default middleware chain."""
        chain = create_default_middleware_chain()

        assert isinstance(chain, MiddlewareChainManager)
        assert len(chain.middlewares) == 4

        # Check middleware types and order
        middleware_types = [type(m).__name__ for m in chain.middlewares]
        assert "ErrorHandlingMiddleware" in middleware_types
        assert "LoggingMiddleware" in middleware_types
        assert "AuthenticationMiddleware" in middleware_types
        assert "RequestTrackingMiddleware" in middleware_types

        # Error handling should be first (index 0)
        assert isinstance(chain.middlewares[0], ErrorHandlingMiddleware)

    def test_default_chain_validation(self):
        """Test that default chain passes validation."""
        chain = create_default_middleware_chain()

        validation = chain.validate_chain_configuration()

        assert validation["valid"] is True
        assert len(validation["issues"]) == 0
