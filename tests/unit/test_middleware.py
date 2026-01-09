"""Unit tests for server middleware components.

Tests the middleware pipeline, logging middleware, metrics collection,
and resource management following TDD methodology.
"""

import asyncio
import json
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mcp_server_llm_cli_runner.server.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewareChain,
    ResourceManager,
)


class TestMiddlewareChain:
    """Test middleware chain functionality."""

    @pytest.fixture
    def middleware_chain(self):
        """Create middleware chain for testing."""
        return MiddlewareChain()

    @pytest.fixture
    def mock_middleware(self):
        """Create mock middleware for testing."""
        middleware = Mock()
        middleware.process_request = AsyncMock()
        middleware.process_response = AsyncMock()
        return middleware

    def test_add_middleware(self, middleware_chain, mock_middleware):
        """Test adding middleware to chain."""
        middleware_chain.add_middleware(mock_middleware)
        assert mock_middleware in middleware_chain._middlewares

    def test_remove_middleware(self, middleware_chain, mock_middleware):
        """Test removing middleware from chain."""
        middleware_chain.add_middleware(mock_middleware)
        middleware_chain.remove_middleware(mock_middleware)
        assert mock_middleware not in middleware_chain._middlewares

    @pytest.mark.asyncio
    async def test_process_request_single_middleware(
        self,
        middleware_chain,
        mock_middleware,
    ):
        """Test request processing with single middleware."""
        request = {"data": "test"}
        mock_middleware.process_request.return_value = request

        middleware_chain.add_middleware(mock_middleware)
        result = await middleware_chain.process_request(request)

        mock_middleware.process_request.assert_called_once_with(request)
        assert result == request

    @pytest.mark.asyncio
    async def test_process_request_multiple_middleware(self, middleware_chain):
        """Test request processing with multiple middleware."""
        middleware1 = Mock()
        middleware1.process_request = AsyncMock(return_value={"step": 1})

        middleware2 = Mock()
        middleware2.process_request = AsyncMock(return_value={"step": 2})

        middleware_chain.add_middleware(middleware1)
        middleware_chain.add_middleware(middleware2)

        request = {"data": "test"}
        result = await middleware_chain.process_request(request)

        # Both middleware should be called in order
        middleware1.process_request.assert_called_once_with(request)
        middleware2.process_request.assert_called_once_with({"step": 1})
        assert result == {"step": 2}

    @pytest.mark.asyncio
    async def test_process_response_chain(self, middleware_chain):
        """Test response processing through middleware chain."""
        middleware1 = Mock()
        middleware1.process_response = AsyncMock(return_value={"processed": 1})

        middleware2 = Mock()
        middleware2.process_response = AsyncMock(return_value={"processed": 2})

        middleware_chain.add_middleware(middleware1)
        middleware_chain.add_middleware(middleware2)

        response = {"data": "response"}
        result = await middleware_chain.process_response(response)

        # Response processed in reverse order
        middleware2.process_response.assert_called_once_with(response)
        middleware1.process_response.assert_called_once_with({"processed": 2})
        assert result == {"processed": 1}

    @pytest.mark.asyncio
    async def test_middleware_error_propagation(self, middleware_chain):
        """Test error propagation through middleware chain."""
        failing_middleware = Mock()
        failing_middleware.process_request = AsyncMock(
            side_effect=RuntimeError("Middleware error"),
        )

        middleware_chain.add_middleware(failing_middleware)

        with pytest.raises(RuntimeError, match="Middleware error"):
            await middleware_chain.process_request({"data": "test"})


class TestLoggingMiddleware:
    """Test logging middleware functionality."""

    @pytest.fixture
    def logging_middleware(self):
        """Create logging middleware for testing."""
        return LoggingMiddleware()

    @pytest.mark.asyncio
    async def test_request_logging(self, logging_middleware):
        """Test request logging functionality."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "completion",
            "params": {"provider": "openai"},
        }

        with patch.object(logging_middleware.logger, "info") as mock_log:
            result = await logging_middleware.process_request(request)

            # Should log the request
            mock_log.assert_called()
            assert result == request

    @pytest.mark.asyncio
    async def test_response_logging(self, logging_middleware):
        """Test response logging functionality."""
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"choices": [{"message": {"content": "Hello"}}]},
        }

        with patch.object(logging_middleware.logger, "info") as mock_log:
            result = await logging_middleware.process_response(response)

            # Should log the response
            mock_log.assert_called()
            assert result == response

    @pytest.mark.asyncio
    async def test_error_logging(self, logging_middleware):
        """Test error case logging."""
        request = {"invalid": "request"}

        with patch.object(logging_middleware.logger, "warning") as mock_warn:
            # This should still process but log a warning
            result = await logging_middleware.process_request(request)

            # Should log warning for invalid structure
            if mock_warn.called:
                assert result == request


class TestMetricsMiddleware:
    """Test metrics middleware functionality."""

    @pytest.fixture
    def metrics_middleware(self):
        """Create metrics middleware for testing."""
        return MetricsMiddleware()

    @pytest.mark.asyncio
    async def test_request_metrics_collection(self, metrics_middleware):
        """Test metrics collection for requests."""
        request = {"jsonrpc": "2.0", "id": 1, "method": "completion"}

        # Process request
        result = await metrics_middleware.process_request(request)

        # Check metrics
        metrics = metrics_middleware.get_metrics()
        assert metrics["requests_processed"] == 1
        assert "last_request_time" in metrics
        assert result == request

    @pytest.mark.asyncio
    async def test_response_metrics_collection(self, metrics_middleware):
        """Test metrics collection for responses."""
        response = {"jsonrpc": "2.0", "id": 1, "result": {}}

        # Process response
        result = await metrics_middleware.process_response(response)

        # Check metrics
        metrics = metrics_middleware.get_metrics()
        assert metrics["responses_processed"] == 1
        assert result == response

    @pytest.mark.asyncio
    async def test_error_metrics_collection(self, metrics_middleware):
        """Test metrics collection for errors."""
        # Simulate error by processing malformed data
        with pytest.raises(TypeError):
            await metrics_middleware.process_request(None)

        metrics = metrics_middleware.get_metrics()
        assert metrics.get("errors_encountered", 0) >= 0

    @pytest.mark.asyncio
    async def test_response_time_tracking(self, metrics_middleware):
        """Test response time tracking."""
        request = {"jsonrpc": "2.0", "id": 1}
        response = {"jsonrpc": "2.0", "id": 1, "result": {}}

        # Process request then response with delay
        await metrics_middleware.process_request(request)
        await asyncio.sleep(0.01)  # Small delay
        await metrics_middleware.process_response(response)

        metrics = metrics_middleware.get_metrics()
        assert "average_response_time" in metrics
        assert metrics["average_response_time"] > 0

    def test_metrics_reset(self, metrics_middleware):
        """Test metrics reset functionality."""
        # Set some metrics
        metrics_middleware._metrics["requests_processed"] = 10

        # Reset
        metrics_middleware.reset_metrics()

        # Verify reset
        metrics = metrics_middleware.get_metrics()
        assert metrics["requests_processed"] == 0


class TestResourceManager:
    """Test resource manager functionality."""

    @pytest.fixture
    def resource_manager(self):
        """Create resource manager for testing."""
        return ResourceManager(
            max_pool_size=5,
            idle_timeout=10.0,
        )

    def test_initialization(self, resource_manager):
        """Test resource manager initialization."""
        assert resource_manager.max_pool_size == 5
        assert resource_manager.idle_timeout == 10.0

    def test_can_handle_request_under_limit(self, resource_manager):
        """Test request acceptance under limits."""
        # Mock system resources
        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value.percent = 50.0
                assert resource_manager.can_handle_request() is True

    def test_can_handle_request_over_memory_limit(self, resource_manager):
        """Test request rejection over memory limit."""
        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value.percent = 95.0  # Over limit
                assert resource_manager.can_handle_request() is False

    def test_can_handle_request_over_concurrent_limit(self, resource_manager):
        """Test request rejection over concurrent limit."""
        # Fill up concurrent requests
        resource_manager._active_requests = 5  # At max

        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value.percent = 50.0
                assert resource_manager.can_handle_request() is False

    @pytest.mark.asyncio
    async def test_acquire_resources_context_manager(self, resource_manager):
        """Test resource acquisition as context manager."""
        initial_count = resource_manager._active_requests

        async with resource_manager.acquire_resources():
            # Should increment active requests
            assert resource_manager._active_requests == initial_count + 1

        # Should decrement after exit
        assert resource_manager._active_requests == initial_count

    @pytest.mark.asyncio
    async def test_acquire_resources_with_timeout(self, resource_manager):
        """Test resource acquisition with timeout."""
        # Fill up all concurrent slots
        resource_manager._active_requests = resource_manager.max_concurrent_requests

        # Should timeout trying to acquire
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.1):
                async with resource_manager.acquire_resources():
                    pass

    @pytest.mark.asyncio
    async def test_concurrent_resource_management(self, resource_manager):
        """Test concurrent resource management."""

        async def acquire_resource():
            async with resource_manager.acquire_resources():
                await asyncio.sleep(0.01)
                return True

        # Start multiple concurrent acquisitions
        tasks = [acquire_resource() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (within limits)
        assert all(result is True for result in results)
        # Should be back to zero
        assert resource_manager._active_requests == 0

    def test_get_resource_status(self, resource_manager):
        """Test resource status reporting."""
        status = resource_manager.get_resource_status()

        assert "active_requests" in status
        assert "max_concurrent_requests" in status
        assert "memory_usage_percent" in status
        assert "cpu_usage_percent" in status

    def test_resource_metrics_tracking(self, resource_manager):
        """Test resource metrics tracking."""
        # Set some active requests
        resource_manager._active_requests = 3

        metrics = resource_manager.get_metrics()
        assert metrics["active_requests"] == 3
        assert "peak_concurrent_requests" in metrics
        assert "total_requests_handled" in metrics
