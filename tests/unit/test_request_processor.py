"""Tests for MCP request processing and routing engine implementation.

This module tests the comprehensive request processing pipeline,
method routing to appropriate handlers, parameter validation and context management.

Task 12.3 - Request Processing and Routing Engine Implementation
Focus: Request routing, parameter validation, context management, async operations
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_server_llm_cli_runner.core.errors import ValidationError
from mcp_server_llm_cli_runner.core.request_processor import (
    ContextManager,
    RequestProcessor,
)


# Mock classes to simulate the request processing components
class MockRequestProcessor:
    """Mock request processor for testing."""

    def __init__(self):
        self.handlers = {}
        self.validation_rules = {}

    def register_handler(self, method: str, handler):
        """Register a method handler."""
        self.handlers[method] = handler

    def add_validation_rule(self, method: str, rule):
        """Add validation rule for method."""
        if method not in self.validation_rules:
            self.validation_rules[method] = []
        self.validation_rules[method].append(rule)

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process incoming MCP request."""
        # Basic validation
        if not isinstance(request, dict):
            raise ValidationError("Request must be a dictionary")

        method = request.get("method")
        if not method:
            raise ValidationError("Request must have a method")

        # Get handler
        handler = self.handlers.get(method)
        if not handler:
            raise ValidationError(f"No handler for method: {method}")

        # Validate parameters
        params = request.get("params", {})
        if method in self.validation_rules:
            for rule in self.validation_rules[method]:
                await rule(params)

        # Call handler
        result = await handler(params)

        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": result,
        }

    async def route_to_handler(self, method: str, params: dict[str, Any]) -> Any:
        """Route request to appropriate handler."""
        handler = self.handlers.get(method)
        if not handler:
            raise ValidationError(f"No handler registered for method: {method}")

        return await handler(params)


class MockContextManager:
    """Mock context manager for request processing."""

    def __init__(self):
        self.request_context = {}
        self.cleanup_tasks = []

    @asynccontextmanager
    async def request_context(self, request_id: str):
        """Create request processing context."""
        self.request_context[request_id] = {
            "start_time": asyncio.get_event_loop().time(),
            "resources": [],
        }

        try:
            yield self.request_context[request_id]
        finally:
            # Cleanup resources
            for cleanup_task in self.cleanup_tasks:
                await cleanup_task()

            # Remove context
            self.request_context.pop(request_id, None)

    def add_cleanup_task(self, task):
        """Add cleanup task."""
        self.cleanup_tasks.append(task)


class TestRequestProcessor:
    """Test MCP request processing functionality."""

    @pytest.fixture
    def request_processor(self):
        """Create request processor for testing."""
        return RequestProcessor()

    @pytest.fixture
    def context_manager(self):
        """Create context manager for testing."""
        return ContextManager()

    @pytest.mark.asyncio
    async def test_handler_registration(self, request_processor):
        """Test method handler registration."""

        async def mock_handler(params):
            return {"status": "handled"}

        request_processor.register_handler("test_method", mock_handler)
        assert "test_method" in request_processor.handlers

    @pytest.mark.asyncio
    async def test_request_routing(self, request_processor):
        """Test request routing to correct handler."""

        # Register handler
        async def completion_handler(params):
            return {"choices": [{"message": {"content": "Hello"}}]}

        request_processor.register_handler("completion", completion_handler)

        # Test routing
        result = await request_processor.route_to_handler(
            "completion",
            {"prompt": "Hello"},
        )

        assert "choices" in result
        assert result["choices"][0]["message"]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_request_processing_pipeline(self, request_processor):
        """Test complete request processing pipeline."""

        # Register handler
        async def echo_handler(params):
            return params

        request_processor.register_handler("echo", echo_handler)

        # Process request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "echo",
            "params": {"data": "test"},
        }

        response = await request_processor.process_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["result"] == {"data": "test"}

    @pytest.mark.asyncio
    async def test_parameter_validation(self, request_processor):
        """Test parameter validation in request processing."""

        async def validate_required_params(params):
            if "required_field" not in params:
                raise ValidationError("required_field is required")

        # Register validation rule
        request_processor.add_validation_rule("test_method", validate_required_params)

        # Register handler
        async def test_handler(params):
            return {"validated": True}

        request_processor.register_handler("test_method", test_handler)

        # Test with missing required parameter
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "test_method",
            "params": {},
        }

        with pytest.raises(ValidationError, match="required_field is required"):
            await request_processor.process_request(request)

        # Test with required parameter
        request["params"]["required_field"] = "value"
        response = await request_processor.process_request(request)

        assert response["result"]["validated"] is True

    @pytest.mark.asyncio
    async def test_unknown_method_handling(self, request_processor):
        """Test handling of unknown methods."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown_method",
            "params": {},
        }

        with pytest.raises(ValidationError, match="No handler for method"):
            await request_processor.process_request(request)

    @pytest.mark.asyncio
    async def test_malformed_request_handling(self, request_processor):
        """Test handling of malformed requests."""
        # Test non-dict request
        with pytest.raises(ValidationError, match="Request must be a dictionary"):
            await request_processor.process_request("invalid")

        # Test request without method
        request = {"jsonrpc": "2.0", "id": 1}
        with pytest.raises(ValidationError, match="Request must have a method"):
            await request_processor.process_request(request)

    @pytest.mark.asyncio
    async def test_async_handler_execution(self, request_processor):
        """Test asynchronous handler execution."""

        async def slow_handler(params):
            await asyncio.sleep(0.01)  # Simulate async work
            return {"processed": True, "delay": 0.01}

        request_processor.register_handler("slow_method", slow_handler)

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "slow_method",
            "params": {},
        }

        start_time = asyncio.get_event_loop().time()
        response = await request_processor.process_request(request)
        end_time = asyncio.get_event_loop().time()

        # Should take at least the delay time
        assert end_time - start_time >= 0.01
        assert response["result"]["processed"] is True

    @pytest.mark.asyncio
    async def test_concurrent_request_processing(self, request_processor):
        """Test concurrent request processing."""

        async def concurrent_handler(params):
            request_id = params.get("id", "unknown")
            await asyncio.sleep(0.01)
            return {"id": request_id, "processed": True}

        request_processor.register_handler("concurrent", concurrent_handler)

        # Create multiple concurrent requests
        requests = [
            {
                "jsonrpc": "2.0",
                "id": i,
                "method": "concurrent",
                "params": {"id": i},
            }
            for i in range(5)
        ]

        # Process concurrently
        tasks = [request_processor.process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)

        # Verify all processed
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response["id"] == i
            assert response["result"]["id"] == i

    @pytest.mark.asyncio
    async def test_request_context_management(self, context_manager):
        """Test request context management."""
        request_id = "test-123"

        async with context_manager.request_context(request_id) as ctx:
            # Context should be available
            assert request_id in context_manager.request_context
            assert "start_time" in ctx
            assert "resources" in ctx

        # Context should be cleaned up
        assert request_id not in context_manager.request_context

    @pytest.mark.asyncio
    async def test_context_cleanup_on_error(self, context_manager):
        """Test context cleanup when error occurs."""
        request_id = "error-test"
        cleanup_called = False

        async def cleanup_task():
            nonlocal cleanup_called
            cleanup_called = True

        context_manager.add_cleanup_task(cleanup_task)

        try:
            async with context_manager.request_context(request_id):
                raise RuntimeError("Test error")
        except RuntimeError:
            pass

        # Cleanup should still be called
        assert cleanup_called
        assert request_id not in context_manager.request_context

    @pytest.mark.asyncio
    async def test_multiple_validation_rules(self, request_processor):
        """Test multiple validation rules for same method."""

        async def validate_type(params):
            if not isinstance(params.get("value"), str):
                raise ValidationError("value must be string")

        async def validate_length(params):
            if len(params.get("value", "")) < 3:
                raise ValidationError("value must be at least 3 characters")

        # Register multiple validation rules
        request_processor.add_validation_rule("validate_test", validate_type)
        request_processor.add_validation_rule("validate_test", validate_length)

        async def validation_handler(params):
            return {"valid": True}

        request_processor.register_handler("validate_test", validation_handler)

        # Test invalid type
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "validate_test",
            "params": {"value": 123},
        }

        with pytest.raises(ValidationError, match="value must be string"):
            await request_processor.process_request(request)

        # Test invalid length
        request["params"]["value"] = "hi"
        with pytest.raises(ValidationError, match="at least 3 characters"):
            await request_processor.process_request(request)

        # Test valid value
        request["params"]["value"] = "hello"
        response = await request_processor.process_request(request)
        assert response["result"]["valid"] is True

    @pytest.mark.asyncio
    async def test_handler_error_propagation(self, request_processor):
        """Test error propagation from handlers."""

        async def failing_handler(params):
            raise RuntimeError("Handler failed")

        request_processor.register_handler("failing", failing_handler)

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "failing",
            "params": {},
        }

        with pytest.raises(RuntimeError, match="Handler failed"):
            await request_processor.process_request(request)

    @pytest.mark.asyncio
    async def test_request_id_preservation(self, request_processor):
        """Test request ID is preserved in response."""

        async def id_handler(params):
            return {"processed": True}

        request_processor.register_handler("id_test", id_handler)

        # Test with various ID types
        test_ids = [1, "string-id", None, {"complex": "id"}]

        for test_id in test_ids:
            request = {
                "jsonrpc": "2.0",
                "id": test_id,
                "method": "id_test",
                "params": {},
            }

            response = await request_processor.process_request(request)
            assert response["id"] == test_id
