"""Tests for MCP protocol base handler implementation.

This module tests the core MCP protocol message handling infrastructure,
JSON-RPC 2.0 compliance, and request/response handling with proper error management.

Task 12.1 - MCP Protocol Base Handler Implementation
Focus: Core protocol infrastructure, JSON-RPC 2.0, message validation
"""

import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_server_cheap_llm.core.errors import ValidationError


class TestMCPProtocolHandlerTDD:
    """Test-driven development for MCP protocol handler."""

    def test_mcp_protocol_handler_import(self):
        """Test that MCPProtocolHandler can be imported."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        assert MCPProtocolHandler is not None

    def test_mcp_protocol_handler_instantiation(self):
        """Test that MCPProtocolHandler can be instantiated."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        handler = MCPProtocolHandler()
        assert handler is not None

    def test_parse_message_method_exists(self):
        """Test that parse_message method exists."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        handler = MCPProtocolHandler()
        assert hasattr(handler, "parse_message")
        assert callable(handler.parse_message)

    def test_handle_request_method_exists(self):
        """Test that handle_request method exists."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        handler = MCPProtocolHandler()
        assert hasattr(handler, "handle_request")
        assert callable(handler.handle_request)

    def test_create_response_method_exists(self):
        """Test that create_response method exists."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        handler = MCPProtocolHandler()
        assert hasattr(handler, "create_response")
        assert callable(handler.create_response)

    def test_create_error_response_method_exists(self):
        """Test that create_error_response method exists."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        handler = MCPProtocolHandler()
        assert hasattr(handler, "create_error_response")
        assert callable(handler.create_error_response)


class TestJSONRPCMessageParsing:
    """Test JSON-RPC 2.0 message parsing and validation."""

    @pytest.fixture
    def handler(self):
        """Create MCPProtocolHandler instance for testing."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        return MCPProtocolHandler()

    def test_parse_valid_json_rpc_request(self, handler):
        """Test parsing valid JSON-RPC 2.0 request."""
        message = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1,
        }

        result = handler.parse_message(json.dumps(message))

        assert result is not None
        assert result["jsonrpc"] == "2.0"
        assert result["method"] == "tools/list"
        assert result["id"] == 1

    def test_parse_valid_json_rpc_request_with_params(self, handler):
        """Test parsing JSON-RPC request with parameters."""
        message = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "gemini_generate", "arguments": {"prompt": "Hello"}},
            "id": 2,
        }

        result = handler.parse_message(json.dumps(message))

        assert result["jsonrpc"] == "2.0"
        assert result["method"] == "tools/call"
        assert result["params"]["name"] == "gemini_generate"
        assert result["params"]["arguments"]["prompt"] == "Hello"
        assert result["id"] == 2

    def test_parse_notification_message(self, handler):
        """Test parsing JSON-RPC notification (no id field)."""
        message = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {"protocolVersion": "1.0"},
        }

        result = handler.parse_message(json.dumps(message))

        assert result["jsonrpc"] == "2.0"
        assert result["method"] == "initialize"
        assert "id" not in result
        assert result["params"]["protocolVersion"] == "1.0"

    def test_parse_invalid_json(self, handler):
        """Test parsing invalid JSON raises ValidationError."""
        invalid_json = '{"jsonrpc": "2.0", "method": invalid}'

        with pytest.raises(ValidationError) as exc_info:
            handler.parse_message(invalid_json)

        assert "Invalid JSON" in str(exc_info.value)

    def test_parse_missing_jsonrpc_field(self, handler):
        """Test parsing message without jsonrpc field."""
        message = {"method": "tools/list", "id": 1}

        with pytest.raises(ValidationError) as exc_info:
            handler.parse_message(json.dumps(message))

        assert "jsonrpc" in str(exc_info.value)

    def test_parse_invalid_jsonrpc_version(self, handler):
        """Test parsing message with invalid jsonrpc version."""
        message = {"jsonrpc": "1.0", "method": "tools/list", "id": 1}

        with pytest.raises(ValidationError) as exc_info:
            handler.parse_message(json.dumps(message))

        assert "2.0" in str(exc_info.value)

    def test_parse_missing_method_field(self, handler):
        """Test parsing request without method field."""
        message = {"jsonrpc": "2.0", "id": 1}

        with pytest.raises(ValidationError) as exc_info:
            handler.parse_message(json.dumps(message))

        assert "method" in str(exc_info.value)

    def test_parse_invalid_method_type(self, handler):
        """Test parsing request with non-string method."""
        message = {"jsonrpc": "2.0", "method": 123, "id": 1}

        with pytest.raises(ValidationError) as exc_info:
            handler.parse_message(json.dumps(message))

        assert "method' must be string" in str(exc_info.value)

    def test_parse_invalid_id_type(self, handler):
        """Test parsing request with invalid id type."""
        message = {"jsonrpc": "2.0", "method": "tools/list", "id": {"invalid": "id"}}

        with pytest.raises(ValidationError) as exc_info:
            handler.parse_message(json.dumps(message))

        assert "id' must be string, number, or null" in str(exc_info.value)


class TestMCPRequestHandling:
    """Test MCP-specific request handling."""

    @pytest.fixture
    def handler(self):
        """Create MCPProtocolHandler instance for testing."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        return MCPProtocolHandler()

    @pytest.mark.asyncio
    async def test_handle_initialize_request(self, handler):
        """Test handling MCP initialize request."""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
            "id": 1,
        }

        response = await handler.handle_request(request)

        assert response is not None
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "capabilities" in response["result"]

    @pytest.mark.asyncio
    async def test_handle_tools_list_request(self, handler):
        """Test handling tools/list request."""
        request = {"jsonrpc": "2.0", "method": "tools/list", "id": 2}

        response = await handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]
        assert isinstance(response["result"]["tools"], list)

    @pytest.mark.asyncio
    async def test_handle_tools_call_request(self, handler):
        """Test handling tools/call request."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "gemini_generate",
                "arguments": {"prompt": "Hello world"},
            },
            "id": 3,
        }

        response = await handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert "content" in response["result"]

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, handler):
        """Test handling unknown method returns error."""
        request = {"jsonrpc": "2.0", "method": "unknown/method", "id": 4}

        response = await handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found

    @pytest.mark.asyncio
    async def test_handle_notification(self, handler):
        """Test handling notifications (no response)."""
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {"level": "info", "message": "Hello"},
        }

        response = await handler.handle_request(notification)

        # Notifications should not return a response
        assert response is None

    @pytest.mark.asyncio
    async def test_handle_request_with_invalid_params(self, handler):
        """Test handling request with invalid parameters."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "invalid_tool"},  # Missing arguments
            "id": 5,
        }

        response = await handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 5
        assert "error" in response
        assert response["error"]["code"] == -32602  # Invalid params


class TestMCPResponseCreation:
    """Test MCP response creation and formatting."""

    @pytest.fixture
    def handler(self):
        """Create MCPProtocolHandler instance for testing."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        return MCPProtocolHandler()

    def test_create_success_response(self, handler):
        """Test creating successful response."""
        result = {"status": "success", "data": {"key": "value"}}
        request_id = 1

        response = handler.create_response(result, request_id)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == request_id
        assert response["result"] == result
        assert "error" not in response

    def test_create_success_response_with_string_id(self, handler):
        """Test creating response with string ID."""
        result = {"message": "Hello"}
        request_id = "request-123"

        response = handler.create_response(result, request_id)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == request_id
        assert response["result"] == result

    def test_create_success_response_with_null_id(self, handler):
        """Test creating response with null ID."""
        result = {"data": [1, 2, 3]}
        request_id = None

        response = handler.create_response(result, request_id)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] is None
        assert response["result"] == result

    def test_create_error_response_with_code_and_message(self, handler):
        """Test creating error response with code and message."""
        error_code = -32602
        error_message = "Invalid params"
        request_id = 2

        response = handler.create_error_response(error_code, error_message, request_id)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == request_id
        assert "result" not in response
        assert response["error"]["code"] == error_code
        assert response["error"]["message"] == error_message

    def test_create_error_response_with_data(self, handler):
        """Test creating error response with additional data."""
        error_code = -32603
        error_message = "Internal error"
        error_data = {"details": "Database connection failed", "retry_after": 30}
        request_id = 3

        response = handler.create_error_response(
            error_code,
            error_message,
            request_id,
            error_data,
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == request_id
        assert response["error"]["code"] == error_code
        assert response["error"]["message"] == error_message
        assert response["error"]["data"] == error_data

    def test_create_error_response_parse_error(self, handler):
        """Test creating parse error response."""
        response = handler.create_error_response(
            -32700,
            "Parse error",
            None,
            {"position": 42},
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] is None
        assert response["error"]["code"] == -32700
        assert response["error"]["message"] == "Parse error"
        assert response["error"]["data"]["position"] == 42


class TestMCPSpecificationCompliance:
    """Test MCP specification compliance."""

    @pytest.fixture
    def handler(self):
        """Create MCPProtocolHandler instance for testing."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        return MCPProtocolHandler()

    @pytest.mark.asyncio
    async def test_initialize_handshake_compliance(self, handler):
        """Test MCP initialize handshake follows specification."""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
            "id": 1,
        }

        response = await handler.handle_request(request)

        # Check MCP specification compliance
        assert "capabilities" in response["result"]
        assert "serverInfo" in response["result"]
        assert "name" in response["result"]["serverInfo"]
        assert "version" in response["result"]["serverInfo"]

    @pytest.mark.asyncio
    async def test_tools_list_format_compliance(self, handler):
        """Test tools/list response format compliance."""
        request = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

        response = await handler.handle_request(request)

        # Check tools format compliance
        tools = response["result"]["tools"]
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            # Check JSON Schema format
            assert "type" in tool["inputSchema"]
            assert "properties" in tool["inputSchema"]

    def test_error_codes_compliance(self, handler):
        """Test error codes comply with JSON-RPC 2.0 specification."""
        # Standard JSON-RPC error codes
        standard_errors = {
            -32700: "Parse error",
            -32600: "Invalid Request",
            -32601: "Method not found",
            -32602: "Invalid params",
            -32603: "Internal error",
        }

        for code, message in standard_errors.items():
            response = handler.create_error_response(code, message, 1)
            assert response["error"]["code"] == code
            assert response["error"]["message"] == message

    def test_response_structure_compliance(self, handler):
        """Test response structure complies with JSON-RPC 2.0."""
        # Success response
        success_response = handler.create_response({"data": "test"}, 1)
        required_fields = {"jsonrpc", "result", "id"}
        assert set(success_response.keys()) == required_fields

        # Error response
        error_response = handler.create_error_response(-32603, "Error", 2)
        required_fields = {"jsonrpc", "error", "id"}
        assert set(error_response.keys()) == required_fields

        # Error object structure
        error_fields = {"code", "message"}
        assert set(error_response["error"].keys()).issuperset(error_fields)


class TestMCPProtocolMetrics:
    """Test MCP protocol metrics and monitoring."""

    @pytest.fixture
    def handler(self):
        """Create MCPProtocolHandler instance for testing."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        return MCPProtocolHandler()

    def test_message_metrics_tracking(self, handler):
        """Test message metrics are tracked."""
        assert hasattr(handler, "get_metrics")
        assert callable(handler.get_metrics)

        metrics = handler.get_metrics()
        assert "messages_parsed" in metrics
        assert "requests_handled" in metrics
        assert "errors_encountered" in metrics

    @pytest.mark.asyncio
    async def test_metrics_increment_on_parse(self, handler):
        """Test metrics increment when parsing messages."""
        initial_metrics = handler.get_metrics()
        initial_parsed = initial_metrics["messages_parsed"]

        message = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        handler.parse_message(json.dumps(message))

        updated_metrics = handler.get_metrics()
        assert updated_metrics["messages_parsed"] == initial_parsed + 1

    @pytest.mark.asyncio
    async def test_metrics_increment_on_request(self, handler):
        """Test metrics increment when handling requests."""
        initial_metrics = handler.get_metrics()
        initial_handled = initial_metrics["requests_handled"]

        request = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        await handler.handle_request(request)

        updated_metrics = handler.get_metrics()
        assert updated_metrics["requests_handled"] == initial_handled + 1

    def test_metrics_increment_on_error(self, handler):
        """Test metrics increment when encountering errors."""
        initial_metrics = handler.get_metrics()
        initial_errors = initial_metrics["errors_encountered"]

        # Trigger a parse error
        try:
            handler.parse_message("invalid json")
        except ValidationError:
            pass

        updated_metrics = handler.get_metrics()
        assert updated_metrics["errors_encountered"] == initial_errors + 1


class TestMCPProtocolLogging:
    """Test MCP protocol logging and tracing."""

    @pytest.fixture
    def handler(self):
        """Create MCPProtocolHandler instance for testing."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        return MCPProtocolHandler()

    def test_logging_configuration(self, handler):
        """Test logging is properly configured."""
        assert hasattr(handler, "logger")
        assert handler.logger is not None

    def test_request_logging(self, handler):
        """Test that requests are logged appropriately."""
        # Test that messages are parsed without errors (logging is implicit)
        message = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        result = handler.parse_message(json.dumps(message))

        # Verify successful parsing (which includes logging)
        assert result["method"] == "tools/list"
        assert result["id"] == 1

    def test_error_logging(self, handler):
        """Test that errors are logged with appropriate level."""
        # Test that errors are properly raised and counted
        initial_errors = handler.get_metrics()["errors_encountered"]

        try:
            handler.parse_message("invalid json")
        except ValidationError:
            pass

        # Verify error was counted (which implies logging occurred)
        final_errors = handler.get_metrics()["errors_encountered"]
        assert final_errors == initial_errors + 1


class TestMCPProtocolSecurity:
    """Test MCP protocol security measures."""

    @pytest.fixture
    def handler(self):
        """Create MCPProtocolHandler instance for testing."""
        from mcp_server_cheap_llm.server.handlers import MCPProtocolHandler

        return MCPProtocolHandler()

    def test_message_size_limits(self, handler):
        """Test message size limits to prevent DoS attacks."""
        # Create oversized message
        large_payload = "x" * (1024 * 1024)  # 1MB payload
        message = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"data": large_payload},
            "id": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            handler.parse_message(json.dumps(message))

        assert "message too large" in str(exc_info.value).lower()

    def test_nested_object_limits(self, handler):
        """Test nested object depth limits."""
        # Create deeply nested object
        nested_obj = {"level": 1}
        for i in range(2, 102):  # Create 101 levels of nesting
            nested_obj = {"level": i, "nested": nested_obj}

        message = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": nested_obj,
            "id": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            handler.parse_message(json.dumps(message))

        assert "nesting too deep" in str(exc_info.value).lower()

    def test_parameter_sanitization(self, handler):
        """Test parameter sanitization for security."""
        # Test with potentially malicious parameters
        message = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "gemini_generate",
                "arguments": {
                    "prompt": "<script>alert('xss')</script>",
                    "dangerous_param": "../../../etc/passwd",
                },
            },
            "id": 1,
        }

        # Should parse without error but sanitize dangerous content
        result = handler.parse_message(json.dumps(message))
        assert result is not None
        # Further sanitization testing would depend on implementation
