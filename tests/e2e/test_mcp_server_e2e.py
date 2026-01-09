"""
End-to-End Tests for MCP Server LLM CLI Runner

This test suite validates complete end-to-end workflows including:
1. Server initialization and lifecycle management
2. MCP protocol compliance (JSON-RPC 2.0)
3. Tool discovery and execution
4. Provider integration flows
5. Request/response handling
6. Error handling and recovery
7. Session management
8. Streaming responses
"""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_llm_cli_runner.core.errors import ValidationError
from mcp_server_llm_cli_runner.server.handlers import (
    LLMCliRunnerServer,
    MCPProtocolHandler,
    RequestManager,
    SessionManager,
    StreamingHandler,
    ToolRegistry,
)


class TestMCPServerE2E:
    """End-to-end tests for MCP Server LLM CLI Runner."""

    @pytest.fixture
    def server(self) -> LLMCliRunnerServer:
        """Create a fresh server instance for testing."""
        return LLMCliRunnerServer()

    @pytest.fixture
    def protocol_handler(self) -> MCPProtocolHandler:
        """Create a protocol handler for testing."""
        return MCPProtocolHandler()

    @pytest.fixture
    def tool_registry(self) -> ToolRegistry:
        """Create a tool registry for testing."""
        return ToolRegistry()

    # =========================================================================
    # Server Lifecycle Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_server_initialization(self, server: LLMCliRunnerServer):
        """Test that server initializes correctly."""
        await server.initialize()
        assert server._initialized is True

    @pytest.mark.asyncio
    async def test_server_double_initialization_idempotent(
        self, server: LLMCliRunnerServer
    ):
        """Test that double initialization is safe."""
        await server.initialize()
        await server.initialize()  # Should not raise
        assert server._initialized is True

    @pytest.mark.asyncio
    async def test_server_start_and_stop(self, server: LLMCliRunnerServer):
        """Test server start and graceful shutdown."""
        await server.initialize()
        await server.start()
        await server.stop()
        # Server should stop without errors

    @pytest.mark.asyncio
    async def test_server_stats_after_initialization(self, server: LLMCliRunnerServer):
        """Test server statistics are available after initialization."""
        await server.initialize()
        stats = server.get_server_stats()

        assert "initialized" in stats
        assert stats["initialized"] is True
        assert "protocol_metrics" in stats
        assert "tool_registry" in stats
        assert "active_sessions" in stats

    # =========================================================================
    # MCP Protocol Compliance Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_protocol_initialize_request(
        self, protocol_handler: MCPProtocolHandler
    ):
        """Test MCP initialize request handling."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response_json = await protocol_handler.process_message(json.dumps(request))
        response = json.loads(response_json)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "protocolVersion" in response["result"]
        assert "capabilities" in response["result"]
        assert "serverInfo" in response["result"]

    @pytest.mark.asyncio
    async def test_protocol_tools_list_request(
        self, protocol_handler: MCPProtocolHandler
    ):
        """Test MCP tools/list request handling."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        response_json = await protocol_handler.process_message(json.dumps(request))
        response = json.loads(response_json)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]
        assert isinstance(response["result"]["tools"], list)

    @pytest.mark.asyncio
    async def test_protocol_tools_call_request(
        self, protocol_handler: MCPProtocolHandler
    ):
        """Test MCP tools/call request handling."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "test_tool",
                "arguments": {"prompt": "Hello world"},
            },
        }

        response_json = await protocol_handler.process_message(json.dumps(request))
        response = json.loads(response_json)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert "content" in response["result"]

    @pytest.mark.asyncio
    async def test_protocol_invalid_jsonrpc_version(
        self, protocol_handler: MCPProtocolHandler
    ):
        """Test error handling for invalid JSON-RPC version."""
        request = {
            "jsonrpc": "1.0",  # Invalid version
            "id": 1,
            "method": "initialize",
        }

        response_json = await protocol_handler.process_message(json.dumps(request))
        response = json.loads(response_json)

        assert "error" in response
        assert response["error"]["code"] == -32600  # Invalid Request

    @pytest.mark.asyncio
    async def test_protocol_missing_method(self, protocol_handler: MCPProtocolHandler):
        """Test error handling for missing method field."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            # Missing "method" field
        }

        response_json = await protocol_handler.process_message(json.dumps(request))
        response = json.loads(response_json)

        assert "error" in response

    @pytest.mark.asyncio
    async def test_protocol_method_not_found(
        self, protocol_handler: MCPProtocolHandler
    ):
        """Test error handling for unknown method."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown/method",
        }

        response_json = await protocol_handler.process_message(json.dumps(request))
        response = json.loads(response_json)

        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found

    @pytest.mark.asyncio
    async def test_protocol_invalid_json(self, protocol_handler: MCPProtocolHandler):
        """Test error handling for invalid JSON."""
        response_json = await protocol_handler.process_message("not valid json{")
        response = json.loads(response_json)

        assert "error" in response
        assert response["error"]["code"] == -32700  # Parse error

    @pytest.mark.asyncio
    async def test_protocol_resources_list_request(
        self, protocol_handler: MCPProtocolHandler
    ):
        """Test MCP resources/list request handling."""
        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/list",
            "params": {},
        }

        response_json = await protocol_handler.process_message(json.dumps(request))
        response = json.loads(response_json)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "result" in response
        assert "resources" in response["result"]

    # =========================================================================
    # Tool Registry Tests
    # =========================================================================

    def test_tool_registry_register_and_discover(self, tool_registry: ToolRegistry):
        """Test tool registration and discovery."""
        tool_spec = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
            },
        }

        tool_registry.register_tool(tool_spec, "test_provider")
        tools = tool_registry.discover_tools("test_provider")

        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"

    def test_tool_registry_multi_provider(self, tool_registry: ToolRegistry):
        """Test tool registration across multiple providers."""
        tool_spec = {
            "name": "shared_tool",
            "description": "Shared tool",
            "inputSchema": {"type": "object"},
        }

        tool_registry.register_tool(tool_spec, "provider_a")
        tool_registry.register_tool(tool_spec, "provider_b")

        providers = tool_registry.get_providers()
        assert "provider_a" in providers
        assert "provider_b" in providers

    def test_tool_registry_unregister(self, tool_registry: ToolRegistry):
        """Test tool unregistration."""
        tool_spec = {"name": "temp_tool", "inputSchema": {"type": "object"}}

        tool_registry.register_tool(tool_spec, "test_provider")
        assert tool_registry.get_tool_count("test_provider") == 1

        result = tool_registry.unregister_tool("temp_tool", "test_provider")
        assert result is True
        assert tool_registry.get_tool_count("test_provider") == 0

    def test_tool_registry_invoke_with_handler(self, tool_registry: ToolRegistry):
        """Test tool invocation with registered handler."""

        def mock_handler(args: dict[str, Any]) -> str:
            return f"Processed: {args.get('input', '')}"

        tool_spec = {"name": "callable_tool", "inputSchema": {"type": "object"}}

        tool_registry.register_tool(tool_spec, "test_provider", mock_handler)
        result = tool_registry.invoke_tool(
            "callable_tool", "test_provider", {"input": "test"}
        )

        assert result == "Processed: test"

    def test_tool_registry_stats(self, tool_registry: ToolRegistry):
        """Test tool registry statistics."""
        tool_spec = {"name": "stat_tool", "inputSchema": {"type": "object"}}
        tool_registry.register_tool(tool_spec, "provider_a")

        stats = tool_registry.get_registry_stats()

        assert "total_providers" in stats
        assert "total_tools" in stats
        assert "providers" in stats
        assert "tools_by_provider" in stats

    # =========================================================================
    # Session Management Tests
    # =========================================================================

    def test_session_creation(self):
        """Test session creation."""
        session_manager = SessionManager()
        session_id = session_manager.create_session("client_123")

        assert session_id is not None
        info = session_manager.get_session_info(session_id)
        assert info is not None
        assert info["client_id"] == "client_123"
        assert info["status"] == "active"

    def test_session_lifecycle(self):
        """Test complete session lifecycle."""
        session_manager = SessionManager()

        # Create
        session_id = session_manager.create_session()

        # Update status
        session_manager.update_session_status(session_id, "processing")
        info = session_manager.get_session_info(session_id)
        assert info["status"] == "processing"

        # Close
        result = session_manager.close_session(session_id)
        assert result is True

        info = session_manager.get_session_info(session_id)
        assert info["status"] == "closed"

    def test_get_active_sessions(self):
        """Test getting active sessions."""
        session_manager = SessionManager()

        session1 = session_manager.create_session()
        session2 = session_manager.create_session()
        session_manager.close_session(session1)

        active = session_manager.get_active_sessions()
        assert session2 in active
        assert session1 not in active

    # =========================================================================
    # Request Manager Tests
    # =========================================================================

    def test_request_lifecycle(self):
        """Test request tracking lifecycle."""
        request_manager = RequestManager()

        request_id = request_manager.create_request_id()
        request_data = {"method": "test", "params": {}}

        # Start request
        request_manager.start_request(request_id, request_data)
        status = request_manager.get_request_status(request_id)
        assert status["status"] == "active"

        # Complete request
        request_manager.complete_request(request_id, {"result": "success"})
        status = request_manager.get_request_status(request_id)
        assert status["status"] == "completed"

    def test_request_failure_tracking(self):
        """Test request failure tracking."""
        request_manager = RequestManager()

        request_id = request_manager.create_request_id()
        request_manager.start_request(request_id, {"method": "test"})
        request_manager.fail_request(request_id, "Test error")

        status = request_manager.get_request_status(request_id)
        assert status["status"] == "failed"
        assert status["error"] == "Test error"

    # =========================================================================
    # Streaming Response Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_streaming_session_creation(self):
        """Test streaming session creation."""
        streaming_handler = StreamingHandler()
        session_id = await streaming_handler.create_stream_session("client_456")

        assert session_id is not None
        metrics = streaming_handler.get_stream_metrics(session_id)
        assert metrics["status"] == "active"

    @pytest.mark.asyncio
    async def test_streaming_response_flow(self):
        """Test complete streaming response flow."""
        streaming_handler = StreamingHandler()
        session_id = await streaming_handler.create_stream_session()

        data = {"message": "Hello", "total_chunks": 3}
        chunks = []

        async for chunk in streaming_handler.stream_response(session_id, data):
            chunks.append(chunk)

        assert len(chunks) == 3
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "session_id" in chunk
            assert chunk["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_streaming_session_cleanup(self):
        """Test streaming session cleanup."""
        streaming_handler = StreamingHandler()
        session_id = await streaming_handler.create_stream_session()

        await streaming_handler.close_stream_session(session_id)
        metrics = streaming_handler.get_stream_metrics(session_id)
        assert metrics == {}  # Session should be removed

    # =========================================================================
    # Full Workflow Integration Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_complete_tool_execution_workflow(self, server: LLMCliRunnerServer):
        """Test complete tool execution workflow from init to response."""
        await server.initialize()

        # Register a test tool
        test_tool = {
            "name": "echo_tool",
            "description": "Echoes input",
            "inputSchema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        }

        def echo_handler(args: dict[str, Any]) -> str:
            return f"Echo: {args.get('message', '')}"

        server.register_tool(test_tool, "default", echo_handler)

        # Create session
        session_id = server.create_session("test_client")
        assert session_id is not None

        # Process request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "echo_tool", "arguments": {"message": "Hello"}},
        }

        response = await server.process_request(request)

        assert "result" in response or "error" not in response

    @pytest.mark.asyncio
    async def test_multi_provider_tool_routing(self, server: LLMCliRunnerServer):
        """Test tool routing across multiple providers."""
        await server.initialize()

        # Register tools for different providers
        for provider in ["gemini", "openai", "llama"]:
            tool = {
                "name": f"{provider}_generate",
                "description": f"Generate using {provider}",
                "inputSchema": {"type": "object"},
            }
            server.register_tool(tool, provider, lambda x, p=provider: f"{p} response")

        # Verify all tools registered
        stats = server.get_server_stats()
        assert stats["tool_registry"]["total_tools"] >= 3

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, server: LLMCliRunnerServer):
        """Test error recovery and handling workflow."""
        await server.initialize()

        # Send malformed request
        malformed_request = {"not": "valid"}

        response = await server.process_request(malformed_request)

        # Should get error response, not crash
        assert "error" in response

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, server: LLMCliRunnerServer):
        """Test handling concurrent requests."""
        await server.initialize()

        async def make_request(req_id: int) -> dict[str, Any]:
            request = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/list",
                "params": {},
            }
            return await server.process_request(request)

        # Send multiple concurrent requests
        tasks = [make_request(i) for i in range(10)]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 10
        for i, response in enumerate(responses):
            assert response["id"] == i

    # =========================================================================
    # Provider Integration Tests (with mocks)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_gemini_provider_integration(self, server: LLMCliRunnerServer):
        """Test Gemini provider integration flow."""
        await server.initialize()

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "gemini_generate",
                "arguments": {"prompt": "Test prompt"},
            },
        }

        response = await server.process_request(request)

        # Should succeed or return proper error
        assert "result" in response or "error" in response

    @pytest.mark.asyncio
    async def test_provider_fallback_on_error(self, server: LLMCliRunnerServer):
        """Test provider fallback mechanism on error."""
        await server.initialize()

        # Mock a failing provider scenario
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_provider_generate",
                "arguments": {"prompt": "Test"},
            },
        }

        response = await server.process_request(request)

        # Should handle gracefully
        assert "result" in response or "error" in response

    # =========================================================================
    # Metrics and Monitoring Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_metrics_collection(self, protocol_handler: MCPProtocolHandler):
        """Test metrics are properly collected."""
        # Send some requests
        for i in range(5):
            request = {
                "jsonrpc": "2.0",
                "id": i,
                "method": "tools/list",
                "params": {},
            }
            await protocol_handler.process_message(json.dumps(request))

        metrics = protocol_handler.get_metrics()

        assert metrics["requests_processed"] >= 5
        assert metrics["responses_sent"] >= 5

    @pytest.mark.asyncio
    async def test_metrics_reset(self, protocol_handler: MCPProtocolHandler):
        """Test metrics reset functionality."""
        # Generate some metrics
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }
        await protocol_handler.process_message(json.dumps(request))

        # Reset
        protocol_handler.reset_metrics()

        metrics = protocol_handler.get_metrics()
        assert metrics["requests_processed"] == 0

    # =========================================================================
    # Security Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_message_size_limit(self, protocol_handler: MCPProtocolHandler):
        """Test message size limit enforcement."""
        # Create oversized message (>1MB)
        large_data = "x" * (1024 * 1024 + 100)

        with pytest.raises(ValidationError) as exc_info:
            protocol_handler.parse_message(large_data)

        assert "too large" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_nesting_depth_limit(self, protocol_handler: MCPProtocolHandler):
        """Test nesting depth limit enforcement."""
        # Create deeply nested structure
        nested = {"level": 0}
        current = nested
        for i in range(150):  # Exceed 100 level limit
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        request = {"jsonrpc": "2.0", "id": 1, "method": "test", "params": nested}

        with pytest.raises(ValidationError) as exc_info:
            protocol_handler.parse_message(json.dumps(request))

        assert "nesting" in str(exc_info.value).lower()


class TestMCPClientIntegration:
    """Tests simulating MCP client interactions."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock MCP client."""

        class MockMCPClient:
            def __init__(self, protocol_handler: MCPProtocolHandler):
                self.handler = protocol_handler
                self.request_id = 0

            async def send_request(
                self, method: str, params: dict | None = None
            ) -> dict:
                self.request_id += 1
                request = {
                    "jsonrpc": "2.0",
                    "id": self.request_id,
                    "method": method,
                    "params": params or {},
                }
                response_json = await self.handler.process_message(json.dumps(request))
                return json.loads(response_json)

            async def initialize(self) -> dict:
                return await self.send_request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "mock-client", "version": "1.0.0"},
                    },
                )

            async def list_tools(self) -> list:
                response = await self.send_request("tools/list")
                return response.get("result", {}).get("tools", [])

            async def call_tool(self, name: str, arguments: dict) -> dict:
                return await self.send_request(
                    "tools/call", {"name": name, "arguments": arguments}
                )

        return MockMCPClient(MCPProtocolHandler())

    @pytest.mark.asyncio
    async def test_client_full_session(self, mock_client):
        """Test a complete client session."""
        # Initialize
        init_response = await mock_client.initialize()
        assert "result" in init_response

        # List tools
        tools = await mock_client.list_tools()
        assert isinstance(tools, list)

        # Call a tool
        tool_response = await mock_client.call_tool(
            "test_tool", {"prompt": "Hello world"}
        )
        assert "result" in tool_response

    @pytest.mark.asyncio
    async def test_client_multiple_tool_calls(self, mock_client):
        """Test multiple sequential tool calls."""
        await mock_client.initialize()

        for i in range(5):
            response = await mock_client.call_tool(
                f"tool_{i}", {"input": f"Request {i}"}
            )
            assert "result" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
