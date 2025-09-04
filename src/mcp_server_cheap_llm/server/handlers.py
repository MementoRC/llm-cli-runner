"""MCP Server handlers for cheap LLM providers."""

import json
from typing import Any

from mcp.server import Server  # type: ignore[import-not-found]
from mcp.types import (  # type: ignore[import-not-found]
    CallToolRequest,
    CallToolResult,
    TextContent,
    Tool,
)

from mcp_server_cheap_llm.core.errors import ValidationError
from mcp_server_cheap_llm.utils.config import ConfigManager
from mcp_server_cheap_llm.utils.logging import get_logger


class MCPProtocolHandler:
    """MCP Protocol handler for JSON-RPC 2.0 message processing.

    This class handles the core MCP protocol infrastructure including:
    - JSON-RPC 2.0 message parsing and validation
    - Request/response handling with proper error management
    - Protocol compliance and security measures
    - Metrics tracking and logging
    """

    def __init__(self):
        """Initialize the MCP protocol handler."""
        self.logger = get_logger(__name__)
        self._metrics = {
            "messages_parsed": 0,
            "requests_handled": 0,
            "errors_encountered": 0,
        }

        # Security limits
        self._max_message_size = 512 * 1024  # 512KB
        self._max_nesting_depth = 100

    def parse_message(self, message: str) -> dict[str, Any]:
        """Parse and validate JSON-RPC 2.0 message.

        Args:
            message: Raw JSON message string

        Returns:
            Parsed message dictionary

        Raises:
            ValidationError: If message is invalid or doesn't comply with JSON-RPC 2.0
        """
        try:
            # Security check: message size limit
            if len(message) > self._max_message_size:
                self._metrics["errors_encountered"] += 1
                raise ValidationError(
                    "Message too large",
                    error_code="MSG001",
                    context={"size": len(message), "max_size": self._max_message_size},
                )

            # Parse JSON
            try:
                parsed = json.loads(message)
            except json.JSONDecodeError as e:
                self._metrics["errors_encountered"] += 1
                raise ValidationError(
                    f"Invalid JSON: {str(e)}",
                    error_code="JSON001",
                    context={"position": e.pos if hasattr(e, "pos") else None},
                ) from e

            # Security check: nesting depth
            if self._check_nesting_depth(parsed) > self._max_nesting_depth:
                self._metrics["errors_encountered"] += 1
                raise ValidationError(
                    "Nesting too deep",
                    error_code="MSG002",
                    context={"max_depth": self._max_nesting_depth},
                )

            # Validate JSON-RPC 2.0 structure
            self._validate_jsonrpc_message(parsed)

            self._metrics["messages_parsed"] += 1
            self.logger.debug(f"Parsed message: {parsed.get('method', 'unknown')}")

            return parsed

        except ValidationError:
            raise
        except Exception as e:
            self._metrics["errors_encountered"] += 1
            raise ValidationError(
                f"Message parsing failed: {str(e)}",
                error_code="MSG003",
                context={"error_type": type(e).__name__},
            ) from e

    def _validate_jsonrpc_message(self, message: dict[str, Any]) -> None:
        """Validate JSON-RPC 2.0 message structure."""
        # Check required jsonrpc field
        if "jsonrpc" not in message:
            raise ValidationError(
                "Missing required 'jsonrpc' field", error_code="RPC001"
            )

        # Check JSON-RPC version
        if message["jsonrpc"] != "2.0":
            raise ValidationError(
                "Invalid jsonrpc version, must be '2.0'",
                error_code="RPC002",
                context={"version": message.get("jsonrpc")},
            )

        # Check method field for requests
        if "method" not in message:
            raise ValidationError(
                "Missing required 'method' field", error_code="RPC003"
            )

        # Validate method type
        if not isinstance(message["method"], str):
            raise ValidationError(
                "Field 'method' must be string",
                error_code="RPC004",
                context={"method_type": type(message["method"]).__name__},
            )

        # Validate id field if present
        if "id" in message:
            id_value = message["id"]
            if not (isinstance(id_value, str | int | float) or id_value is None):
                raise ValidationError(
                    "Field 'id' must be string, number, or null",
                    error_code="RPC005",
                    context={"id_type": type(id_value).__name__},
                )

    def _check_nesting_depth(self, obj: Any, depth: int = 0) -> int:
        """Check nesting depth of object to prevent stack overflow."""
        if depth > self._max_nesting_depth:
            return depth

        max_depth = depth

        if isinstance(obj, dict):
            for value in obj.values():
                max_depth = max(max_depth, self._check_nesting_depth(value, depth + 1))
        elif isinstance(obj, list):
            for item in obj:
                max_depth = max(max_depth, self._check_nesting_depth(item, depth + 1))

        return max_depth

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle MCP request and generate response.

        Args:
            request: Parsed JSON-RPC request

        Returns:
            Response dictionary or None for notifications
        """
        try:
            method = request.get("method")
            request_id = request.get("id")
            params = request.get("params", {})

            self._metrics["requests_handled"] += 1
            self.logger.info(f"Handling request: {method}")

            # Handle notifications (no response required)
            if request_id is None:
                if method == "notifications/message":
                    self.logger.info(f"Notification: {params.get('message', '')}")
                return None

            # Handle MCP methods
            if method == "initialize":
                return self._handle_initialize(request_id, params)
            elif method == "tools/list":
                return self._handle_tools_list(request_id)
            elif method == "tools/call":
                return await self._handle_tools_call(request_id, params)
            else:
                return self.create_error_response(
                    -32601,  # Method not found
                    f"Method not found: {method}",
                    request_id,
                )

        except Exception as e:
            self.logger.error(f"Request handling failed: {e}")
            return self.create_error_response(
                -32603,  # Internal error
                "Internal error",
                request.get("id"),
                {"details": str(e)},
            )

    def _handle_initialize(
        self, request_id: str | int | None, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle MCP initialize request."""
        # MCP protocol handshake
        result = {
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "cheap-llm-server", "version": "1.0.0"},
        }

        return self.create_response(result, request_id)

    def _handle_tools_list(self, request_id: str | int | None) -> dict[str, Any]:
        """Handle tools/list request."""
        # Return mock tools for now - this would integrate with actual tool discovery
        tools = [
            {
                "name": "gemini_generate",
                "description": "Generate text using Gemini CLI",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate text from",
                        }
                    },
                    "required": ["prompt"],
                },
            }
        ]

        result = {"tools": tools}
        return self.create_response(result, request_id)

    async def _handle_tools_call(
        self, request_id: str | int | None, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            return self.create_error_response(
                -32602,  # Invalid params
                "Missing required parameter: name",
                request_id,
            )

        # Mock tool execution
        if tool_name == "gemini_generate":
            prompt = arguments.get("prompt", "")
            content = [{"type": "text", "text": f"Generated response for: {prompt}"}]
            result = {"content": content}
            return self.create_response(result, request_id)
        else:
            return self.create_error_response(
                -32602,  # Invalid params
                f"Unknown tool: {tool_name}",
                request_id,
            )

    def create_response(
        self, result: Any, request_id: str | int | None
    ) -> dict[str, Any]:
        """Create successful JSON-RPC 2.0 response.

        Args:
            result: Response result data
            request_id: Request ID to echo back

        Returns:
            JSON-RPC 2.0 response dictionary
        """
        return {"jsonrpc": "2.0", "result": result, "id": request_id}

    def create_error_response(
        self,
        code: int,
        message: str,
        request_id: str | int | None,
        data: Any | None = None,
    ) -> dict[str, Any]:
        """Create JSON-RPC 2.0 error response.

        Args:
            code: JSON-RPC error code
            message: Error message
            request_id: Request ID to echo back
            data: Optional additional error data

        Returns:
            JSON-RPC 2.0 error response dictionary
        """
        error = {"code": code, "message": message}

        if data is not None:
            error["data"] = data

        return {"jsonrpc": "2.0", "error": error, "id": request_id}

    def get_metrics(self) -> dict[str, int]:
        """Get protocol handler metrics.

        Returns:
            Dictionary of metrics counters
        """
        return self._metrics.copy()


class CheapLLMServer:
    """Main MCP server implementation for cheap LLM providers."""

    def __init__(self, config_manager: ConfigManager):
        """Initialize the CheapLLMServer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = get_logger(__name__)
        self._server = Server("cheap-llm")
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup MCP server handlers."""

        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return await self._list_tools()

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            from mcp.types import CallToolRequestParams

            params = CallToolRequestParams(name=name, arguments=arguments)
            request = CallToolRequest(method="tools/call", params=params)
            result = await self._call_tool(request)
            # Extract TextContent from result.content
            return [
                content
                for content in result.content
                if isinstance(content, TextContent)
            ]

        self.logger.info("Server handlers initialized")

    async def _list_tools(self) -> list[Tool]:
        """List available tools based on enabled providers.

        Returns:
            List of available tools
        """
        tools = []
        enabled_providers = self.config_manager.get_enabled_providers()

        for provider in enabled_providers:
            if provider == "gemini":
                tools.append(
                    Tool(
                        name="gemini_generate",
                        description="Generate text using Gemini CLI",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The prompt to generate text from",
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Gemini model to use (optional)",
                                    "default": "gemini-1.5-flash",
                                },
                            },
                            "required": ["prompt"],
                        },
                    )
                )

            elif provider == "codex":
                tools.append(
                    Tool(
                        name="codex_generate",
                        description="Generate code using OpenAI Codex",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The code prompt to generate from",
                                },
                                "language": {
                                    "type": "string",
                                    "description": "Programming language (optional)",
                                    "default": "python",
                                },
                            },
                            "required": ["prompt"],
                        },
                    )
                )

            elif provider == "llama":
                tools.append(
                    Tool(
                        name="llama_generate",
                        description="Generate text using local LLaMA model",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The prompt to generate text from",
                                },
                                "max_tokens": {
                                    "type": "integer",
                                    "description": "Maximum tokens to generate (optional)",
                                    "default": 256,
                                },
                            },
                            "required": ["prompt"],
                        },
                    )
                )

        self.logger.info(
            f"Listed {len(tools)} tools for providers: {enabled_providers}"
        )
        return tools

    async def _call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Call a tool based on the request.

        Args:
            request: Tool call request

        Returns:
            Tool call result
        """
        tool_name = request.params.name
        arguments = request.params.arguments or {}

        self.logger.info(f"Calling tool: {tool_name}")

        try:
            if tool_name == "gemini_generate":
                response = await self._call_gemini(arguments)
            elif tool_name == "codex_generate":
                response = await self._call_codex(arguments)
            elif tool_name == "llama_generate":
                response = await self._call_llama(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            return CallToolResult(content=[TextContent(type="text", text=response)])

        except Exception as e:
            self.logger.error(f"Tool call failed: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True,
            )

    async def _call_gemini(self, arguments: dict[str, Any]) -> str:
        """Call Gemini CLI provider.

        Args:
            arguments: Tool arguments

        Returns:
            Generated response
        """
        prompt = arguments["prompt"]
        model = arguments.get("model", "gemini-1.5-flash")

        # TODO: Implement actual Gemini CLI call
        # For now, return a placeholder
        return f"Gemini ({model}) response to: {prompt}"

    async def _call_codex(self, arguments: dict[str, Any]) -> str:
        """Call OpenAI Codex provider.

        Args:
            arguments: Tool arguments

        Returns:
            Generated response
        """
        prompt = arguments["prompt"]
        language = arguments.get("language", "python")

        # TODO: Implement actual Codex API call
        # For now, return a placeholder
        return f"Codex ({language}) response to: {prompt}"

    async def _call_llama(self, arguments: dict[str, Any]) -> str:
        """Call local LLaMA provider.

        Args:
            arguments: Tool arguments

        Returns:
            Generated response
        """
        prompt = arguments["prompt"]
        max_tokens = arguments.get("max_tokens", 256)

        # TODO: Implement actual LLaMA call
        # For now, return a placeholder
        return f"LLaMA (max_tokens={max_tokens}) response to: {prompt}"

    def get_mcp_server(self) -> Server:
        """Get the MCP server instance.

        Returns:
            MCP Server instance
        """
        return self._server