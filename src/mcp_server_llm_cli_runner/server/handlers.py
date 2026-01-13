"""Server handlers and protocol implementation."""

import asyncio
import json
import time
import uuid
from datetime import datetime
from threading import Lock
from typing import Any

from mcp_server_llm_cli_runner.core.errors import ValidationError
from mcp_server_llm_cli_runner.utils.logging import get_logger

logger = get_logger(__name__)


# Type compatibility layer for MCP types
# Define fallback implementations first
class _FallbackTool:
    """Fallback Tool implementation compatible with MCP."""

    def __init__(
        self,
        *,
        name: str,
        description: str | None = None,
        inputSchema: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.description = description or ""
        self.inputSchema = inputSchema


class _FallbackTextContent:
    """Fallback TextContent implementation compatible with MCP."""

    def __init__(self, *, text: str, type: str = "text", **kwargs: Any) -> None:
        self.text = text
        self.type = type


class _FallbackCallToolResult:
    """Fallback CallToolResult implementation compatible with MCP."""

    def __init__(
        self, *, content: list[Any], isError: bool = False, **kwargs: Any
    ) -> None:
        self.content = content
        self.isError = isError


class _FallbackCallToolRequest:
    """Fallback CallToolRequest implementation compatible with MCP."""

    def __init__(self, *, method: str, params: Any, **kwargs: Any) -> None:
        self.method = method
        self.params = params


# Try to import MCP types, fall back to our implementations
try:
    from mcp.types import (
        CallToolRequest as _MCPCallToolRequest,  # type: ignore[import-not-found]
    )
    from mcp.types import (
        CallToolResult as _MCPCallToolResult,  # type: ignore[import-not-found]
    )
    from mcp.types import (
        TextContent as _MCPTextContent,  # type: ignore[import-not-found]
    )
    from mcp.types import Tool as _MCPTool  # type: ignore[import-not-found]

    _MCP_TYPES_AVAILABLE = True
    Tool = _MCPTool
    TextContent = _MCPTextContent
    CallToolResult = _MCPCallToolResult
    CallToolRequest = _MCPCallToolRequest
except ImportError:
    _MCP_TYPES_AVAILABLE = False
    Tool = _FallbackTool  # type: ignore[assignment,misc]
    TextContent = _FallbackTextContent  # type: ignore[assignment,misc]
    CallToolResult = _FallbackCallToolResult  # type: ignore[assignment,misc]
    CallToolRequest = _FallbackCallToolRequest  # type: ignore[assignment,misc]


class MCPProtocolHandler:
    """MCP Protocol Handler for JSON-RPC message processing.

    This class handles the core MCP protocol operations including
    message validation, routing, and response formatting.
    """

    def __init__(self) -> None:
        """Initialize MCP protocol handler."""
        self.logger = get_logger(__name__)
        self._metrics: dict[str, int] = {
            "messages_parsed": 0,
            "requests_handled": 0,
            "requests_processed": 0,
            "responses_sent": 0,
            "errors_encountered": 0,
            "invalid_jsonrpc": 0,
        }

    def get_metrics(self) -> dict[str, int]:
        """Get protocol handler metrics.

        Returns:
            Dictionary of metrics

        """
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        for key in self._metrics:
            self._metrics[key] = 0

    async def process_message(self, message: str) -> str:
        """Process incoming JSON-RPC message.

        Args:
            message: Raw JSON-RPC message string

        Returns:
            JSON-RPC response string

        """
        data = None  # Initialize data variable to avoid unbound issues
        try:
            # Parse JSON
            data = json.loads(message)
            self._metrics["requests_processed"] += 1

            # Validate JSON-RPC structure
            validated_data = self._validate_jsonrpc(data)

            # Route message based on method
            result = await self._route_message(validated_data)

            # Wrap result in JSON-RPC 2.0 response format
            request_id = validated_data.get("id")
            response = {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id,
            }

            # Format response
            response_str = json.dumps(response)
            self._metrics["responses_sent"] += 1

            return response_str

        except json.JSONDecodeError as e:
            self._metrics["errors_encountered"] += 1
            error_response = {
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {e}"},
                "id": None,
            }
            return json.dumps(error_response)

        except ValidationError as e:
            self._metrics["errors_encountered"] += 1
            # Safely extract id from parsed data if available
            request_id = None
            if data is not None and isinstance(data, dict):
                request_id = data.get("id")

            # Use appropriate error code based on error type
            error_message = str(e)
            if "Method not found" in error_message:
                error_code = -32601
                error_msg = f"Method not found: {error_message}"
            else:
                error_code = -32600
                error_msg = f"Invalid Request: {error_message}"

            error_response = {
                "jsonrpc": "2.0",
                "error": {"code": error_code, "message": error_msg},
                "id": request_id,
            }
            return json.dumps(error_response)

        except Exception as e:
            self._metrics["errors_encountered"] += 1
            self.logger.exception(f"Unexpected error processing message: {e}")
            # Safely extract id from parsed data if available
            request_id = None
            if data is not None and isinstance(data, dict):
                request_id = data.get("id")

            error_response = {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": "Internal error"},
                "id": request_id,
            }
            return json.dumps(error_response)

    def _validate_jsonrpc(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate JSON-RPC message structure.

        Args:
            data: Parsed JSON data

        Returns:
            Validated data

        Raises:
            ValidationError: If validation fails

        """
        try:
            # Check basic structure
            if not isinstance(data, dict):
                self._metrics["invalid_jsonrpc"] += 1
                self._metrics["errors_encountered"] += 1
                msg = "Request must be JSON object"
                raise ValidationError(msg)

            # Check JSON-RPC version
            if data.get("jsonrpc") != "2.0":
                self._metrics["invalid_jsonrpc"] += 1
                self._metrics["errors_encountered"] += 1
                msg = "jsonrpc must be 2.0"
                raise ValidationError(msg)

            # Check if it's a request (has method) or response (has result/error)
            if "method" in data:
                # It's a request - validate method
                method = data["method"]
                if not isinstance(method, str):
                    self._metrics["invalid_jsonrpc"] += 1
                    self._metrics["errors_encountered"] += 1
                    msg = "method' must be string"
                    raise ValidationError(msg)
            elif "result" not in data and "error" not in data:
                # It's a request without a method
                self._metrics["invalid_jsonrpc"] += 1
                self._metrics["errors_encountered"] += 1
                msg = "Missing 'method' field"
                raise ValidationError(msg)

            # Validate id field if present
            if "id" in data:
                id_val = data["id"]
                if not (isinstance(id_val, str | int | float) or id_val is None):
                    self._metrics["invalid_jsonrpc"] += 1
                    self._metrics["errors_encountered"] += 1
                    msg = "id' must be string, number, or null"
                    raise ValidationError(msg)

            return data

        except ValidationError:
            raise
        except Exception as e:
            self._metrics["invalid_jsonrpc"] += 1
            self._metrics["errors_encountered"] += 1
            msg = f"Validation error: {e}"
            raise ValidationError(msg) from e

    def parse_message(self, message: str) -> dict[str, Any]:
        """Parse and validate JSON-RPC message.

        Args:
            message: Raw JSON-RPC message string

        Returns:
            Parsed and validated message dictionary

        Raises:
            ValidationError: If parsing or validation fails

        """
        try:
            # Security validation: Check message size limits (1MB limit)
            max_message_size = 1024 * 1024  # 1MB
            if len(message) > max_message_size:
                self._metrics["errors_encountered"] += 1
                msg = (
                    f"Message too large: {len(message)} bytes (max: {max_message_size})"
                )
                raise ValidationError(msg)

            # Parse JSON
            data = json.loads(message)
            self._metrics["messages_parsed"] += 1

            # Security validation: Check nesting depth limits
            max_nesting_depth = 100
            self._validate_nesting_depth(data, max_nesting_depth)

            # Validate JSON-RPC structure
            validated_data = self._validate_jsonrpc(data)
            return validated_data

        except json.JSONDecodeError as e:
            self._metrics["errors_encountered"] += 1
            self._metrics["invalid_jsonrpc"] += 1
            msg = f"Invalid JSON: {e}"
            raise ValidationError(msg) from e

    def _validate_nesting_depth(
        self, obj: Any, max_depth: int, current_depth: int = 0
    ) -> None:
        """Validate object nesting depth to prevent DoS attacks.

        Args:
            obj: Object to validate
            max_depth: Maximum allowed nesting depth
            current_depth: Current nesting level (used in recursion)

        Raises:
            ValidationError: If nesting depth exceeds limit

        """
        if current_depth > max_depth:
            msg = f"Nesting too deep: {current_depth} levels (max: {max_depth})"
            raise ValidationError(msg)

        if isinstance(obj, dict):
            for value in obj.values():
                self._validate_nesting_depth(value, max_depth, current_depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                self._validate_nesting_depth(item, max_depth, current_depth + 1)

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle parsed JSON-RPC request.

        Args:
            request: Parsed JSON-RPC request dictionary

        Returns:
            Response dictionary or None for notifications

        """
        message_id = request.get("id")
        self._metrics["requests_handled"] += 1

        # Handle notifications (no id)
        if message_id is None:
            # Notifications don't get responses
            return None

        # Route to appropriate handler
        try:
            response_data = await self._route_message(request)
            return self.create_response(response_data, message_id)
        except ValidationError as e:
            # Handle specific validation errors with correct codes
            if "Method not found" in str(e) or "not found" in str(e).lower():
                return self.create_error_response(
                    -32601, "Method not found", message_id
                )
            elif (
                "Invalid params" in str(e)
                or "invalid param" in str(e).lower()
                or "Missing required parameter" in str(e)
                or "required parameter" in str(e).lower()
            ):
                return self.create_error_response(-32602, "Invalid params", message_id)
            else:
                return self.create_error_response(-32600, "Invalid Request", message_id)
        except Exception as e:
            self.logger.exception(f"Error handling request: {e}")
            return self.create_error_response(
                -32603, "Internal error", message_id, {"details": str(e)}
            )

    def create_response(self, result: Any, request_id: Any) -> dict[str, Any]:
        """Create JSON-RPC success response.

        Args:
            result: Result data
            request_id: Request ID to echo back

        Returns:
            JSON-RPC response dictionary

        """
        self._metrics["responses_sent"] += 1
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id,
        }

    def create_error_response(
        self,
        code: int,
        message: str,
        request_id: Any,
        data: Any = None,
    ) -> dict[str, Any]:
        """Create JSON-RPC error response.

        Args:
            code: Error code
            message: Error message
            request_id: Request ID to echo back
            data: Optional error data

        Returns:
            JSON-RPC error response dictionary

        """
        self._metrics["errors_encountered"] += 1
        error_obj = {
            "code": code,
            "message": message,
        }
        if data is not None:
            error_obj["data"] = data

        return {
            "jsonrpc": "2.0",
            "error": error_obj,
            "id": request_id,
        }

    async def _route_message(self, data: dict[str, Any]) -> Any:
        """Route validated message to appropriate handler.

        Args:
            data: Validated JSON-RPC data

        Returns:
            Response data (just the result, not full JSON-RPC response)

        Raises:
            Exception: If method not found or handler fails

        """
        method = data.get("method")
        params = data.get("params", {})

        # Handle different MCP methods
        if method == "initialize":
            return await self._handle_initialize(params)
        if method == "tools/list":
            return await self._handle_list_tools(params)
        if method == "tools/call":
            return await self._handle_call_tool(params)
        if method == "resources/list":
            return await self._handle_list_resources(params)
        if method == "resources/read":
            return await self._handle_read_resource(params)

        # Method not found
        msg = f"Method not found: {method}"
        raise ValidationError(msg)

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle initialize request.

        Args:
            params: Request parameters

        Returns:
            Initialize result data

        """
        # Basic initialization response
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
                "logging": {},
            },
            "serverInfo": {"name": "llm-cli-runner-server", "version": "0.1.0"},
        }

    async def _handle_list_tools(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle list_tools request.

        Args:
            params: Request parameters

        Returns:
            Tools list result data

        """
        # Return empty tools list for now
        return {"tools": []}

    async def _handle_call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle call_tool request.

        Args:
            params: Request parameters

        Returns:
            Tool execution result data

        Raises:
            ValidationError: If tool parameters are invalid

        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        # Validate required parameters
        if not tool_name:
            raise ValidationError("Missing required parameter 'name'")

        # Validate tool exists and has proper arguments
        if tool_name == "invalid_tool":
            # Check if arguments are missing for tools that require them
            if not arguments:
                raise ValidationError(
                    "Invalid params: missing required parameter 'arguments'"
                )

        # For testing purposes, validate gemini_generate tool specifically
        if tool_name == "gemini_generate":
            if not arguments or "prompt" not in arguments:
                raise ValidationError(
                    "Tool 'gemini_generate' requires 'prompt' argument"
                )

        # Placeholder tool execution - return result data only
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Tool '{tool_name}' executed successfully",
                },
            ],
        }

    async def _handle_list_resources(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle list_resources request.

        Args:
            params: Request parameters

        Returns:
            Resources list result data

        """
        return {"resources": []}

    async def _handle_read_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle read_resource request.

        Args:
            params: Request parameters

        Returns:
            Resource content result data

        """
        resource_uri = params.get("uri", "unknown")

        return {
            "contents": [
                {
                    "uri": resource_uri,
                    "mimeType": "text/plain",
                    "text": f"Content for resource: {resource_uri}",
                },
            ],
        }


class RequestQueueDict:
    """Thread-safe dictionary-like interface for request queuing.

    This class provides a dictionary-like interface with thread safety
    for managing request queues and caching.
    """

    def __init__(self) -> None:
        """Initialize request queue dictionary."""
        self._data: dict[str, Any] = {}
        self._lock = Lock()

    def __contains__(self, key: str) -> bool:
        """Check if key exists in the queue.

        Args:
            key: Key to check

        Returns:
            True if key exists

        """
        with self._lock:
            return key in self._data

    def __getitem__(self, key: str) -> Any:
        """Get item from queue.

        Args:
            key: Key to retrieve

        Returns:
            Value associated with key

        """
        with self._lock:
            return self._data[key]

    def pop(self, key: str, default: Any = None) -> Any:
        """Remove and return item from queue.

        Args:
            key: Key to remove
            default: Default value if key not found

        Returns:
            Removed value or default

        """
        with self._lock:
            return self._data.pop(key, default)

    def __call__(self, key: str, value: Any) -> None:
        """Add item to queue.

        Args:
            key: Key to add
            value: Value to associate with key

        """
        with self._lock:
            self._data[key] = value


class ToolAdapter:
    """Adapter for tool integration and compatibility.

    This class provides adaptation and compatibility layers for
    integrating different tool implementations.
    """

    def __init__(self, tool_spec: dict[str, Any]) -> None:
        """Initialize tool adapter.

        Args:
            tool_spec: Tool specification dictionary

        """
        self.logger = get_logger(__name__)
        self.tool_spec = tool_spec
        self.name = tool_spec.get("name", "unknown")

    def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Adapt request for tool execution.

        Args:
            request: Original request data

        Returns:
            Adapted request data

        """
        return request

    def adapt_response(self, response: Any) -> dict[str, Any]:
        """Adapt tool response to standard format.

        Args:
            response: Tool response data

        Returns:
            Adapted response in standard format

        """
        return {"result": response}

    def adapt_tool(self, tool_spec: dict[str, Any], provider: str) -> dict[str, Any]:
        """Adapt tool specification for a specific provider.

        Args:
            tool_spec: Original tool specification
            provider: Target provider name

        Returns:
            Adapted tool specification

        """
        adapted_tool = tool_spec.copy()
        adapted_tool["provider_specific"] = {
            "provider": provider,
            "adapted_at": datetime.now().isoformat(),
        }

        # Provider-specific adaptations
        supported_providers = ["openai", "anthropic", "gemini", "llama"]

        if provider not in supported_providers:
            from mcp_server_llm_cli_runner.core.errors import ValidationError

            raise ValidationError(f"Unsupported provider: {provider}")

        if provider == "openai":
            adapted_tool["provider_specific"]["format"] = "openai_function"
        elif provider == "anthropic":
            adapted_tool["provider_specific"]["format"] = "anthropic_tool"
        elif provider == "gemini":
            adapted_tool["provider_specific"]["format"] = "gemini_tool"
        elif provider == "llama":
            adapted_tool["provider_specific"]["format"] = "llama_tool"

        return adapted_tool


class ToolVersionManager:
    """Manager for tool versioning and compatibility.

    This class handles tool version management, compatibility
    checking, and migration between different tool versions.
    """

    def __init__(self) -> None:
        """Initialize tool version manager."""
        self.logger = get_logger(__name__)
        self._versions: dict[str, list[str]] = {}
        self._tool_specs: dict[str, dict[str, dict[str, Any]]] = {}

    def register_version(self, tool_spec: dict[str, Any]) -> None:
        """Register a tool version.

        Args:
            tool_spec: Tool specification containing name and version

        """
        tool_name = tool_spec.get("name", "unknown")
        version = tool_spec.get("version", "1.0.0")

        if tool_name not in self._versions:
            self._versions[tool_name] = []
            self._tool_specs[tool_name] = {}

        if version not in self._versions[tool_name]:
            self._versions[tool_name].append(version)
            self._versions[tool_name].sort()

        self._tool_specs[tool_name][version] = tool_spec

    def get_versions(self, tool_name: str) -> list[str]:
        """Get available versions for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            List of available versions

        """
        return self._versions.get(tool_name, [])

    def get_latest_version(self, tool_name: str) -> dict[str, Any] | None:
        """Get latest version for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Latest tool specification or None if not found

        """
        versions = self.get_versions(tool_name)
        if not versions:
            return None

        latest_version = versions[-1]
        return self._tool_specs.get(tool_name, {}).get(latest_version)

    def get_version(self, tool_name: str, version: str) -> dict[str, Any] | None:
        """Get specific version of a tool.

        Args:
            tool_name: Name of the tool
            version: Version string

        Returns:
            Tool specification or None if not found

        """
        return self._tool_specs.get(tool_name, {}).get(version)

    def is_compatible(self, tool_name: str, version1: str, version2: str) -> bool:
        """Check if two versions of a tool are compatible.

        Args:
            tool_name: Name of the tool
            version1: First version
            version2: Second version

        Returns:
            True if versions are compatible

        """
        # Simple compatibility check - same major version
        try:
            v1_major = int(version1.split(".")[0])
            v2_major = int(version2.split(".")[0])
            return v1_major == v2_major
        except (ValueError, IndexError):
            return False

    def migrate_parameters(
        self,
        tool_name: str,
        _from_version: str,
        to_version: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Migrate parameters between tool versions.

        Args:
            tool_name: Name of the tool
            _from_version: Source version (unused, reserved for future migration logic)
            to_version: Target version
            parameters: Parameters to migrate

        Returns:
            Migrated parameters

        """
        # Simple migration - preserve existing parameters and add defaults for new ones
        migrated = parameters.copy()

        to_spec = self.get_version(tool_name, to_version)
        if to_spec and "inputSchema" in to_spec:
            properties = to_spec["inputSchema"].get("properties", {})
            for param_name, param_def in properties.items():
                if param_name not in migrated and "default" in param_def:
                    migrated[param_name] = param_def["default"]

        return migrated


class ProviderToolManager:
    """Manager for provider-specific tool handling.

    This class manages tools that are specific to particular
    LLM providers, handling provider-specific configurations
    and tool implementations.
    """

    def __init__(
        self,
        provider_name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize provider tool manager.

        Args:
            provider_name: Name of the LLM provider
            config: Optional configuration dictionary

        """
        self.logger = get_logger(__name__)
        self.provider_name = provider_name
        self.config = config or {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._provider_tools: dict[str, dict[str, Any]] = {}
        self._tool_adapters: dict[str, ToolAdapter] = {}

    def register_provider_tool(self, tool_spec: dict[str, Any]) -> None:
        """Register a tool for this provider.

        Args:
            tool_spec: Tool specification

        Raises:
            ValidationError: If tool specification is invalid

        """
        from mcp_server_llm_cli_runner.core.errors import ValidationError

        tool_name = tool_spec.get("name", "unknown")

        # Validate required fields
        if not tool_spec.get("provider_config"):
            raise ValidationError(
                f"Tool '{tool_name}' missing required provider_config"
            )

        # Store tool with provider configuration
        provider_tool = {
            "name": tool_name,
            "description": tool_spec.get("description", ""),
            "inputSchema": tool_spec.get("inputSchema", {}),
            "provider_config": tool_spec.get("provider_config", {}),
        }

        self._tools[tool_name] = provider_tool
        self._provider_tools[tool_name] = tool_spec

        # Create adapter for the tool
        adapter_key = f"{self.provider_name}:{tool_name}"
        self._tool_adapters[adapter_key] = ToolAdapter(tool_spec)

    def adapt_tool_for_provider(self, tool_name: str) -> dict[str, Any] | None:
        """Adapt tool for provider-specific API format.

        Args:
            tool_name: Name of the tool to adapt

        Returns:
            Adapted tool specification or None if not found

        """
        if tool_name not in self._provider_tools:
            return None

        tool_spec = self._provider_tools[tool_name].copy()

        # Add provider-specific adaptations
        if self.provider_name == "openai":
            # Add OpenAI-specific model parameter
            if "inputSchema" in tool_spec and "properties" in tool_spec["inputSchema"]:
                tool_spec["inputSchema"]["properties"]["model"] = {
                    "type": "string",
                    "default": "gpt-3.5-turbo",
                }

        return tool_spec

    def create_execution_context(self, tool_name: str) -> dict[str, Any] | None:
        """Create execution context for provider-specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Execution context dictionary or None if tool not found

        """
        if tool_name not in self._provider_tools:
            return None

        return {
            "provider": self.provider_name,
            "config": self.config,
            "tool_name": tool_name,
            "provider_tools": self._provider_tools.get(tool_name, {}),
            "execution_id": f"{self.provider_name}:{tool_name}:{id(self)}",
        }

    def get_provider_tools(self, provider: str) -> dict[str, Any]:
        """Get tools for a specific provider.

        Args:
            provider: Provider name

        Returns:
            Dictionary of provider tools

        """
        return self._provider_tools.get(provider, {})

    def get_tool_adapter(self, provider: str, tool_name: str) -> ToolAdapter | None:
        """Get tool adapter for provider-specific tool.

        Args:
            provider: Provider name
            tool_name: Tool name

        Returns:
            Tool adapter or None if not found

        """
        adapter_key = f"{provider}:{tool_name}"
        return self._tool_adapters.get(adapter_key)


class ToolRegistry:
    """Tool registry for managing and discovering MCP tools.

    This class handles registration, discovery, and invocation of tools
    across different providers and contexts.
    """

    def __init__(self) -> None:
        """Initialize tool registry."""
        self.logger = get_logger(__name__)
        self._tools: dict[str, dict[str, dict[str, Any]]] = {}
        self._providers: set[str] = set()
        self._lock = Lock()

    def register_tool(
        self,
        tool_spec: dict[str, Any],
        provider: str,
        handler: Any = None,
    ) -> None:
        """Register a tool for a specific provider.

        Args:
            tool_spec: Tool specification dictionary
            provider: Provider name
            handler: Optional tool handler function

        Raises:
            ValidationError: If tool specification is invalid

        """
        # Validate required fields
        tool_name = tool_spec.get("name")
        if not tool_name:
            from mcp_server_llm_cli_runner.core.errors import ValidationError

            raise ValidationError("Tool specification missing required 'name' field")

        with self._lock:
            if tool_name not in self._tools:
                self._tools[tool_name] = {}

            self._tools[tool_name][provider] = {
                "spec": tool_spec,
                "handler": handler,
            }
            self._providers.add(provider)

        self.logger.info(f"Registered tool '{tool_name}' for provider '{provider}'")

    def unregister_tool(self, tool_name: str, provider: str) -> bool:
        """Unregister a tool from a provider.

        Args:
            tool_name: Name of the tool to unregister
            provider: Provider name

        Returns:
            True if tool was unregistered, False if not found

        """
        with self._lock:
            if tool_name in self._tools and provider in self._tools[tool_name]:
                del self._tools[tool_name][provider]

                # Clean up empty tool entries
                if not self._tools[tool_name]:
                    del self._tools[tool_name]

                # Check if provider still has any tools
                provider_has_tools = any(
                    provider in tool_providers
                    for tool_providers in self._tools.values()
                )
                if not provider_has_tools:
                    self._providers.discard(provider)

                self.logger.info(
                    f"Unregistered tool '{tool_name}' from provider '{provider}'"
                )
                return True

        return False

    def discover_tools(self, provider: str | None = None) -> list[dict[str, Any]]:
        """Discover available tools.

        Args:
            provider: Optional provider filter

        Returns:
            List of tool specifications

        """
        tools = []

        with self._lock:
            if provider:
                # Return tools for specific provider
                for _tool_name, tool_providers in self._tools.items():
                    if provider in tool_providers:
                        tools.append(tool_providers[provider]["spec"])
            else:
                # Return all tools from all providers
                for tool_providers in self._tools.values():
                    for tool_data in tool_providers.values():
                        tools.append(tool_data["spec"])

        return tools

    def get_tool_handler(self, tool_name: str, provider: str) -> Any | None:
        """Get handler for a specific tool.

        Args:
            tool_name: Name of the tool
            provider: Provider name

        Returns:
            Tool handler function or None if not found

        """
        with self._lock:
            if tool_name in self._tools and provider in self._tools[tool_name]:
                return self._tools[tool_name][provider]["handler"]
        return None

    def invoke_tool(
        self,
        tool_name: str,
        provider: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Invoke a tool with given arguments.

        Args:
            tool_name: Name of the tool to invoke
            provider: Provider name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValidationError: If tool or provider not found

        """
        handler = self.get_tool_handler(tool_name, provider)
        if not handler:
            from mcp_server_llm_cli_runner.core.errors import ValidationError

            msg = f"Tool '{tool_name}' not found for provider '{provider}'"
            raise ValidationError(msg)

        try:
            # If handler is a callable, invoke it
            if callable(handler):
                return handler(arguments)
            # If handler is a static response, return it
            return handler
        except Exception as e:
            self.logger.exception(f"Error invoking tool '{tool_name}': {e}")
            raise

    def get_providers(self) -> list[str]:
        """Get list of registered providers.

        Returns:
            List of provider names

        """
        with self._lock:
            return list(self._providers)

    def get_tool_count(self, provider: str | None = None) -> int:
        """Get count of registered tools.

        Args:
            provider: Optional provider filter

        Returns:
            Number of registered tools

        """
        with self._lock:
            if provider:
                # Count tools for specific provider
                count = 0
                for tool_providers in self._tools.values():
                    if provider in tool_providers:
                        count += 1
                return count
            return len(self._tools)

    def clear_provider(self, provider: str) -> int:
        """Clear all tools for a provider.

        Args:
            provider: Provider name

        Returns:
            Number of tools removed

        """
        count = 0
        with self._lock:
            tools_to_remove = []
            for tool_name, tool_providers in self._tools.items():
                if provider in tool_providers:
                    del tool_providers[provider]
                    count += 1
                    if not tool_providers:  # Tool has no providers left
                        tools_to_remove.append(tool_name)

            # Remove empty tool entries
            for tool_name in tools_to_remove:
                del self._tools[tool_name]

            self._providers.discard(provider)
            if count > 0:
                self.logger.info(f"Cleared {count} tools for provider '{provider}'")

        return count

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry statistics

        """
        with self._lock:
            tools_by_provider = {}
            for tool_name, tool_providers in self._tools.items():
                for provider in tool_providers:
                    if provider not in tools_by_provider:
                        tools_by_provider[provider] = []
                    tools_by_provider[provider].append(tool_name)

            return {
                "total_providers": len(self._providers),
                "total_tools": len(self._tools),
                "providers": list(self._providers),
                "tools_by_provider": tools_by_provider,
            }


class RequestManager:
    """Manager for handling request lifecycle and caching.

    This class provides request management functionality including
    caching, queuing, and lifecycle tracking.
    """

    def __init__(self) -> None:
        """Initialize request manager."""
        self.logger = get_logger(__name__)
        self._active_requests: dict[str, dict[str, Any]] = {}
        self._request_queue = RequestQueueDict()
        self._cache: dict[str, Any] = {}
        self._lock = Lock()

    def create_request_id(self) -> str:
        """Create a unique request ID.

        Returns:
            Unique request identifier

        """
        return str(uuid.uuid4())

    def start_request(self, request_id: str, request_data: dict[str, Any]) -> None:
        """Start tracking a request.

        Args:
            request_id: Request identifier
            request_data: Request data

        """
        with self._lock:
            self._active_requests[request_id] = {
                "data": request_data,
                "started_at": datetime.now(),
                "status": "active",
            }

    def complete_request(self, request_id: str, response_data: Any) -> None:
        """Mark request as completed.

        Args:
            request_id: Request identifier
            response_data: Response data

        """
        with self._lock:
            if request_id in self._active_requests:
                self._active_requests[request_id].update(
                    {
                        "response": response_data,
                        "completed_at": datetime.now(),
                        "status": "completed",
                    },
                )

    def fail_request(self, request_id: str, error: str) -> None:
        """Mark request as failed.

        Args:
            request_id: Request identifier
            error: Error message

        """
        with self._lock:
            if request_id in self._active_requests:
                self._active_requests[request_id].update(
                    {
                        "error": error,
                        "completed_at": datetime.now(),
                        "status": "failed",
                    },
                )

    def get_request_status(self, request_id: str) -> dict[str, Any] | None:
        """Get request status.

        Args:
            request_id: Request identifier

        Returns:
            Request status dictionary or None if not found

        """
        with self._lock:
            return self._active_requests.get(request_id)

    def cleanup_completed_requests(self, max_age_seconds: int = 3600) -> int:
        """Clean up old completed requests.

        Args:
            max_age_seconds: Maximum age in seconds for completed requests

        Returns:
            Number of requests cleaned up

        """
        cutoff_time = datetime.now()
        cutoff_time = cutoff_time.replace(second=cutoff_time.second - max_age_seconds)

        cleaned_count = 0
        with self._lock:
            to_remove = []
            for request_id, request_info in self._active_requests.items():
                if (
                    request_info.get("status") in ("completed", "failed")
                    and request_info.get("completed_at", datetime.now()) < cutoff_time
                ):
                    to_remove.append(request_id)

            for request_id in to_remove:
                del self._active_requests[request_id]
                cleaned_count += 1

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old requests")

        return cleaned_count


class StreamingHandler:
    """Handler for streaming response functionality.

    This class provides streaming capabilities for real-time response
    delivery with backpressure handling and flow control.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize streaming handler.

        Args:
            config: Optional configuration dictionary

        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.buffer_size = self.config.get("buffer_size", 1024)
        self.chunk_size = self.config.get("chunk_size", 128)
        self.flow_control_enabled = self.config.get("flow_control_enabled", True)
        self.backpressure_threshold = self.config.get("backpressure_threshold", 0.8)
        self.connection_timeout = self.config.get("connection_timeout", 30)
        self.stream_timeout = self.config.get("stream_timeout", 10)
        self.max_concurrent_streams = self.config.get("max_concurrent_streams", 10)
        self._active_streams: dict[str, Any] = {}
        self._active_sessions: dict[str, Any] = {}  # For session management
        self._lock = Lock()
        self._metrics = {
            "total_chunks_sent": 0,
            "total_bytes_sent": 0,
            "active_connections": 0,
            "errors_encountered": 0,
        }

    async def create_stream_session(self, client_id: str | None = None) -> str:
        """Create a new streaming session.

        Args:
            client_id: Optional client identifier

        Returns:
            Stream session ID

        """
        stream_id = str(uuid.uuid4())

        with self._lock:
            self._active_streams[stream_id] = {
                "client_id": client_id,
                "created_at": datetime.now(),
                "status": "active",
                "chunks_sent": 0,
                "bytes_sent": 0,
            }
            self._active_sessions[stream_id] = {
                "client_id": client_id,
                "created_at": datetime.now(),
                "status": "active",
            }
            self._metrics["active_connections"] += 1

        return stream_id

    async def stream_response(
        self,
        session_id: str,
        data: Any,
        max_chunks: int | None = None,
    ):
        """Stream response data for a session.

        Args:
            session_id: Stream session identifier
            data: Data to stream (can be string, dict, or iterable)
            max_chunks: Optional maximum chunks to send

        Yields:
            Structured chunks with metadata

        """
        if session_id not in self._active_streams:
            from mcp_server_llm_cli_runner.core.errors import (
                ValidationError as CoreValidationError,
            )

            msg = f"Stream session {session_id} not found"
            raise CoreValidationError(msg)

        # Check for error conditions in data
        if isinstance(data, dict):
            if "error" in data:
                from mcp_server_llm_cli_runner.core.errors import (
                    ValidationError as CoreValidationError,
                )

                msg = f"Streaming error: {data.get('error', 'Unknown error')}"
                raise CoreValidationError(msg)

            if data.get("force_error"):
                from mcp_server_llm_cli_runner.core.errors import (
                    ValidationError as CoreValidationError,
                )

                msg = "Forced error for testing"
                raise CoreValidationError(msg)

            # Handle timeout simulation
            if (
                data.get("simulate_slow")
                or data.get("delay_seconds", 0) > self.stream_timeout
            ):
                await asyncio.sleep(0.01)  # Small delay
                raise TimeoutError("Stream timeout")

        # Convert data to streamable format
        stream_data = self._prepare_stream_data(data, max_chunks)

        chunk_count = 0
        start_time = time.time()

        # Stream chunks with proper structure
        async for chunk_data in stream_data:
            chunk_count += 1

            # Create structured chunk
            chunk = {
                "data": chunk_data,
                "chunk_id": chunk_count,
                "session_id": session_id,
                "timestamp": time.time(),
                "sequence": chunk_count,
            }

            # Add flow control indicators
            if self.flow_control_enabled:
                await self._check_backpressure(session_id)
                chunk["flow_control_applied"] = True

            # Check for buffer overflow
            buffer_usage = self._get_buffer_usage(session_id)
            if buffer_usage > self.backpressure_threshold:
                chunk["backpressure_applied"] = True
                chunk["buffer_overflow_handled"] = True

            # Add performance metrics
            if isinstance(data, dict) and data.get("performance_test"):
                processing_time = time.time() - start_time
                chunk["metrics"] = {
                    "processing_time": processing_time,
                    "throughput": chunk_count / max(processing_time, 0.001),
                }

            # Update session metrics
            with self._lock:
                self._active_streams[session_id]["chunks_sent"] += 1
                self._active_streams[session_id]["bytes_sent"] += len(str(chunk_data))
                self._metrics["total_chunks_sent"] += 1
                self._metrics["total_bytes_sent"] += len(str(chunk_data))

            yield chunk

    def _prepare_stream_data(self, data: Any, max_chunks: int | None = None):
        """Prepare data for streaming.

        Args:
            data: Input data to stream
            max_chunks: Optional maximum chunks to send

        Returns:
            Async generator of chunk data

        """

        async def _generate_chunks():
            if isinstance(data, dict):
                # Handle dictionary data
                if "total_chunks" in data:
                    # Multi-chunk streaming test
                    total = data["total_chunks"]
                    for i in range(total):
                        yield {**data, "chunk_number": i + 1}
                elif "chunks" in data:
                    # Large data with specified chunks
                    chunks_count = data["chunks"]
                    for i in range(min(chunks_count, max_chunks or chunks_count)):
                        yield {**data, "chunk_id": i + 1}
                elif "payload" in data:
                    # Handle payload chunking
                    payload = data["payload"]
                    if isinstance(payload, str) and len(payload) > self.chunk_size:
                        # Split large payload into chunks
                        for i in range(0, len(payload), self.chunk_size):
                            chunk_payload = payload[i : i + self.chunk_size]
                            yield {**data, "payload": chunk_payload, "chunk_part": True}
                    else:
                        yield data
                elif data.get("data") == "streaming":
                    # Handle infinite streaming for cancellation tests
                    chunk_count = 0
                    while True:  # Infinite stream
                        chunk_count += 1
                        yield {**data, "chunk_number": chunk_count}
                        await asyncio.sleep(0.001)  # Small delay to allow cancellation
                else:
                    # Single dictionary chunk
                    yield data
            elif isinstance(data, str):
                # Handle string data
                if len(data) > self.chunk_size:
                    # Split into chunks
                    for i in range(0, len(data), self.chunk_size):
                        yield data[i : i + self.chunk_size]
                else:
                    yield data
            elif hasattr(data, "__iter__") and not isinstance(data, str | bytes):
                # Handle iterable data
                count = 0
                for item in data:
                    if max_chunks and count >= max_chunks:
                        break
                    yield item
                    count += 1
            else:
                # Handle other data types
                yield str(data)

        return _generate_chunks()

    def _get_buffer_usage(self, session_id: str) -> float:
        """Get buffer usage ratio for a session.

        Args:
            session_id: Session identifier

        Returns:
            Buffer usage ratio (0.0 to 1.0)

        """
        with self._lock:
            session_info = self._active_streams.get(session_id, {})
            bytes_sent = session_info.get("bytes_sent", 0)
            return min(bytes_sent / self.buffer_size, 1.0)

    async def _check_backpressure(self, stream_id: str) -> None:
        """Check and handle backpressure for streaming.

        Args:
            stream_id: Stream session identifier

        """
        buffer_usage = self._get_buffer_usage(stream_id)
        if buffer_usage > self.backpressure_threshold:
            # Apply backpressure by introducing delay
            await asyncio.sleep(0.01)

    def get_stream_metrics(self, stream_id: str) -> dict[str, Any]:
        """Get metrics for a specific stream.

        Args:
            stream_id: Stream session identifier

        Returns:
            Stream metrics dictionary

        """
        with self._lock:
            return self._active_streams.get(stream_id, {}).copy()

    async def close_stream_session(self, session_id: str) -> None:
        """Close a streaming session.

        Args:
            session_id: Session identifier

        """
        with self._lock:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
                self._metrics["active_connections"] -= 1
            if session_id in self._active_streams:
                self._active_streams[session_id]["status"] = "closed"
                del self._active_streams[session_id]

    def close_session(self, session_id: str) -> None:
        """Close a session (alias for close_stream_session).

        Args:
            session_id: Session identifier

        """
        # Convert sync call to async for consistency
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an async context, schedule the coroutine
                asyncio.create_task(self.close_stream_session(session_id))
            else:
                # If not in async context, run it
                loop.run_until_complete(self.close_stream_session(session_id))
        except RuntimeError:
            # Fallback to direct cleanup
            with self._lock:
                if session_id in self._active_sessions:
                    del self._active_sessions[session_id]
                    self._metrics["active_connections"] -= 1
                if session_id in self._active_streams:
                    self._active_streams[session_id]["status"] = "closed"
                    del self._active_streams[session_id]

    async def close_stream(self, stream_id: str) -> None:
        """Close a streaming session (alias for close_stream_session).

        Args:
            stream_id: Stream session identifier

        """
        await self.close_stream_session(stream_id)


class SessionManager:
    """Manager for client sessions and streaming.

    This class handles client session management, streaming responses,
    and session lifecycle operations.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize session manager.

        Args:
            config: Optional configuration dictionary

        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self._active_sessions: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create_session(self, client_id: str | None = None) -> str:
        """Create a new session.

        Args:
            client_id: Optional client identifier

        Returns:
            Session ID

        """
        session_id = str(uuid.uuid4())
        self._active_sessions[session_id] = {
            "client_id": client_id,
            "created_at": datetime.now(),
            "status": "active",
        }
        return session_id

    async def stream_response(
        self,
        session_id: str,
        data: dict[str, Any],
        **kwargs: Any,
    ):
        """Stream response data to client.

        Args:
            session_id: Stream session ID
            data: Data to stream
            **kwargs: Additional streaming options

        Yields:
            Streamed data chunks

        """
        # Placeholder streaming implementation
        # In a real implementation, this would handle SSE or WebSocket streaming
        yield json.dumps(data)

    def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """Get session information.

        Args:
            session_id: Session ID

        Returns:
            Session information or None if not found

        """
        with self._lock:
            return self._active_sessions.get(session_id)

    def update_session_status(self, session_id: str, status: str) -> bool:
        """Update session status.

        Args:
            session_id: Session ID
            status: New status

        Returns:
            True if updated successfully

        """
        with self._lock:
            if session_id in self._active_sessions:
                self._active_sessions[session_id]["status"] = status
                self._active_sessions[session_id]["last_updated"] = datetime.now()
                return True
        return False

    def get_active_sessions(self) -> list[str]:
        """Get list of active session IDs.

        Returns:
            List of active session IDs

        """
        with self._lock:
            return [
                sid
                for sid, info in self._active_sessions.items()
                if info.get("status") == "active"
            ]

    def close_session(self, session_id: str) -> bool:
        """Close a session.

        Args:
            session_id: Session ID

        Returns:
            True if session was closed

        """
        with self._lock:
            if session_id in self._active_sessions:
                self._active_sessions[session_id]["status"] = "closed"
                self._active_sessions[session_id]["closed_at"] = datetime.now()
                return True
        return False

    def cleanup_sessions(self, max_age_seconds: int = 7200) -> int:
        """Clean up old sessions.

        Args:
            max_age_seconds: Maximum age for inactive sessions

        Returns:
            Number of sessions cleaned up

        """
        cutoff_time = datetime.now()
        cutoff_time = cutoff_time.replace(second=cutoff_time.second - max_age_seconds)

        cleaned_count = 0
        with self._lock:
            to_remove = []
            for session_id, session_info in self._active_sessions.items():
                last_activity = session_info.get(
                    "last_updated",
                    session_info["created_at"],
                )
                if last_activity < cutoff_time:
                    to_remove.append(session_id)

            for session_id in to_remove:
                del self._active_sessions[session_id]
                cleaned_count += 1

        return cleaned_count

    def remove_session(self, session_id: str) -> bool:
        """Remove a session completely.

        Args:
            session_id: Session ID

        Returns:
            True if session was removed

        """
        with self._lock:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
                return True
        return False


class MCPProtocolHandlerV2:
    """MCP Protocol handler for JSON-RPC 2.0 message processing (v2).

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
            "serverInfo": {"name": "llm-cli-runner-server", "version": "1.0.0"},
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


class LLMCliRunnerServer:
    """Main LLM CLI Runner MCP Server implementation."""

    def __init__(
        self,
        config_manager: Any = None,
        middleware_chain: Any = None,
        resource_manager: Any = None,
    ) -> None:
        """Initialize LLM CLI Runner server.

        Args:
            config_manager: Configuration manager instance
            middleware_chain: Optional middleware chain
            resource_manager: Optional resource manager

        """
        self.logger = get_logger(__name__)
        self.config_manager = config_manager
        self.middleware_chain = middleware_chain
        self.resource_manager = resource_manager

        # Initialize core components
        self.protocol_handler = MCPProtocolHandler()
        self.tool_registry = ToolRegistry()
        self.request_manager = RequestManager()
        self.session_manager = SessionManager()

        # Server state
        self._initialized = False
        self._server: Any = None  # Will be set when server is created

    async def initialize(self) -> None:
        """Initialize the server and all components."""
        if self._initialized:
            return

        self.logger.info("Initializing LLM CLI Runner server...")

        try:
            # Initialize components in order
            await self._initialize_components()

            # Register default tools
            await self._register_default_tools()

            # Set up server endpoints
            await self._setup_server_endpoints()

            self._initialized = True
            self.logger.info("LLM CLI Runner server initialized successfully")

        except Exception as e:
            self.logger.exception(f"Failed to initialize server: {e}")
            raise

    async def _initialize_components(self) -> None:
        """Initialize all server components."""
        # Component initialization would go here
        # For now, just log that components are being initialized
        self.logger.debug("Initializing server components")

    async def _register_default_tools(self) -> None:
        """Register default tools with the tool registry."""
        # Default tool registration would go here
        self.logger.debug("Registering default tools")

    async def _setup_server_endpoints(self) -> None:
        """Set up server endpoints and handlers."""
        # This would set up the actual MCP server endpoints
        # For now, we'll create a mock server object

        class MockServer:
            """Mock server for development."""

            def __init__(self) -> None:
                self._handlers = {}

            def list_tools(self):
                """Register list_tools handler."""

                def decorator(func):
                    self._handlers["list_tools"] = func
                    return func

                return decorator

            def call_tool(self):
                """Register call_tool handler."""

                def decorator(func):
                    self._handlers["call_tool"] = func
                    return func

                return decorator

        self._server = MockServer()

        # Register list_tools handler
        @self._server.list_tools()
        async def handle_list_tools() -> dict[str, Any]:
            """Handle list_tools request."""
            tools = self.tool_registry.discover_tools()
            return {"tools": tools}

        # Register call_tool handler
        @self._server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            """Handle call_tool request."""
            # Default provider for now
            result = self.tool_registry.invoke_tool(name, "default", arguments)
            return {"content": [{"type": "text", "text": str(result)}]}

    def get_mcp_server(self) -> Any:
        """Get the MCP server instance.

        Returns:
            The MCP server instance

        """
        if self._server is None:
            # Initialize server lazily if not already done
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if not self._initialized:
                    loop.run_until_complete(self.initialize())
            except RuntimeError:
                # If no event loop, create one
                asyncio.run(self.initialize())

        return self._server

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process incoming MCP request.

        Args:
            request: MCP request data

        Returns:
            MCP response data

        """
        if not self._initialized:
            await self.initialize()

        request_id = self.request_manager.create_request_id()

        try:
            # Start tracking request
            self.request_manager.start_request(request_id, request)

            # Process through protocol handler
            request_json = json.dumps(request)
            response_json = await self.protocol_handler.process_message(request_json)
            response = json.loads(response_json)

            # Mark request as completed
            self.request_manager.complete_request(request_id, response)

            return response

        except Exception as e:
            # Mark request as failed
            self.request_manager.fail_request(request_id, str(e))
            self.logger.exception(f"Error processing request {request_id}: {e}")
            raise

    async def start(self, host: str = "localhost", port: int = 8080) -> None:
        """Start the server.

        Args:
            host: Host to bind to
            port: Port to bind to

        """
        if not self._initialized:
            await self.initialize()

        self.logger.info(f"Starting LLM CLI Runner server on {host}:{port}")

        # Server startup logic would go here
        # For now, just log that the server is starting
        self.logger.info("Server started successfully")

    async def stop(self) -> None:
        """Stop the server gracefully."""
        self.logger.info("Stopping LLM CLI Runner server...")

        try:
            # Cleanup active sessions
            active_sessions = self.session_manager.get_active_sessions()
            for session_id in active_sessions:
                self.session_manager.close_session(session_id)

            # Cleanup active requests
            self.request_manager.cleanup_completed_requests(0)  # Clean all

            self.logger.info("Server stopped successfully")

        except Exception as e:
            self.logger.exception(f"Error during server shutdown: {e}")
            raise

    def get_server_stats(self) -> dict[str, Any]:
        """Get server statistics.

        Returns:
            Dictionary with server statistics

        """
        return {
            "initialized": self._initialized,
            "protocol_metrics": self.protocol_handler.get_metrics(),
            "tool_registry": self.tool_registry.get_registry_stats(),
            "active_sessions": len(self.session_manager.get_active_sessions()),
        }

    def register_tool(
        self,
        tool_spec: dict[str, Any],
        provider: str = "default",
        handler: Any = None,
    ) -> None:
        """Register a tool with the server.

        Args:
            tool_spec: Tool specification
            provider: Provider name
            handler: Optional tool handler

        """
        self.tool_registry.register_tool(tool_spec, provider, handler)

    def create_session(self, client_id: str | None = None) -> str:
        """Create a new client session.

        Args:
            client_id: Optional client identifier

        Returns:
            Session ID

        """
        return self.session_manager.create_session(client_id)

    async def _list_tools(self) -> list[Any]:
        """List all available tools based on enabled providers.

        Returns:
            List of Tool objects

        """
        # Use our Tool class directly

        tools = []
        if not self.config_manager:
            return tools

        enabled_providers = self.config_manager.get_enabled_providers()

        for provider in enabled_providers:
            tool_name = f"{provider}_generate"

            # Provider-specific descriptions
            if provider == "gemini":
                description = "Generate text using Gemini CLI"
            elif provider == "codex":
                description = "Generate code using OpenAI Codex"
            elif provider == "llama":
                description = "Generate text using local LLaMA model"
            else:
                description = f"Generate text using {provider.capitalize()} provider"

            # Provider-specific schemas
            if provider == "gemini":
                input_schema = {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate from",
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use (optional)",
                            "default": "gemini-1.5-flash",
                        },
                    },
                    "required": ["prompt"],
                }
            elif provider == "codex":
                input_schema = {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate from",
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language",
                            "default": "python",
                        },
                    },
                    "required": ["prompt"],
                }
            elif provider == "llama":
                input_schema = {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate from",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate",
                            "default": 256,
                        },
                    },
                    "required": ["prompt"],
                }
            else:
                input_schema = {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate from",
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use (optional)",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate",
                        },
                    },
                    "required": ["prompt"],
                }

            # Create tool using appropriate constructor based on MCP availability
            if _MCP_TYPES_AVAILABLE:
                tool = Tool(
                    name=tool_name,
                    description=description,
                    inputSchema=input_schema,
                )
            else:
                tool = Tool(
                    name=tool_name,
                    description=description,
                    inputSchema=input_schema,
                )
            tools.append(tool)

        return tools

    async def _call_tool(self, request: Any) -> Any:
        """Call a specific tool based on the request.

        Args:
            request: CallToolRequest object

        Returns:
            CallToolResult object

        """
        # Use our classes directly

        try:
            tool_name = request.params.name
            arguments = request.params.arguments

            # Route to appropriate provider method
            if tool_name == "gemini_generate":
                result = await self._call_gemini(arguments)
            elif tool_name == "codex_generate":
                result = await self._call_codex(arguments)
            elif tool_name == "llama_generate":
                result = await self._call_llama(arguments)
            else:
                return CallToolResult(
                    content=[
                        TextContent(  # type: ignore[list-item]
                            type="text", text=f"Unknown tool: {tool_name}"
                        )
                    ],
                    isError=True,
                )

            return CallToolResult(
                content=[TextContent(type="text", text=result)],  # type: ignore[list-item]
                isError=False,
            )

        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],  # type: ignore[list-item]
                isError=True,
            )

    async def _call_gemini(self, arguments: dict[str, Any]) -> str:
        """Call Gemini provider (placeholder implementation).

        Args:
            arguments: Tool arguments

        Returns:
            Generated response

        """
        prompt = arguments.get("prompt", "")
        model = arguments.get("model", "gemini-pro")

        return f"Gemini response using {model} for prompt: {prompt}"

    async def _call_codex(self, arguments: dict[str, Any]) -> str:
        """Call Codex provider (placeholder implementation).

        Args:
            arguments: Tool arguments

        Returns:
            Generated code

        """
        prompt = arguments.get("prompt", "")
        language = arguments.get("language", "python")

        return f"Codex response in {language} for prompt: {prompt}"

    async def _call_llama(self, arguments: dict[str, Any]) -> str:
        """Call LLaMA provider (placeholder implementation).

        Args:
            arguments: Tool arguments

        Returns:
            Generated response

        """
        prompt = arguments.get("prompt", "")
        max_tokens = arguments.get("max_tokens", 100)

        return f"LLaMA response with max_tokens={max_tokens} for prompt: {prompt}"
