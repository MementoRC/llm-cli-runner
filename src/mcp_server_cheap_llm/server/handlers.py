"""Server handlers and protocol implementation.

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

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom validation error for protocol handling."""

    def __init__(self, message: str) -> None:
        """Initialize ValidationError with message.

        Args:
            message: Error message describing the validation failure

        """
        super().__init__(message)
        self.message = message


# Type compatibility layer for MCP types
try:
    # Try to import MCP types if available
    from mcp.types import CallToolRequest, CallToolResult, TextContent, Tool

except ImportError:
    # Protocol definitions for type checking when MCP types unavailable
    @runtime_checkable
    class Tool(Protocol):
        """Tool protocol for type safety."""

        name: str
        description: str
        inputSchema: dict[str, Any]

        def __init__(
            self,
            name: str,
            description: str,
            inputSchema: dict[str, Any],
        ) -> None:
            """Initialize Tool."""
            ...

    @runtime_checkable
    class TextContent(Protocol):
        """TextContent protocol for type safety."""

        text: str
        type: str

        def __init__(self, text: str, type: str = "text") -> None:
            """Initialize TextContent."""
            ...

    @runtime_checkable
    class CallToolResult(Protocol):
        """CallToolResult protocol for type safety."""

        content: list[Any]
        isError: bool

        def __init__(self, content: list[Any], isError: bool = False) -> None:
            """Initialize CallToolResult."""
            ...

    @runtime_checkable
    class CallToolRequest(Protocol):
        """CallToolRequest protocol for type safety."""

        method: str
        params: Any

        def __init__(self, method: str, params: Any) -> None:
            """Initialize CallToolRequest."""
            ...

    # Fallback implementations for runtime
    class _Tool:
        """Fallback Tool implementation."""

        def __init__(
            self,
            name: str,
            description: str,
            inputSchema: dict[str, Any],
        ) -> None:
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

    class _TextContent:
        """Fallback TextContent implementation."""

        def __init__(self, text: str, type: str = "text") -> None:
            self.text = text
            self.type = type

    class _CallToolResult:
        """Fallback CallToolResult implementation."""

        def __init__(self, content: list[Any], isError: bool = False) -> None:
            self.content = content
            self.isError = isError

    class _CallToolRequest:
        """Fallback CallToolRequest implementation."""

        def __init__(self, method: str, params: Any) -> None:
            self.method = method
            self.params = params

    # Use fallback implementations at runtime
    Tool = _Tool  # type: ignore[misc,assignment]  # noqa: F811
    TextContent = _TextContent  # type: ignore[misc,assignment]  # noqa: F811
    CallToolResult = _CallToolResult  # type: ignore[misc,assignment]  # noqa: F811
    CallToolRequest = _CallToolRequest  # type: ignore[misc,assignment]  # noqa: F811


class MCPProtocolHandler:
    """MCP Protocol Handler for JSON-RPC message processing.

    This class handles the core MCP protocol operations including
    message validation, routing, and response formatting.
    """

    def __init__(self) -> None:
        """Initialize MCP protocol handler."""
        self.logger = get_logger(__name__)
        self._metrics: dict[str, int] = {
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
        try:
            # Parse JSON
            data = json.loads(message)
            self._metrics["requests_processed"] += 1

            # Validate JSON-RPC structure
            validated_data = self._validate_jsonrpc(data)

            # Route message based on method
            response = await self._route_message(validated_data)

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
            error_response = {
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": f"Invalid Request: {e.message}"},
                "id": data.get("id") if "data" in locals() else None,
            }
            return json.dumps(error_response)

        except Exception as e:
            self._metrics["errors_encountered"] += 1
            self.logger.exception(f"Unexpected error processing message: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": "Internal error"},
                "id": data.get("id") if "data" in locals() else None,
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
                msg = "Invalid or missing 'jsonrpc' field"
                raise ValidationError(msg)

            # Check if it's a request (has method) or response (has result/error)
            if "method" in data:
                # It's a request - validate method
                method = data["method"]
                if not isinstance(method, str):
                    self._metrics["invalid_jsonrpc"] += 1
                    self._metrics["errors_encountered"] += 1
                    msg = "'method' must be string"
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
                    msg = "'id' must be string, number, or null"
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
            # Parse JSON
            data = json.loads(message)
            self._metrics["requests_processed"] += 1

            # Validate JSON-RPC structure
            validated_data = self._validate_jsonrpc(data)
            return validated_data

        except json.JSONDecodeError as e:
            self._metrics["errors_encountered"] += 1
            self._metrics["invalid_jsonrpc"] += 1
            msg = f"Invalid JSON: {e}"
            raise ValidationError(msg) from e

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle parsed JSON-RPC request.

        Args:
            request: Parsed JSON-RPC request dictionary

        Returns:
            Response dictionary

        """
        message_id = request.get("id")

        # Handle notifications (no id)
        if message_id is None:
            # Notifications don't get responses
            return None

        # Route to appropriate handler
        try:
            response_data = await self._route_message(request)
            return self.create_response(message_id, response_data)
        except Exception as e:
            self.logger.exception(f"Error handling request: {e}")
            return self.create_error_response(
                message_id, -32603, "Internal error", {"details": str(e)}
            )

    def create_response(self, message_id: Any, result: Any) -> dict[str, Any]:
        """Create JSON-RPC success response.

        Args:
            message_id: Request ID to echo back
            result: Result data

        Returns:
            JSON-RPC response dictionary

        """
        self._metrics["responses_sent"] += 1
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": message_id,
        }

    def create_error_response(
        self,
        message_id: Any,
        code: int,
        message: str,
        data: Any = None,
    ) -> dict[str, Any]:
        """Create JSON-RPC error response.

        Args:
            message_id: Request ID to echo back
            code: Error code
            message: Error message
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
            "id": message_id,
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
            "serverInfo": {"name": "cheap-llm-server", "version": "0.1.0"},
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

        """
        tool_name = params.get("name", "unknown")

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
            from mcp_server_cheap_llm.core.errors import ValidationError

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
        from_version: str,
        to_version: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Migrate parameters between tool versions.

        Args:
            tool_name: Name of the tool
            from_version: Source version
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

        """
        tool_name = tool_spec.get("name", "unknown")

        # Store tool with provider configuration
        provider_tool = {
            "name": tool_name,
            "description": tool_spec.get("description", ""),
            "inputSchema": tool_spec.get("inputSchema", {}),
            "provider_config": tool_spec.get("inputSchema", {}).get("properties", {}),
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

        """
        with self._lock:
            if provider not in self._tools:
                self._tools[provider] = {}

            tool_name = tool_spec.get("name", "unknown")
            self._tools[provider][tool_name] = {
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
            if provider in self._tools and tool_name in self._tools[provider]:
                del self._tools[provider][tool_name]

                # Clean up empty provider entries
                if not self._tools[provider]:
                    del self._tools[provider]
                    self._providers.discard(provider)

                self.logger.info(
                    f"Unregistered tool '{tool_name}' from provider '{provider}'",
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
                if provider in self._tools:
                    for tool_data in self._tools[provider].values():
                        tools.append(tool_data["spec"])
            else:
                # Return all tools from all providers
                for provider_tools in self._tools.values():
                    for tool_data in provider_tools.values():
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
            ValueError: If tool or provider not found

        """
        handler = self.get_tool_handler(tool_name, provider)
        if not handler:
            msg = f"Tool '{tool_name}' not found for provider '{provider}'"
            raise ValueError(msg)

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
                return len(self._tools.get(provider, {}))
            return sum(len(tools) for tools in self._tools.values())

    def clear_provider(self, provider: str) -> int:
        """Clear all tools for a provider.

        Args:
            provider: Provider name

        Returns:
            Number of tools removed

        """
        with self._lock:
            if provider in self._tools:
                count = len(self._tools[provider])
                del self._tools[provider]
                self._providers.discard(provider)
                self.logger.info(f"Cleared {count} tools for provider '{provider}'")
                return count
        return 0

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry statistics

        """
        with self._lock:
            return {
                "total_providers": len(self._providers),
                "total_tools": sum(len(tools) for tools in self._tools.values()),
                "providers": list(self._providers),
                "tools_by_provider": {
                    provider: list(tools.keys())
                    for provider, tools in self._tools.items()
                },
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

    async def create_stream_session(self, session_id: str | None = None) -> str:
        """Create a new streaming session.

        Args:
            session_id: Optional session identifier

        Returns:
            Stream session ID

        """
        stream_id = session_id or str(uuid.uuid4())

        with self._lock:
            self._active_streams[stream_id] = {
                "created_at": datetime.now(),
                "status": "active",
                "chunks_sent": 0,
                "bytes_sent": 0,
            }

        return stream_id

    async def stream_response(
        self,
        stream_id: str,
        data: Any,
        max_chunks: int | None = None,
    ):
        """Stream response data for a session.

        Args:
            stream_id: Stream session identifier
            data: Data to stream (can be string or iterable)
            max_chunks: Optional maximum chunks to send

        Yields:
            Chunks of data

        """
        if stream_id not in self._active_streams:
            msg = f"Stream session {stream_id} not found"
            raise ValidationError(msg)

        # Convert data to chunks
        if isinstance(data, str):
            chunks = [
                data[i : i + self.chunk_size]
                for i in range(0, len(data), self.chunk_size)
            ]
        elif hasattr(data, "__iter__"):
            chunks = list(data)
        else:
            chunks = [str(data)]

        # Limit chunks if specified
        if max_chunks:
            chunks = chunks[:max_chunks]

        # Stream chunks with flow control
        for chunk in chunks:
            # Check for backpressure
            if self.flow_control_enabled:
                await self._check_backpressure(stream_id)

            # Update metrics
            with self._lock:
                self._active_streams[stream_id]["chunks_sent"] += 1
                self._active_streams[stream_id]["bytes_sent"] += len(str(chunk))
                self._metrics["total_chunks_sent"] += 1
                self._metrics["total_bytes_sent"] += len(str(chunk))

            yield chunk

    async def _check_backpressure(self, stream_id: str) -> None:
        """Check and handle backpressure for streaming.

        Args:
            stream_id: Stream session identifier

        """
        # Simulate backpressure handling
        import asyncio

        await asyncio.sleep(0)  # Yield control

    def get_stream_metrics(self, stream_id: str) -> dict[str, Any]:
        """Get metrics for a specific stream.

        Args:
            stream_id: Stream session identifier

        Returns:
            Stream metrics dictionary

        """
        with self._lock:
            return self._active_streams.get(stream_id, {}).copy()

    def close_session(self, session_id: str) -> None:
        """Close a session (alias for close_stream).

        Args:
            session_id: Session identifier

        """
        with self._lock:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
                self._metrics["active_connections"] -= 1
            if session_id in self._active_streams:
                del self._active_streams[session_id]

    async def close_stream(self, stream_id: str) -> None:
        """Close a streaming session.

        Args:
            stream_id: Stream session identifier

        """
        with self._lock:
            if stream_id in self._active_streams:
                self._active_streams[stream_id]["status"] = "closed"


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
    """Main CheapLLM MCP Server implementation."""

    def __init__(
        self,
        config_manager: Any = None,
        middleware_chain: Any = None,
        resource_manager: Any = None,
    ) -> None:
        """Initialize CheapLLM server.

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

        self.logger.info("Initializing CheapLLM server...")

        try:
            # Initialize components in order
            await self._initialize_components()

            # Register default tools
            await self._register_default_tools()

            # Set up server endpoints
            await self._setup_server_endpoints()

            self._initialized = True
            self.logger.info("CheapLLM server initialized successfully")

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

        self.logger.info(f"Starting CheapLLM server on {host}:{port}")

        # Server startup logic would go here
        # For now, just log that the server is starting
        self.logger.info("Server started successfully")

    async def stop(self) -> None:
        """Stop the server gracefully."""
        self.logger.info("Stopping CheapLLM server...")

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
