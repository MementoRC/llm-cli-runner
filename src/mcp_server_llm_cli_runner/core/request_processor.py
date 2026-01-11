"""Request processing and routing functionality for the MCP server.

This module provides request processing, routing, and context management
functionality for handling MCP protocol requests efficiently.
"""

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from mcp_server_llm_cli_runner.utils.logging import get_logger


class RequestProcessor:
    """Main request processor for handling MCP requests.

    This class handles the core request processing pipeline including
    routing, validation, and response generation.
    """

    def __init__(self) -> None:
        """Initialize request processor."""
        self._logger = get_logger(__name__)
        self._handlers: dict[str, Callable] = {}
        self._middleware: list[Callable] = []
        self._validation_rules: dict[str, list[Callable]] = {}
        self._metrics = {
            "requests_processed": 0,
            "requests_failed": 0,
            "total_processing_time": 0.0,
        }

    @property
    def handlers(self) -> dict[str, Callable]:
        """Get the registered handlers dictionary.

        Returns:
            Dictionary of registered handlers
        """
        return self._handlers

    def register_handler(self, method: str, handler: Callable) -> None:
        """Register a handler for a specific method.

        Args:
            method: The method name to handle
            handler: The handler function

        """
        self._handlers[method] = handler
        self._logger.info(f"Registered handler for method: {method}")

    def register_middleware(self, middleware: Callable) -> None:
        """Register middleware for request processing.

        Args:
            middleware: The middleware function

        """
        self._middleware.append(middleware)
        self._logger.info("Registered middleware")

    def add_validation_rule(self, method: str, rule: Callable) -> None:
        """Add a validation rule for a specific method.

        Args:
            method: The method name to add validation for
            rule: The validation rule function

        """
        if method not in self._validation_rules:
            self._validation_rules[method] = []
        self._validation_rules[method].append(rule)
        self._logger.info(f"Added validation rule for method: {method}")

    async def route_to_handler(self, method: str, params: dict[str, Any]) -> Any:
        """Route request to appropriate handler.

        Args:
            method: The method name
            params: The parameters

        Returns:
            The handler response

        Raises:
            ValueError: If no handler found for method

        """
        if method not in self._handlers:
            msg = f"No handler registered for method: {method}"
            raise ValueError(msg)

        handler = self._handlers[method]
        return await handler(params)

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process a request through the handler pipeline.

        Args:
            request: The request data

        Returns:
            The processed response

        """
        # Validate request structure
        from mcp_server_llm_cli_runner.core.errors import ValidationError

        if not isinstance(request, dict):
            raise ValidationError("Request must be a dictionary")

        method = request.get("method")
        if not method:
            raise ValidationError("Request must have a method")

        start_time = time.time()

        try:
            self._metrics["requests_processed"] += 1

            # Apply middleware
            for middleware in self._middleware:
                request = await middleware(request)

            # Apply validation rules
            params = request.get("params", {})
            if method in self._validation_rules:
                for rule in self._validation_rules[method]:
                    await rule(params)

            # Route to handler
            handler_response = await self._route_request(method, request)

            # Record metrics
            processing_time = time.time() - start_time
            self._metrics["total_processing_time"] += processing_time

            # Preserve request ID in response
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": handler_response,
            }

            return response

        except Exception as e:
            self._metrics["requests_failed"] += 1
            self._logger.exception(f"Request processing failed: {e}")
            raise

    async def _route_request(
        self,
        method: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Route request to appropriate handler.

        Args:
            method: The method to route to
            request: The request data

        Returns:
            The handler response

        """
        from mcp_server_llm_cli_runner.core.errors import ValidationError

        if method not in self._handlers:
            msg = f"No handler for method: {method}"
            raise ValidationError(msg)

        handler = self._handlers[method]
        params = request.get("params", {})

        self._logger.debug(f"Routing to handler for method: {method}")
        return await handler(params)


class RequestContextManager:
    """Hybrid object that acts as both async context manager and dict."""

    def __init__(self, parent: "ContextManager") -> None:
        """Initialize request context manager.

        Args:
            parent: The parent context manager instance

        """
        self._parent = parent

    def __contains__(self, key: str) -> bool:
        """Check if key exists in context.

        Args:
            key: The key to check

        Returns:
            True if key exists in context

        """
        return key in self._parent._request_contexts

    def __getitem__(self, key: str) -> Any:
        """Get item from context.

        Args:
            key: The key to retrieve

        Returns:
            The value associated with the key

        """
        return self._parent._request_contexts[key]

    def pop(self, key: str, default: Any = None) -> Any:
        """Remove and return item from context.

        Args:
            key: The key to remove
            default: Default value if key not found

        Returns:
            The removed value or default

        """
        return self._parent._request_contexts.pop(key, default)

    def __call__(self, request_id: str) -> Any:
        """Create request context for given ID.

        Args:
            request_id: The request identifier

        Returns:
            The created request context (async context manager)

        """
        return self._parent._request_context_manager(request_id)


class ContextManager:
    """Context manager for request processing.

    This class manages request contexts, providing isolation and
    cleanup for concurrent request processing.
    """

    def __init__(self) -> None:
        """Initialize context manager."""
        self._logger = get_logger(__name__)
        self._request_contexts: dict[str, dict[str, Any]] = {}
        self._context_lock = asyncio.Lock()
        self._cleanup_tasks: list[Callable] = []

        # Create a hybrid object that acts as both method and dict
        self.request_context = RequestContextManager(self)

    async def _create_request_context(self, request_id: str) -> dict[str, Any]:
        """Create a new request context.

        Args:
            request_id: The request identifier

        Returns:
            The created context dictionary

        """
        async with self._context_lock:
            context = {
                "request_id": request_id,
                "start_time": time.time(),
                "metadata": {},
                "resources": [],
            }
            self._request_contexts[request_id] = context
            return context

    def add_cleanup_task(self, task: Callable) -> None:
        """Add a cleanup task.

        Args:
            task: The cleanup task to add

        """
        self._cleanup_tasks.append(task)

    async def cleanup_context(self, request_id: str) -> None:
        """Clean up a request context.

        Args:
            request_id: The request identifier to clean up

        """
        async with self._context_lock:
            # Run cleanup tasks
            for cleanup_task in self._cleanup_tasks:
                try:
                    if asyncio.iscoroutinefunction(cleanup_task):
                        await cleanup_task()
                    else:
                        cleanup_task()
                except Exception as e:
                    self._logger.error(f"Cleanup task failed: {e}")

            if request_id in self._request_contexts:
                del self._request_contexts[request_id]
                self._logger.debug(f"Cleaned up context for request: {request_id}")

    @asynccontextmanager
    async def _request_context_manager(self, request_id: str | None = None):
        """Async context manager for request processing.

        Args:
            request_id: Optional request identifier

        Yields:
            The request context

        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        context = await self._create_request_context(request_id)
        try:
            yield context
        finally:
            await self.cleanup_context(request_id)

    def get_active_contexts(self) -> list[str]:
        """Get list of active context IDs.

        Returns:
            List of active request context IDs

        """
        return list(self._request_contexts.keys())

    def get_context_stats(self) -> dict[str, Any]:
        """Get context manager statistics.

        Returns:
            Dictionary with context statistics

        """
        return {
            "active_contexts": len(self._request_contexts),
            "context_ids": list(self._request_contexts.keys()),
        }


class RequestRouter:
    """Router for directing requests to appropriate processors.

    This class provides request routing functionality based on
    method names, content types, and custom routing rules.
    """

    def __init__(self) -> None:
        """Initialize request router."""
        self._logger = get_logger(__name__)
        self._routes: dict[str, Callable] = {}
        self._default_handler: Callable | None = None

    def register_route(self, pattern: str, handler: Callable) -> None:
        """Register a route pattern with a handler.

        Args:
            pattern: The route pattern (method name)
            handler: The handler function

        """
        self._routes[pattern] = handler
        self._logger.info(f"Registered route: {pattern}")

    def set_default_handler(self, handler: Callable) -> None:
        """Set the default handler for unmatched routes.

        Args:
            handler: The default handler function

        """
        self._default_handler = handler
        self._logger.info("Set default route handler")

    async def route_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Route a request to the appropriate handler.

        Args:
            request: The request to route

        Returns:
            The handler response

        """
        method = request.get("method", "")

        # Try exact match first
        if method in self._routes:
            handler = self._routes[method]
            self._logger.debug(f"Routing {method} to registered handler")
            return await handler(request)

        # Fall back to default handler
        if self._default_handler:
            self._logger.debug(f"Routing {method} to default handler")
            return await self._default_handler(request)

        # No handler found
        msg = f"No route handler found for method: {method}"
        raise ValueError(msg)

    def get_routes(self) -> list[str]:
        """Get list of registered routes.

        Returns:
            List of registered route patterns

        """
        return list(self._routes.keys())

    def get_route_stats(self) -> dict[str, Any]:
        """Get router statistics.

        Returns:
            Dictionary with router statistics

        """
        return {
            "registered_routes": len(self._routes),
            "routes": list(self._routes.keys()),
            "has_default_handler": self._default_handler is not None,
        }


class ResponseFormatter:
    """Formatter for standardizing response formats.

    This class handles response formatting, serialization,
    and protocol compliance for MCP responses.
    """

    def __init__(self) -> None:
        """Initialize response formatter."""
        self._logger = get_logger(__name__)

    def format_success_response(self, request_id: Any, result: Any) -> dict[str, Any]:
        """Format a successful response.

        Args:
            request_id: The request identifier
            result: The result data

        Returns:
            Formatted success response

        """
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def format_error_response(
        self,
        request_id: Any,
        error_code: int,
        error_message: str,
    ) -> dict[str, Any]:
        """Format an error response.

        Args:
            request_id: The request identifier
            error_code: The error code
            error_message: The error message

        Returns:
            Formatted error response

        """
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": error_code,
                "message": error_message,
            },
        }

    def serialize_response(self, response: dict[str, Any]) -> str:
        """Serialize response to JSON string.

        Args:
            response: The response dictionary

        Returns:
            JSON string representation

        """
        try:
            return json.dumps(response, ensure_ascii=False)
        except Exception as e:
            self._logger.exception(f"Response serialization failed: {e}")
            # Return a basic error response
            return json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32603,
                        "message": "Internal error during response serialization",
                    },
                },
            )


class RequestValidator:
    """Validator for request data and structure.

    This class provides request validation functionality
    including schema validation and protocol compliance.
    """

    def __init__(self) -> None:
        """Initialize request validator."""
        self._logger = get_logger(__name__)

    def validate_jsonrpc_structure(self, request: dict[str, Any]) -> bool:
        """Validate JSON-RPC 2.0 structure.

        Args:
            request: The request to validate

        Returns:
            True if valid, False otherwise

        """
        if not isinstance(request, dict):
            return False

        # Check required fields
        if request.get("jsonrpc") != "2.0":
            return False

        # Must have either method (request) or result/error (response)
        has_method = "method" in request
        has_result_or_error = "result" in request or "error" in request

        return has_method or has_result_or_error

    def validate_method_request(self, request: dict[str, Any]) -> bool:
        """Validate method request structure.

        Args:
            request: The request to validate

        Returns:
            True if valid, False otherwise

        """
        if not self.validate_jsonrpc_structure(request):
            return False

        # Method requests must have a method field
        if "method" not in request:
            return False

        # Method must be string
        return isinstance(request["method"], str)

    async def validate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate request and return normalized version.

        Args:
            request: The request to validate

        Returns:
            Normalized request

        Raises:
            ValueError: If validation fails

        """
        if not self.validate_jsonrpc_structure(request):
            msg = "Invalid JSON-RPC structure"
            raise ValueError(msg)

        if "method" in request and not self.validate_method_request(request):
            msg = "Invalid method request structure"
            raise ValueError(msg)

        return request


# Convenience functions for common operations
async def create_request_processor() -> RequestProcessor:
    """Create and configure a default request processor.

    Returns:
        Configured request processor instance

    """
    return RequestProcessor()

    # Register default handlers here if needed


async def create_context_manager() -> ContextManager:
    """Create a context manager instance.

    Returns:
        Context manager instance

    """
    return ContextManager()


async def create_request_router() -> RequestRouter:
    """Create a request router instance.

    Returns:
        Request router instance

    """
    return RequestRouter()
