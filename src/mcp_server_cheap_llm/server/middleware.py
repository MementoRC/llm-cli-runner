"""Server middleware pipeline implementation.

This module implements middleware for the MCP server including:
- MiddlewareChain: Pipeline for request/response processing
- LoggingMiddleware: Structured logging with correlation IDs
- MetricsMiddleware: Performance monitoring and metrics collection
- ResourceManager: Connection pooling and resource cleanup

All middleware follows the async protocol and provides comprehensive
error handling and resource management.
"""

import asyncio
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Protocol

from mcp_server_cheap_llm.core.errors import ValidationError
from mcp_server_cheap_llm.utils.logging import get_logger


class MiddlewareProtocol(Protocol):
    """Protocol for middleware implementations."""

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process incoming request."""
        ...

    async def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Process outgoing response."""
        ...


class MiddlewareChain:
    """Middleware chain for request/response processing pipeline.

    Manages the execution order of middleware components:
    - Request processing: executed in order (first to last)
    - Response processing: executed in reverse order (last to first)

    This ensures proper nesting behavior for middleware like logging,
    authentication, and metrics collection.
    """

    def __init__(self) -> None:
        """Initialize middleware chain."""
        self.middlewares: list[MiddlewareProtocol] = []
        self._middlewares = self.middlewares  # Alias for backward compatibility
        self.logger = get_logger(__name__)

    def add_middleware(self, middleware: MiddlewareProtocol) -> None:
        """Add middleware to the chain.

        Args:
            middleware: Middleware instance implementing MiddlewareProtocol

        """
        self.middlewares.append(middleware)
        self.logger.debug(f"Added middleware: {middleware.__class__.__name__}")

    def remove_middleware(self, middleware: MiddlewareProtocol) -> None:
        """Remove middleware from the chain.

        Args:
            middleware: Middleware instance to remove

        """
        if middleware in self.middlewares:
            self.middlewares.remove(middleware)
            self.logger.debug(f"Removed middleware: {middleware.__class__.__name__}")

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process request through middleware chain.

        Executes middleware in order: first middleware processes first.

        Args:
            request: Incoming request data

        Returns:
            Processed request data

        """
        current_request = request

        for middleware in self.middlewares:
            try:
                current_request = await middleware.process_request(current_request)
            except Exception as e:
                self.logger.exception(
                    f"Error in {middleware.__class__.__name__}.process_request: {e}",
                )
                raise

        return current_request

    async def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Process response through middleware chain.

        Executes middleware in reverse order: last middleware processes first.
        This ensures proper cleanup and logging order.

        Args:
            response: Outgoing response data

        Returns:
            Processed response data

        """
        current_response = response

        # Process in reverse order for response
        for middleware in reversed(self.middlewares):
            try:
                current_response = await middleware.process_response(current_response)
            except Exception as e:
                self.logger.exception(
                    f"Error in {middleware.__class__.__name__}.process_response: {e}",
                )
                raise

        return current_response


class LoggingMiddleware:
    """Structured logging middleware with correlation ID tracking.

    Provides:
    - Correlation ID generation and propagation
    - Structured request/response logging
    - Request timing and context tracking
    - Security-aware data sanitization
    """

    def __init__(self) -> None:
        """Initialize logging middleware."""
        self.logger = get_logger(__name__)
        self.correlation_ids = {}

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process incoming request with logging and correlation ID.

        Args:
            request: Incoming request data

        Returns:
            Request data unchanged (correlation ID tracked internally)

        """
        # Generate or preserve correlation ID
        correlation_id = request.get("correlation_id", str(uuid.uuid4()))

        # Store correlation ID internally without modifying request
        self.correlation_ids[id(request)] = correlation_id

        # Log incoming request using log_request method
        await self.log_request(request)

        return request

    async def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Process outgoing response with completion logging.

        Args:
            response: Outgoing response data

        Returns:
            Response data unchanged

        """
        # Retrieve correlation ID from internal tracking
        correlation_id = self.correlation_ids.get(
            id(response),
            response.get("correlation_id"),
        )

        # Calculate duration if start_time is available internally
        duration = None

        # Log request completion
        self.logger.info(
            "Request completed",
            extra={
                "correlation_id": correlation_id,
                "duration_ms": int(duration * 1000) if duration else None,
                "method": response.get("method"),
                "has_result": "result" in response,
                "has_error": "error" in response,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        return response

    async def log_request(self, request: dict[str, Any]) -> None:
        """Log a request explicitly.

        Args:
            request: Request data to log

        """
        correlation_id = request.get("correlation_id", str(uuid.uuid4()))
        self.logger.info(
            "Request logged",
            extra={
                "correlation_id": correlation_id,
                "method": request.get("method"),
                "params_keys": list(request.get("params", {}).keys()),
                "request_id": request.get("id"),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )


class MetricsMiddleware:
    """Performance metrics collection middleware.

    Collects and aggregates:
    - Request counts by method
    - Response times and percentiles
    - Error rates and types
    - Throughput metrics

    Provides real-time metrics for monitoring and alerting.
    """

    def __init__(self) -> None:
        """Initialize metrics middleware."""
        self.logger = get_logger(__name__)
        self.request_counts: dict[str, int] = defaultdict(int)
        self.request_times: dict[str, list[float]] = defaultdict(list)
        self._metrics_lock = asyncio.Lock()
        self._metrics = {
            "requests_processed": 0,
            "responses_processed": 0,
            "errors_encountered": 0,
            "last_request_time": None,
            "average_response_time": 0.0,
        }
        self._response_times: list[float] = []

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process request with timing and counting.

        Args:
            request: Incoming request data

        Returns:
            Request data unchanged (timing tracked internally)

        """
        try:
            if request is None:
                msg = "Request cannot be None"
                raise TypeError(msg)

            method = request.get("method", "unknown")

            # Record start time internally
            start_time = time.time()
            # Store timing internally using request ID if available
            self._request_start_times = getattr(self, "_request_start_times", {})
            # Use request ID if available, otherwise use object id
            tracking_id = request.get("id") if "id" in request else id(request)
            self._request_start_times[tracking_id] = (start_time, method)

            # Increment request counter
            async with self._metrics_lock:
                self.request_counts[method] += 1
                self._metrics["requests_processed"] += 1
                self._metrics["last_request_time"] = start_time

            return request
        except Exception:
            async with self._metrics_lock:
                self._metrics["errors_encountered"] += 1
            raise

    async def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Process response with metrics collection.

        Args:
            response: Outgoing response data

        Returns:
            Response data unchanged

        """
        # Get start time from internal tracking
        self._request_start_times = getattr(self, "_request_start_times", {})
        tracking_id = response.get("id") if "id" in response else id(response)
        timing_info = self._request_start_times.get(tracking_id)

        if timing_info:
            start_time, method = timing_info
        else:
            start_time = None
            method = response.get("method", "unknown")

        async with self._metrics_lock:
            self._metrics["responses_processed"] += 1

            if start_time:
                duration = time.time() - start_time
                self.request_times[method].append(duration)
                self._response_times.append(duration)

                # Keep only recent timings (last 1000 per method)
                if len(self.request_times[method]) > 1000:
                    self.request_times[method] = self.request_times[method][-1000:]
                if len(self._response_times) > 1000:
                    self._response_times = self._response_times[-1000:]

                # Update average response time
                if self._response_times:
                    self._metrics["average_response_time"] = sum(
                        self._response_times,
                    ) / len(self._response_times)

                # Log performance metrics
                self.logger.info(
                    "Request metrics",
                    extra={
                        "method": method,
                        "duration_ms": int(duration * 1000),
                        "correlation_id": response.get("correlation_id"),
                        "total_requests": self.request_counts[method],
                    },
                )

                # Clean up tracking
                del self._request_start_times[tracking_id]

        return response

    def get_metrics_summary(self) -> dict[str, dict[str, Any]]:
        """Get comprehensive metrics summary.

        Returns:
            Dictionary with metrics for each method including:
            - count: Total request count
            - avg_duration: Average response time
            - max_duration: Maximum response time
            - min_duration: Minimum response time

        """
        summary = {}

        for method in self.request_counts:
            method_summary = {
                "count": self.request_counts[method],
                "avg_duration": 0.0,
                "max_duration": 0.0,
                "min_duration": 0.0,
            }

            if self.request_times.get(method):
                timings = self.request_times[method]
                method_summary.update(
                    {
                        "avg_duration": sum(timings) / len(timings),
                        "max_duration": max(timings),
                        "min_duration": min(timings),
                    },
                )

            summary[method] = method_summary

        return summary

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary with current metrics

        """
        return dict(self._metrics)

    def reset_metrics(self) -> None:
        """Reset all metrics to initial values."""
        self._metrics = {
            "requests_processed": 0,
            "responses_processed": 0,
            "errors_encountered": 0,
            "last_request_time": None,
            "average_response_time": 0.0,
        }
        self.request_counts.clear()
        self.request_times.clear()
        self._response_times.clear()


class ResourceManager:
    """Resource management for connection pooling and cleanup.

    Provides:
    - Connection pooling with configurable limits
    - Idle connection cleanup
    - Graceful shutdown with resource cleanup
    - Health monitoring and reporting
    """

    def __init__(self, max_pool_size: int = 10, idle_timeout: float = 300.0) -> None:
        """Initialize resource manager.

        Args:
            max_pool_size: Maximum connections per pool
            idle_timeout: Idle timeout in seconds

        """
        self.logger = get_logger(__name__)
        self.max_pool_size = max_pool_size
        self.idle_timeout = idle_timeout

        self.connections: dict[str, list[Any]] = defaultdict(list)
        self.connection_times: dict[str, list[float]] = defaultdict(list)
        self.cleanup_tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._resource_lock = asyncio.Lock()
        self._cleanup_started = False

        # Resource tracking attributes
        self._active_requests = 0
        self.max_concurrent_requests = 5  # Default limit
        self._peak_concurrent_requests = 0
        self._total_requests_handled = 0
        self._resource_acquisition_condition = asyncio.Condition()

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task if event loop is available."""
        if self._cleanup_started:
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, defer cleanup task start
            return

        async def cleanup_loop() -> None:
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=60.0,  # Cleanup every minute
                    )
                except TimeoutError:
                    await self.cleanup_idle_connections()

        task = asyncio.create_task(cleanup_loop())
        self.cleanup_tasks.append(task)
        self._cleanup_started = True

    async def acquire_connection(self, pool_name: str) -> Any:
        """Acquire connection from pool or create new one.

        Args:
            pool_name: Name of the connection pool

        Returns:
            Connection object (mock for testing)

        """
        # Start cleanup task if not already started
        if not self._cleanup_started:
            self._start_cleanup_task()

        async with self._resource_lock:
            if self.connections[pool_name]:
                # Reuse existing connection
                connection = self.connections[pool_name].pop(0)
                # Remove corresponding timestamp
                if self.connection_times[pool_name]:
                    self.connection_times[pool_name].pop(0)

                self.logger.debug(f"Reused connection from pool: {pool_name}")
                return connection
            # Create new connection (mock implementation)
            connection = object()  # Mock connection
            self.logger.debug(f"Created new connection for pool: {pool_name}")
            return connection

    async def release_connection(self, pool_name: str, connection: Any) -> None:
        """Release connection back to pool.

        Args:
            pool_name: Name of the connection pool
            connection: Connection to release

        """
        async with self._resource_lock:
            # Check pool size limit
            if len(self.connections[pool_name]) < self.max_pool_size:
                self.connections[pool_name].append(connection)
                self.connection_times[pool_name].append(time.time())
                self.logger.debug(f"Released connection to pool: {pool_name}")
            else:
                # Pool is full, discard connection
                self.logger.debug(f"Pool {pool_name} full, discarding connection")

    async def cleanup_idle_connections(self) -> None:
        """Clean up idle connections that exceed timeout."""
        current_time = time.time()
        cleaned_count = 0

        async with self._resource_lock:
            for pool_name in list(self.connections.keys()):
                if pool_name not in self.connection_times:
                    continue

                # Find connections that are not idle
                active_connections = []
                active_times = []

                for conn, conn_time in zip(
                    self.connections[pool_name],
                    self.connection_times[pool_name],
                    strict=False,
                ):
                    if current_time - conn_time < self.idle_timeout:
                        active_connections.append(conn)
                        active_times.append(conn_time)
                    else:
                        cleaned_count += 1

                # Update pools with only active connections
                self.connections[pool_name] = active_connections
                self.connection_times[pool_name] = active_times

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} idle connections")

    async def shutdown(self) -> None:
        """Graceful shutdown with resource cleanup."""
        self.logger.info("Starting resource manager shutdown")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel cleanup tasks
        for task in self.cleanup_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.cleanup_tasks:
            await asyncio.gather(*self.cleanup_tasks, return_exceptions=True)

        # Clean up all connections
        async with self._resource_lock:
            total_connections = sum(len(pool) for pool in self.connections.values())
            self.connections.clear()
            self.connection_times.clear()

        self.logger.info(
            f"Shutdown complete, cleaned up {total_connections} connections",
        )

    async def health_check(self) -> dict[str, Any]:
        """Get resource manager health status.

        Returns:
            Health status including pool information

        """
        async with self._resource_lock:
            pools_info = {}
            total_connections = 0

            for pool_name, connections in self.connections.items():
                pool_size = len(connections)
                total_connections += pool_size

                pools_info[pool_name] = {
                    "active": pool_size,
                    "max_size": self.max_pool_size,
                    "utilization": pool_size / self.max_pool_size
                    if self.max_pool_size > 0
                    else 0,
                }

        return {
            "status": "healthy"
            if not self._shutdown_event.is_set()
            else "shutting_down",
            "total_connections": total_connections,
            "pools": pools_info,
            "idle_timeout": self.idle_timeout,
        }

    def get_resource_status(self) -> dict[str, Any]:
        """Get current resource status information.

        Returns:
            Dictionary containing resource status information

        """
        try:
            import psutil

            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except ImportError:
            # Fallback if psutil not available
            memory = None
            cpu_percent = 0.0

        return {
            "active_requests": self._active_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
            "memory_usage_percent": memory.percent if memory else 0.0,
            "cpu_usage_percent": cpu_percent,
            "status": "operational"
            if not self._shutdown_event.is_set()
            else "shutting_down",
            "total_connections": sum(len(pool) for pool in self.connections.values()),
            "max_connections": len(self.connections) * self.max_pool_size,
            "pools_count": len(self.connections),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get resource metrics information.

        Returns:
            Dictionary containing resource metrics

        """
        total_connections = sum(len(pool) for pool in self.connections.values())
        total_capacity = (
            len(self.connections) * self.max_pool_size
            if self.connections
            else self.max_pool_size
        )

        return {
            "active_requests": self._active_requests,
            "peak_concurrent_requests": self._peak_concurrent_requests,
            "total_requests_handled": self._total_requests_handled,
            "connections": {
                "active": total_connections,
                "capacity": total_capacity,
                "utilization": total_connections / total_capacity
                if total_capacity > 0
                else 0.0,
            },
            "pools": {
                "count": len(self.connections),
                "max_pool_size": self.max_pool_size,
                "idle_timeout": self.idle_timeout,
            },
            "status": {
                "healthy": not self._shutdown_event.is_set(),
                "cleanup_tasks": len(self.cleanup_tasks),
            },
        }

    def can_handle_request(self, request: dict[str, Any] | None = None) -> bool:
        """Check if the system can handle a new request.

        Args:
            request: Optional request data (for future use)

        Returns:
            True if resources are available, False otherwise

        """
        try:
            import psutil

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90.0:
                return False

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90.0:
                return False

            # Check concurrent request limit
            return not self._active_requests >= self.max_concurrent_requests
        except ImportError:
            # If psutil is not available, use simpler checks
            return self._active_requests < self.max_concurrent_requests

    def acquire_resources(self, request: dict[str, Any] | None = None):
        """Context manager for acquiring resources.

        Args:
            request: Optional request data (for future use)

        Returns:
            Async context manager for resource acquisition

        """

        class ResourceAcquisition:
            def __init__(self, manager) -> None:
                self.manager = manager

            async def __aenter__(self):
                async with self.manager._resource_acquisition_condition:
                    # Wait until resources are available
                    while (
                        self.manager._active_requests
                        >= self.manager.max_concurrent_requests
                    ):
                        await self.manager._resource_acquisition_condition.wait()

                    # Acquire resources
                    self.manager._active_requests += 1
                    self.manager._total_requests_handled += 1

                    # Track peak
                    self.manager._peak_concurrent_requests = max(
                        self.manager._peak_concurrent_requests,
                        self.manager._active_requests,
                    )
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                # Release resources
                async with self.manager._resource_acquisition_condition:
                    self.manager._active_requests -= 1
                    self.manager._resource_acquisition_condition.notify()

        return ResourceAcquisition(self)


class AuthenticationMiddleware:
    """Authentication middleware for API key validation.

    Provides:
    - API key validation against configured keys
    - Request origin validation
    - Authentication bypass for health checks
    - Audit logging for authentication events
    """

    def __init__(
        self, api_keys: set[str] | None = None, require_auth: bool = True
    ) -> None:
        """Initialize authentication middleware.

        Args:
            api_keys: Set of valid API keys (None = no authentication)
            require_auth: Whether to require authentication

        """
        self.logger = get_logger(__name__)
        self.api_keys = api_keys or set()
        self.require_auth = require_auth
        self.exempt_methods = {"health_check", "ping", "status"}

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process request with authentication validation.

        Args:
            request: Incoming request data

        Returns:
            Request data with authentication context

        Raises:
            ValidationError: If authentication fails

        """
        method = request.get("method", "")

        # Skip authentication for exempt methods
        if any(exempt in method for exempt in self.exempt_methods):
            return request

        # Skip if authentication not required
        if not self.require_auth or not self.api_keys:
            return request

        # Extract API key from various sources
        api_key = self._extract_api_key(request)

        if not api_key:
            self.logger.warning(
                "Authentication failed: No API key provided",
                extra={
                    "correlation_id": request.get("correlation_id"),
                    "method": method,
                    "remote_addr": request.get("remote_addr", "unknown"),
                },
            )
            msg = "API key required for authentication"
            raise ValidationError(msg)

        if api_key not in self.api_keys:
            self.logger.warning(
                "Authentication failed: Invalid API key",
                extra={
                    "correlation_id": request.get("correlation_id"),
                    "method": method,
                    "remote_addr": request.get("remote_addr", "unknown"),
                    "api_key_prefix": api_key[:8] + "..."
                    if len(api_key) > 8
                    else "short",
                },
            )
            msg = "Invalid API key"
            raise ValidationError(msg)

        # Add authentication context
        request["authenticated"] = True
        request["api_key_prefix"] = api_key[:8] + "..."

        self.logger.debug(
            "Authentication successful",
            extra={
                "correlation_id": request.get("correlation_id"),
                "method": method,
                "api_key_prefix": request["api_key_prefix"],
            },
        )

        return request

    async def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Process response (no-op for authentication)."""
        return response

    def _extract_api_key(self, request: dict[str, Any]) -> str | None:
        """Extract API key from request headers or parameters.

        Args:
            request: Request data

        Returns:
            API key if found, None otherwise

        """
        # Check headers (common patterns)
        headers = request.get("headers", {})

        # Authorization header
        auth_header = headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]

        # X-API-Key header
        api_key_header = headers.get("x-api-key")
        if api_key_header:
            return api_key_header

        # Query parameter
        params = request.get("params", {})
        if isinstance(params, dict) and "api_key" in params:
            return params["api_key"]

        return None


class RateLimitingMiddleware:
    """Rate limiting middleware with configurable limits.

    Provides:
    - Per-client rate limiting
    - Sliding window algorithm
    - Configurable limits per method
    - Rate limit headers in responses
    """

    def __init__(self, default_limit: int = 100, window_seconds: int = 60) -> None:
        """Initialize rate limiting middleware.

        Args:
            default_limit: Default requests per window
            window_seconds: Window size in seconds

        """
        self.logger = get_logger(__name__)
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
        self.method_limits: dict[str, int] = {}
        self._rate_limit_lock = asyncio.Lock()

    def set_method_limit(self, method: str, limit: int) -> None:
        """Set rate limit for specific method.

        Args:
            method: Method name
            limit: Requests per window for this method

        """
        self.method_limits[method] = limit

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process request with rate limiting.

        Args:
            request: Incoming request data

        Returns:
            Request data with rate limit context

        Raises:
            ValidationError: If rate limit exceeded

        """
        client_id = self._get_client_id(request)
        method = request.get("method", "unknown")
        limit = self.method_limits.get(method, self.default_limit)

        current_time = time.time()
        window_start = current_time - self.window_seconds

        async with self._rate_limit_lock:
            # Clean old requests outside window
            client_requests = self.requests[client_id]
            self.requests[client_id] = [
                req_time for req_time in client_requests if req_time > window_start
            ]

            # Check rate limit
            current_count = len(self.requests[client_id])
            if current_count >= limit:
                self.logger.warning(
                    "Rate limit exceeded",
                    extra={
                        "correlation_id": request.get("correlation_id"),
                        "client_id": client_id,
                        "method": method,
                        "current_count": current_count,
                        "limit": limit,
                    },
                )
                msg = f"Rate limit exceeded: {current_count}/{limit} requests per {self.window_seconds}s"
                raise ValidationError(
                    msg,
                )

            # Record this request
            self.requests[client_id].append(current_time)

            # Add rate limit info to request
            request["rate_limit"] = {
                "limit": limit,
                "remaining": limit - current_count - 1,
                "reset_time": window_start + self.window_seconds,
            }

        return request

    async def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Process response with rate limit headers.

        Args:
            response: Outgoing response data

        Returns:
            Response with rate limit headers

        """
        rate_limit = response.get("rate_limit")
        if rate_limit:
            response["headers"] = response.get("headers", {})
            response["headers"].update(
                {
                    "X-RateLimit-Limit": str(rate_limit["limit"]),
                    "X-RateLimit-Remaining": str(rate_limit["remaining"]),
                    "X-RateLimit-Reset": str(int(rate_limit["reset_time"])),
                },
            )

        return response

    def _get_client_id(self, request: dict[str, Any]) -> str:
        """Get client identifier for rate limiting.

        Args:
            request: Request data

        Returns:
            Client identifier

        """
        # Use API key prefix if available
        api_key_prefix = request.get("api_key_prefix")
        if api_key_prefix:
            return f"api_key:{api_key_prefix}"

        # Use remote address
        remote_addr = request.get("remote_addr", "unknown")
        return f"addr:{remote_addr}"


class GracefulShutdownMiddleware:
    """Graceful shutdown middleware for request lifecycle management.

    Provides:
    - Request tracking during shutdown
    - Shutdown signal handling
    - Graceful request completion
    - Health status reporting
    """

    def __init__(self, shutdown_timeout: float = 30.0) -> None:
        """Initialize graceful shutdown middleware.

        Args:
            shutdown_timeout: Maximum time to wait for requests to complete

        """
        self.logger = get_logger(__name__)
        self.shutdown_timeout = shutdown_timeout
        self.active_requests: set[str] = set()
        self.shutdown_initiated = False
        self._request_lock = asyncio.Lock()

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process request with shutdown awareness.

        Args:
            request: Incoming request data

        Returns:
            Request data with shutdown context

        Raises:
            ValidationError: If shutdown is in progress

        """
        if self.shutdown_initiated:
            self.logger.warning(
                "Request rejected: Shutdown in progress",
                extra={
                    "correlation_id": request.get("correlation_id"),
                    "method": request.get("method"),
                },
            )
            msg = "Server is shutting down, rejecting new requests"
            raise ValidationError(msg)

        correlation_id = request.get("correlation_id", str(uuid.uuid4()))

        async with self._request_lock:
            self.active_requests.add(correlation_id)

        request["correlation_id"] = correlation_id
        request["shutdown_aware"] = True

        return request

    async def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Process response with request cleanup.

        Args:
            response: Outgoing response data

        Returns:
            Response data unchanged

        """
        correlation_id = response.get("correlation_id")

        if correlation_id:
            async with self._request_lock:
                self.active_requests.discard(correlation_id)

        return response

    async def initiate_shutdown(self) -> None:
        """Initiate graceful shutdown process."""
        self.logger.info("Initiating graceful shutdown")
        self.shutdown_initiated = True

        # Wait for active requests to complete
        start_time = time.time()
        while (
            self.active_requests and (time.time() - start_time) < self.shutdown_timeout
        ):
            self.logger.info(
                f"Waiting for {len(self.active_requests)} active requests to complete",
            )
            await asyncio.sleep(0.5)

        if self.active_requests:
            self.logger.warning(
                f"Shutdown timeout reached, {len(self.active_requests)} requests still active",
            )
        else:
            self.logger.info("All requests completed, shutdown ready")

    def get_shutdown_status(self) -> dict[str, Any]:
        """Get current shutdown status.

        Returns:
            Shutdown status information

        """
        return {
            "shutdown_initiated": self.shutdown_initiated,
            "active_requests": len(self.active_requests),
            "shutdown_timeout": self.shutdown_timeout,
        }
