"""Structured logging configuration for MCP Server Cheap LLM.

This module sets up structured logging using structlog for consistent
and parseable log output. Follows atomic design principles (100-200 lines).

Key classes:
    LogContext: Thread-safe context manager for correlation IDs
    StructuredLogger: Enhanced logger with JSON formatting and correlation ID injection

Key functions:
    setup_logging: Configure structured logging
    get_logger: Get a logger instance
    log_request: Log request/response pairs
    log_error: Log errors with context

Example:
    >>> setup_logging(debug=True)
    >>> logger = get_logger(__name__)
    >>> logger.info("Server started", port=8080)
    >>>
    >>> # New structured logging with correlation IDs
    >>> with LogContext() as context:
    ...     structured_logger = StructuredLogger("my_logger")
    ...     structured_logger.info("Processing request", user_id="123")
"""

import json
import logging
import os
import sys
import time
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog  # type: ignore[import-not-found]
from structlog.processors import (  # type: ignore[import-not-found]
    JSONRenderer,
    TimeStamper,
)
from structlog.stdlib import (  # type: ignore[import-not-found]
    LoggerFactory,
    filter_by_level,
)

# Thread-local storage for correlation IDs
_correlation_id_context: ContextVar[str | None] = ContextVar(
    "correlation_id", default=None
)


class LogContext:
    """Thread-safe context manager for correlation IDs.

    This class provides correlation ID management for request tracking across
    threads and async contexts. Each context maintains a unique correlation ID
    that can be used to trace related log entries.

    Attributes:
        correlation_id: Unique string identifier for this context

    Example:
        >>> with LogContext() as context:
        ...     print(f"Request ID: {context.correlation_id}")
        ...     # All logging within this context will include the correlation ID
    """

    def __init__(self, correlation_id: str | None = None):
        """Initialize LogContext with optional correlation ID.

        Args:
            correlation_id: Optional existing correlation ID to use.
                           If None, a new UUID will be generated.
        """
        self._correlation_id = correlation_id or str(uuid4())
        self._token: Any = None

    @property
    def correlation_id(self) -> str:
        """Get the correlation ID for this context.

        Returns:
            The correlation ID string
        """
        return self._correlation_id

    def __enter__(self) -> "LogContext":
        """Enter the context and set the correlation ID.

        Returns:
            Self for use in with statement
        """
        self._token = _correlation_id_context.set(self._correlation_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and reset the correlation ID.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        if self._token is not None:
            _correlation_id_context.reset(self._token)


class StructuredLogger:
    """Enhanced logger with JSON formatting and correlation ID injection.

    This logger automatically injects correlation IDs from the current LogContext
    and formats messages as JSON for structured logging. It provides thread-safe
    logging with performance optimization.

    Attributes:
        name: Logger name
        level: Current log level

    Example:
        >>> logger = StructuredLogger("my_module")
        >>> with LogContext() as context:
        ...     logger.info("Processing request", user_id="123")
        ...     # Output will include correlation_id automatically
    """

    def __init__(self, name: str):
        """Initialize StructuredLogger with name and configuration.

        Args:
            name: Logger name (typically module name)
        """
        self.name = name
        self._logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Set up the underlying logger with proper configuration."""
        # Get log level from environment or default to INFO
        level_name = os.getenv("MCP_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        self.level = level
        self._logger.setLevel(level)

        # Ensure we have a handler for output
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            # Prevent double-logging by setting propagate to False
            self._logger.propagate = False
            self._logger.addHandler(handler)

    def _format_log_entry(self, level: str, message: str, **kwargs) -> str:
        """Format log entry as JSON with correlation ID and metadata.

        Args:
            level: Log level string
            message: Log message
            **kwargs: Additional fields to include

        Returns:
            JSON formatted log entry
        """
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "logger": self.name,
            "message": message,
        }

        # Add correlation ID if available (performance optimized)
        correlation_id = _correlation_id_context.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add custom fields with improved serialization handling
        for key, value in kwargs.items():
            # Handle common types efficiently
            if isinstance(value, str | int | float | bool | type(None)):
                log_entry[key] = value
            elif isinstance(value, list | dict):
                # Try to serialize complex types
                try:
                    # Quick check if it's JSON serializable
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)
            else:
                # Convert other types to string
                log_entry[key] = str(value)

        try:
            return json.dumps(log_entry, separators=(",", ":"))  # Compact JSON
        except (TypeError, ValueError):
            # Fallback to string representation
            return str(log_entry)

    def debug(self, message: str, **kwargs):
        """Log debug message with correlation ID.

        Args:
            message: Log message
            **kwargs: Additional fields to include
        """
        if self._logger.isEnabledFor(logging.DEBUG):
            formatted_msg = self._format_log_entry("DEBUG", message, **kwargs)
            self._logger.debug(formatted_msg)

    def info(self, message: str, **kwargs):
        """Log info message with correlation ID.

        Args:
            message: Log message
            **kwargs: Additional fields to include
        """
        if self._logger.isEnabledFor(logging.INFO):
            formatted_msg = self._format_log_entry("INFO", message, **kwargs)
            self._logger.info(formatted_msg)

    def warning(self, message: str, **kwargs):
        """Log warning message with correlation ID.

        Args:
            message: Log message
            **kwargs: Additional fields to include
        """
        if self._logger.isEnabledFor(logging.WARNING):
            formatted_msg = self._format_log_entry("WARNING", message, **kwargs)
            self._logger.warning(formatted_msg)

    def error(self, message: str, **kwargs):
        """Log error message with correlation ID.

        Args:
            message: Log message
            **kwargs: Additional fields to include
        """
        if self._logger.isEnabledFor(logging.ERROR):
            formatted_msg = self._format_log_entry("ERROR", message, **kwargs)
            self._logger.error(formatted_msg)

    def critical(self, message: str, **kwargs):
        """Log critical message with correlation ID.

        Args:
            message: Log message
            **kwargs: Additional fields to include
        """
        if self._logger.isEnabledFor(logging.CRITICAL):
            formatted_msg = self._format_log_entry("CRITICAL", message, **kwargs)
            self._logger.critical(formatted_msg)


def setup_logging(debug: bool = False, json_output: bool = False) -> None:
    """Configure structured logging for the application.

    Sets up structlog with appropriate processors for development
    or production environments.

    Args:
        debug: Enable debug level logging
        json_output: Use JSON format for log output

    Example:
        >>> setup_logging(debug=True, json_output=False)
        >>> # Logs will be colorized and human-readable
    """
    log_level = logging.DEBUG if debug else logging.INFO

    # Configure standard library logging
    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stdout)

    # Choose processors based on environment
    processors = [
        filter_by_level,
        TimeStamper(fmt="ISO", utc=True),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        # Production: JSON output for parsing
        processors.append(JSONRenderer())
    else:
        # Development: Human-readable output
        processors.extend(
            [
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Operation completed", duration_ms=150)
    """
    return structlog.get_logger(name)


def log_request_response(
    logger: structlog.stdlib.BoundLogger,
    request_id: str,
    provider: str,
    prompt_length: int,
    response_length: int,
    duration_ms: int,
    success: bool,
    error: str | None = None,
) -> None:
    """Log request/response pair with structured data.

    Args:
        logger: Logger instance
        request_id: Unique request identifier
        provider: Name of the LLM provider
        prompt_length: Length of input prompt
        response_length: Length of response
        duration_ms: Request duration in milliseconds
        success: Whether request succeeded
        error: Error message if request failed

    Example:
        >>> log_request_response(
        ...     logger, "req_123", "gemini", 50, 200, 1500, True
        ... )
    """
    log_data = {
        "request_id": request_id,
        "provider": provider,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "duration_ms": duration_ms,
        "success": success,
    }

    if error:
        log_data["error"] = error
        logger.error("Request failed", **log_data)
    else:
        logger.info("Request completed", **log_data)


def log_provider_status(
    logger: structlog.stdlib.BoundLogger,
    provider: str,
    status: str,
    total_requests: int,
    success_rate: float,
    avg_response_time: float,
) -> None:
    """Log provider status information.

    Args:
        logger: Logger instance
        provider: Provider name
        status: Current status
        total_requests: Total requests processed
        success_rate: Success rate percentage
        avg_response_time: Average response time in ms

    Example:
        >>> log_provider_status(
        ...     logger, "gemini", "available", 100, 95.5, 1200.0
        ... )
    """
    logger.info(
        "Provider status update",
        provider=provider,
        status=status,
        total_requests=total_requests,
        success_rate=round(success_rate, 2),
        avg_response_time_ms=round(avg_response_time, 2),
    )


def log_security_event(
    logger: structlog.stdlib.BoundLogger,
    event_type: str,
    description: str,
    source: str,
    context: dict[str, Any] | None = None,
) -> None:
    """Log security-related events.

    Args:
        logger: Logger instance
        event_type: Type of security event
        description: Event description
        source: Source of the event
        context: Additional context information

    Example:
        >>> log_security_event(
        ...     logger, "command_blocked", "Unsafe command detected",
        ...     "user_input", {"command": "rm -rf /"}
        ... )
    """
    log_data = {
        "event_type": event_type,
        "description": description,
        "source": source,
    }

    if context:
        log_data.update(context)

    logger.warning("Security event", **log_data)


def log_configuration_loaded(
    logger: structlog.stdlib.BoundLogger,
    config_source: str,
    provider_count: int,
    enabled_providers: list,
    default_provider: str,
) -> None:
    """Log configuration loading information.

    Args:
        logger: Logger instance
        config_source: Source of configuration (file/env/default)
        provider_count: Total number of providers configured
        enabled_providers: List of enabled provider names
        default_provider: Default provider name

    Example:
        >>> log_configuration_loaded(
        ...     logger, "config.toml", 3, ["gemini", "codex"], "gemini"
        ... )
    """
    logger.info(
        "Configuration loaded",
        config_source=config_source,
        provider_count=provider_count,
        enabled_providers=enabled_providers,
        default_provider=default_provider,
    )
