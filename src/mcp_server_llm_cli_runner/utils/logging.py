"""Structured logging configuration for MCP Server LLM CLI Runner.

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
import re
import sys
import threading
import time
from collections import deque
from contextvars import ContextVar
from datetime import datetime, timedelta
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

from ..utils.errors import LLMCliRunnerError

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

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Exit the context and reset the correlation ID.

        Args:
            _exc_type: Exception type (unused)
            _exc_val: Exception value (unused)
            _exc_tb: Exception traceback (unused)
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

        # Initialize event storage for security logging compatibility
        self._events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

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

    def exception(self, message: str, **kwargs):
        """Log exception with traceback and correlation ID.

        Args:
            message: Log message
            **kwargs: Additional fields to include
        """
        if self._logger.isEnabledFor(logging.ERROR):
            # Include exception info automatically
            kwargs["exc_info"] = True
            formatted_msg = self._format_log_entry("ERROR", message, **kwargs)
            self._logger.error(formatted_msg)

    def log_security_event(
        self,
        event_type: str,
        description: str,
        source: str,
        error: Exception | None = None,
        severity: str = "medium",
        **kwargs,
    ) -> None:
        """Log a security event with structured data for test compatibility.

        This method provides compatibility with security logging tests while
        maintaining the structured logging approach of this class.

        Args:
            event_type: Type of security event (e.g., "unauthorized_access")
            description: Human-readable description
            source: Source of the event (e.g., "api_endpoint")
            error: Optional exception object for error details
            severity: Event severity ("low", "medium", "high")
            **kwargs: Additional context fields
        """
        # Build event data similar to SecurityLogger but simpler
        event_data = {
            "timestamp": time.time(),
            "event_type": event_type,
            "description": description,
            "source": source,
            "severity": severity,
            "logger": self.name,
        }

        # Add correlation ID if available
        correlation_id = _correlation_id_context.get()
        if correlation_id:
            event_data["correlation_id"] = correlation_id

        # Add error details if provided
        if error:
            event_data["error_type"] = type(error).__name__
            event_data["error_message"] = str(error)

            # Add error code if available (for custom exceptions)
            if isinstance(error, LLMCliRunnerError) and error.error_code:
                event_data["error_code"] = error.error_code

        # Add additional context
        for key, value in kwargs.items():
            event_data[key] = value

        # Store event for test compatibility (thread-safe)
        with self._events_lock:
            self._events.append(event_data.copy())

        # Log the event using warning level for security events
        if self._logger.isEnabledFor(logging.WARNING):
            formatted_msg = self._format_log_entry(
                "WARNING", "Security event detected", **event_data
            )
            self._logger.warning(formatted_msg)


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


class SecurityLogger:
    """Security-focused logger with error handling integration and monitoring.

    This logger provides specialized security event logging, sensitive data filtering,
    error rate monitoring with configurable thresholds, and alert generation.
    Integrates with existing logging infrastructure while adding security-specific
    functionality.

    Attributes:
        name: Logger name
        error_rate_threshold: Maximum errors allowed per time window
        threshold_window_minutes: Time window for error rate calculation
        enable_sensitive_data_filtering: Whether to filter sensitive data

    Example:
        >>> security_logger = SecurityLogger("security", error_rate_threshold=5)
        >>> security_logger.log_security_event(
        ...     event_type="unauthorized_access",
        ...     description="Failed auth attempt",
        ...     source="api_endpoint"
        ... )
    """

    # Sensitive data patterns for filtering
    SENSITIVE_PATTERNS = {
        "api_key": re.compile(r"sk-[a-zA-Z0-9]{32,}|[a-zA-Z0-9]{32,}"),
        "password": re.compile(r"password|passwd|pwd", re.IGNORECASE),
        "token": re.compile(
            r"bearer[_\s]+[a-zA-Z0-9]+|token[_\s]*[:=][_\s]*[a-zA-Z0-9]+", re.IGNORECASE
        ),
        "secret": re.compile(r"secret[_\s]*[:=][_\s]*[a-zA-Z0-9]+", re.IGNORECASE),
        "auth": re.compile(r"authorization[_\s]*[:=][_\s]*[a-zA-Z0-9]+", re.IGNORECASE),
    }

    def __init__(
        self,
        name: str,
        error_rate_threshold: int = 10,
        threshold_window_minutes: int = 5,
        enable_sensitive_data_filtering: bool = True,
    ):
        """Initialize SecurityLogger with configuration.

        Args:
            name: Logger name (typically module name)
            error_rate_threshold: Maximum errors allowed per time window
            threshold_window_minutes: Time window for error rate calculation in minutes
            enable_sensitive_data_filtering: Whether to filter sensitive data from logs
        """
        self.name = name
        self.error_rate_threshold = error_rate_threshold
        self.threshold_window_minutes = threshold_window_minutes
        self.enable_sensitive_data_filtering = enable_sensitive_data_filtering

        # Initialize logging infrastructure
        self._structured_logger = StructuredLogger(f"security.{name}")

        # Event storage for testing and monitoring
        self._events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        # Error rate monitoring
        self._error_timestamps: deque = deque()
        self._error_lock = threading.Lock()
        self._threshold_exceeded = False

    def log_security_event(
        self,
        event_type: str,
        description: str,
        source: str,
        error: Exception | None = None,
        severity: str = "medium",
        **kwargs,
    ) -> None:
        """Log a security event with optional error integration.

        Args:
            event_type: Type of security event (e.g., "unauthorized_access")
            description: Human-readable description
            source: Source of the event (e.g., "api_endpoint")
            error: Optional exception object for error details
            severity: Event severity ("low", "medium", "high")
            **kwargs: Additional context fields
        """
        # Build event data
        event_data = {
            "timestamp": time.time(),
            "event_type": event_type,
            "description": description,
            "source": source,
            "severity": severity,
            "logger": self.name,
        }

        # Add correlation ID if available
        correlation_id = _correlation_id_context.get()
        if correlation_id:
            event_data["correlation_id"] = correlation_id

        # Add error details if provided
        if error:
            event_data["error_type"] = type(error).__name__
            event_data["error_message"] = str(error)

            # Add error code if available (for custom exceptions)
            if isinstance(error, LLMCliRunnerError) and error.error_code:
                event_data["error_code"] = error.error_code

            # Add filtered context if available (for custom exceptions)
            if isinstance(error, LLMCliRunnerError) and error.context:
                filtered_context = self._filter_sensitive_data(error.context)
                event_data["error_context"] = filtered_context

        # Add and filter additional context
        for key, value in kwargs.items():
            if self.enable_sensitive_data_filtering:
                event_data[key] = self._filter_sensitive_data({key: value})[key]
            else:
                event_data[key] = value

        # Store event for monitoring
        with self._events_lock:
            self._events.append(event_data.copy())

        # Log the event
        self._structured_logger.warning("Security event detected", **event_data)

    def log_error_with_filtering(self, error: Exception) -> None:
        """Log an error with sensitive data filtering.

        Args:
            error: Exception to log with filtering applied
        """
        error_data: dict[str, Any] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        # Add error code if available (for custom exceptions)
        if isinstance(error, LLMCliRunnerError) and error.error_code:
            error_data["error_code"] = error.error_code

        # Add filtered context if available (for custom exceptions)
        if isinstance(error, LLMCliRunnerError) and error.context:
            filtered_context = self._filter_sensitive_data(error.context)
            error_data["error_context"] = filtered_context

        # Store as security event for monitoring
        event_data = {
            "timestamp": time.time(),
            "event_type": "error_logged",
            "description": f"Error logged with filtering: {type(error).__name__}",
            "source": "error_handler",
            "severity": self.detect_security_severity(error),
            "logger": self.name,
            **error_data,
        }

        # Add correlation ID if available
        correlation_id = _correlation_id_context.get()
        if correlation_id:
            event_data["correlation_id"] = correlation_id

        with self._events_lock:
            self._events.append(event_data.copy())

        # Log the filtered error
        self._structured_logger.error("Filtered error logged", **error_data)

    def monitor_error_rate(self, error: Exception) -> None:
        """Monitor error rate and trigger alerts if threshold exceeded.

        Args:
            error: Exception to count towards error rate monitoring
        """
        current_time = datetime.now()

        with self._error_lock:
            # Add current error timestamp
            self._error_timestamps.append(current_time)

            # Remove old timestamps outside the window
            cutoff_time = current_time - timedelta(
                minutes=self.threshold_window_minutes
            )
            while self._error_timestamps and self._error_timestamps[0] < cutoff_time:
                self._error_timestamps.popleft()

            # Check if threshold is exceeded
            if len(self._error_timestamps) >= self.error_rate_threshold:
                if not self._threshold_exceeded:
                    self._threshold_exceeded = True
                    self._trigger_error_rate_alert(len(self._error_timestamps))

    def detect_security_severity(self, error: Exception) -> str:
        """Detect security severity level based on error type.

        Args:
            error: Exception to analyze for security severity

        Returns:
            Severity level: "low", "medium", or "high"
        """
        from ..utils.errors import (
            ConfigurationError,
            LLMCliRunnerError,
            ProviderError,
            SecurityError,
            ValidationError,
        )

        if isinstance(error, SecurityError):
            return "high"
        elif isinstance(error, ValidationError | ProviderError):
            return "medium"
        elif isinstance(error, ConfigurationError | LLMCliRunnerError):
            return "low"
        else:
            return "low"

    def is_threshold_exceeded(self) -> bool:
        """Check if error rate threshold has been exceeded.

        Returns:
            True if threshold exceeded, False otherwise
        """
        return self._threshold_exceeded

    def detect_sensitive_data(self, data: dict[str, Any]) -> bool:
        """Detect if data contains sensitive information.

        Args:
            data: Dictionary to check for sensitive data

        Returns:
            True if sensitive data detected, False otherwise
        """
        for key, value in data.items():
            key_lower = key.lower()
            value_str = str(value).lower() if value is not None else ""

            # Check if key or value contains sensitive patterns
            for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
                if (
                    pattern_name in key_lower
                    or pattern.search(key_lower)
                    or pattern.search(value_str)
                ):
                    return True

        return False

    def _filter_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Filter sensitive data from a dictionary.

        Args:
            data: Dictionary potentially containing sensitive data

        Returns:
            Dictionary with sensitive data filtered/masked
        """
        if not self.enable_sensitive_data_filtering:
            return data

        filtered_data = {}

        for key, value in data.items():
            key_lower = key.lower()
            value_str = str(value).lower() if value is not None else ""

            # Check if key or value contains sensitive patterns
            is_sensitive = False

            for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
                if (
                    pattern_name in key_lower
                    or pattern.search(key_lower)
                    or pattern.search(value_str)
                ):
                    is_sensitive = True
                    break

            if is_sensitive:
                # Mask sensitive data
                filtered_data[key] = "[FILTERED]"
            else:
                # Preserve non-sensitive data
                filtered_data[key] = value

        return filtered_data

    def _trigger_error_rate_alert(self, error_count: int) -> None:
        """Trigger alert when error rate threshold is exceeded.

        Args:
            error_count: Current number of errors in the time window
        """
        alert_data = {
            "timestamp": time.time(),
            "event_type": "error_rate_threshold_exceeded",
            "description": f"Error rate threshold exceeded: {error_count} errors in {self.threshold_window_minutes} minutes",
            "source": "error_rate_monitor",
            "severity": "high",
            "logger": self.name,
            "error_count": error_count,
            "threshold": self.error_rate_threshold,
            "window_minutes": self.threshold_window_minutes,
        }

        # Add correlation ID if available
        correlation_id = _correlation_id_context.get()
        if correlation_id:
            alert_data["correlation_id"] = correlation_id

        # Store alert event
        with self._events_lock:
            self._events.append(alert_data.copy())

        # Log the alert
        self._structured_logger.critical("Error rate threshold exceeded", **alert_data)


class PerformanceLogger:
    """Performance monitoring logger with timing decorators and metrics aggregation.

    This logger provides timing decorators for function performance monitoring,
    metric aggregation, and statistical analysis. Integrates with structured logging
    infrastructure and correlation ID tracking.

    Attributes:
        name: Logger name
        metrics: List of timing metrics

    Example:
        >>> perf_logger = PerformanceLogger("performance")
        >>> @perf_logger.time_function
        ... def my_function():
        ...     time.sleep(0.1)
        ...     return "result"
        >>> result = my_function()
        >>> stats = perf_logger.get_aggregated_stats()
    """

    def __init__(self, name: str):
        """Initialize PerformanceLogger with name.

        Args:
            name: Logger name (typically module name)
        """
        self.name = name
        self._metrics: list[dict[str, Any]] = []
        self._metrics_lock = threading.Lock()
        self._structured_logger = StructuredLogger(f"performance.{name}")

    def time_function(self, func):
        """Decorator to time function execution and log metrics.

        Args:
            func: Function to time

        Returns:
            Decorated function with timing functionality
        """

        def wrapper(*args, **kwargs):
            start_time = time.time()
            correlation_id = _correlation_id_context.get()
            success = True
            error_msg = None
            result = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000

                # Create timing metric
                metric = {
                    "timestamp": start_time,
                    "function_name": func.__name__,
                    "duration_ms": duration_ms,
                    "success": success,
                    "logger": self.name,
                }

                if correlation_id:
                    metric["correlation_id"] = correlation_id

                if error_msg:
                    metric["error"] = error_msg

                # Store metric
                with self._metrics_lock:
                    self._metrics.append(metric)

                # Log the performance metric
                self._structured_logger.info("Function performance metric", **metric)

        return wrapper

    def get_timing_metrics(self) -> list[dict[str, Any]]:
        """Get all timing metrics.

        Returns:
            List of timing metrics
        """
        with self._metrics_lock:
            return self._metrics.copy()

    def get_aggregated_stats(self) -> dict[str, dict[str, Any]]:
        """Get aggregated performance statistics by function.

        Returns:
            Dictionary of function statistics
        """
        with self._metrics_lock:
            stats = {}

            # Group metrics by function name
            function_metrics = {}
            for metric in self._metrics:
                func_name = metric["function_name"]
                if func_name not in function_metrics:
                    function_metrics[func_name] = []
                function_metrics[func_name].append(metric)

            # Calculate statistics for each function
            for func_name, metrics in function_metrics.items():
                durations = [m["duration_ms"] for m in metrics]
                successes = [m["success"] for m in metrics]

                # Basic statistics
                call_count = len(metrics)
                avg_duration = sum(durations) / call_count if call_count > 0 else 0
                min_duration = min(durations) if durations else 0
                max_duration = max(durations) if durations else 0
                success_rate = sum(successes) / call_count if call_count > 0 else 0

                # Standard deviation
                if call_count > 1:
                    variance = sum((d - avg_duration) ** 2 for d in durations) / (
                        call_count - 1
                    )
                    std_deviation = variance**0.5
                else:
                    std_deviation = 0

                stats[func_name] = {
                    "call_count": call_count,
                    "avg_duration_ms": avg_duration,
                    "min_duration_ms": min_duration,
                    "max_duration_ms": max_duration,
                    "std_deviation_ms": std_deviation,
                    "success_rate": success_rate,
                }

            return stats


class AuditLogger:
    """Audit trail logger with security event detection and classification.

    This logger provides comprehensive audit trail generation, security event
    detection, and event classification. Integrates with structured logging
    infrastructure and correlation ID tracking.

    Attributes:
        name: Logger name
        audit_trail: List of audit events

    Example:
        >>> audit_logger = AuditLogger("audit")
        >>> audit_logger.log_audit_event(
        ...     event_type="user_action",
        ...     description="User logged in",
        ...     user_id="user123"
        ... )
        >>> trail = audit_logger.get_audit_trail()
    """

    # Security event types that should be classified as security events
    SECURITY_EVENT_TYPES = {
        "security_violation",
        "unauthorized_access",
        "privilege_escalation",
        "failed_login",
        "data_breach",
        "suspicious_activity",
        "authentication_failure",
        "authorization_failure",
    }

    def __init__(self, name: str):
        """Initialize AuditLogger with name.

        Args:
            name: Logger name (typically module name)
        """
        self.name = name
        self._audit_trail: list[dict[str, Any]] = []
        self._audit_lock = threading.Lock()
        self._structured_logger = StructuredLogger(f"audit.{name}")

    def log_audit_event(self, event_type: str, description: str = "", **kwargs) -> None:
        """Log an audit event with automatic security classification.

        Args:
            event_type: Type of audit event
            description: Human-readable description
            **kwargs: Additional context fields
        """
        # Create audit event
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "description": description,
            "logger": self.name,
        }

        # Add correlation ID if available
        correlation_id = _correlation_id_context.get()
        if correlation_id:
            event["correlation_id"] = correlation_id

        # Add additional context
        for key, value in kwargs.items():
            event[key] = value

        # Classify as security event
        event["is_security_event"] = self._is_security_event(event_type, kwargs)

        # Add default severity if not provided
        if "severity" not in event:
            event["severity"] = self._classify_severity(event_type, kwargs)

        # Store in audit trail
        with self._audit_lock:
            self._audit_trail.append(event)

        # Log the audit event
        self._structured_logger.info("Audit event logged", **event)

    def get_audit_trail(self) -> list[dict[str, Any]]:
        """Get complete audit trail.

        Returns:
            List of audit events in chronological order
        """
        with self._audit_lock:
            return sorted(self._audit_trail, key=lambda x: x["timestamp"])

    def get_security_events(self, severity: str | None = None) -> list[dict[str, Any]]:
        """Get security events, optionally filtered by severity.

        Args:
            severity: Optional severity filter ("low", "medium", "high")

        Returns:
            List of security events
        """
        with self._audit_lock:
            security_events = [
                event
                for event in self._audit_trail
                if event.get("is_security_event", False)
            ]

            if severity:
                security_events = [
                    event
                    for event in security_events
                    if event.get("severity") == severity
                ]

            return sorted(security_events, key=lambda x: x["timestamp"])

    def _is_security_event(self, event_type: str, context: dict[str, Any]) -> bool:
        """Determine if event is security-related.

        Args:
            event_type: Type of event
            context: Additional context

        Returns:
            True if security event, False otherwise
        """
        # Check if event type is explicitly security-related
        if event_type in self.SECURITY_EVENT_TYPES:
            return True

        # Check for security-related context
        security_keywords = ["security", "auth", "unauthorized", "violation", "breach"]

        # Check event type for security keywords
        if any(keyword in event_type.lower() for keyword in security_keywords):
            return True

        # Check description for security keywords
        description = context.get("description", "")
        if any(keyword in description.lower() for keyword in security_keywords):
            return True

        # Check for explicit severity indication
        severity = context.get("severity", "").lower()
        if severity in ["high", "critical"]:
            return True

        return False

    def _classify_severity(self, event_type: str, context: dict[str, Any]) -> str:
        """Classify event severity based on type and context.

        Args:
            event_type: Type of event
            context: Additional context

        Returns:
            Severity level: "low", "medium", or "high"
        """
        # Check if severity is explicitly provided
        if "severity" in context:
            return context["severity"]

        # High severity events
        high_severity_events = {
            "security_violation",
            "unauthorized_access",
            "privilege_escalation",
            "data_breach",
        }

        if event_type in high_severity_events:
            return "high"

        # Medium severity events
        medium_severity_events = {
            "failed_login",
            "authentication_failure",
            "authorization_failure",
        }

        if event_type in medium_severity_events:
            return "medium"

        # Default to low severity
        return "low"
