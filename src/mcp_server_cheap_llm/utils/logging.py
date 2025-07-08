"""Structured logging configuration for MCP Server Cheap LLM.

This module sets up structured logging using structlog for consistent
and parseable log output. Follows atomic design principles (100-200 lines).

Key functions:
    setup_logging: Configure structured logging
    get_logger: Get a logger instance
    log_request: Log request/response pairs
    log_error: Log errors with context

Example:
    >>> setup_logging(debug=True)
    >>> logger = get_logger(__name__)
    >>> logger.info("Server started", port=8080)
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.processors import JSONRenderer, TimeStamper
from structlog.stdlib import LoggerFactory, filter_by_level


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
    error: Optional[str] = None,
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
    context: Optional[Dict[str, Any]] = None,
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
        log_data["context"] = context

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
