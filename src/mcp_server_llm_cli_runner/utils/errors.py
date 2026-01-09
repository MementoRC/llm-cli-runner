"""Error serialization utilities for MCP protocol compliance.

This module provides ErrorSerializer class for converting custom exceptions
to MCP protocol format and deserializing them back to exception objects.
"""

import json
import uuid
from datetime import datetime
from typing import Any

from mcp_server_llm_cli_runner.core.errors import (
    ConfigurationError,
    LLMCliRunnerError,
    ProviderError,
    RateLimitError,
    SecurityError,
    ValidationError,
)


class ErrorSerializer:
    """Serializes and deserializes errors for MCP protocol compliance.

    This class handles conversion between custom exception objects and
    MCP protocol JSON format, preserving error context and metadata.

    MCP Protocol Format:
    {
        "error": {
            "code": "error_code",
            "message": "error_message",
            "data": {
                "error_type": "ExceptionClassName",
                "context": {...},
                "cause": "...",
                "timestamp": "..."
            }
        }
    }
    """

    # Error type to class mapping for deserialization
    ERROR_CLASS_MAP: dict[str, type[LLMCliRunnerError]] = {
        "LLMCliRunnerError": LLMCliRunnerError,
        "ConfigurationError": ConfigurationError,
        "ProviderError": ProviderError,
        "ValidationError": ValidationError,
        "RateLimitError": RateLimitError,
        "SecurityError": SecurityError,
    }

    # Default error code prefixes by error type
    DEFAULT_ERROR_CODES: dict[str, str] = {
        "LLMCliRunnerError": "GEN",
        "ConfigurationError": "CFG",
        "ProviderError": "PRV",
        "ValidationError": "VAL",
        "RateLimitError": "RLT",
        "SecurityError": "SEC",
    }

    def serialize(self, error: LLMCliRunnerError) -> dict[str, Any]:
        """Serialize an exception to MCP protocol format.

        Args:
            error: The exception to serialize

        Returns:
            Dictionary in MCP protocol format

        """
        error_code = self._get_error_code(error)
        error_message = str(error)
        error_data = self._extract_error_data(error)

        return {
            "error": {"code": error_code, "message": error_message, "data": error_data},
        }

    def deserialize(self, mcp_error: dict[str, Any]) -> LLMCliRunnerError:
        """Deserialize MCP protocol error back to exception object.

        Args:
            mcp_error: Dictionary in MCP protocol format

        Returns:
            Exception object reconstructed from MCP format

        """
        error_info = mcp_error.get("error", {})
        error_code = error_info.get("code")
        error_message = error_info.get("message", "Unknown error")
        error_data = error_info.get("data", {})

        # Extract error type and context
        error_type = error_data.get("error_type", "LLMCliRunnerError")
        context = error_data.get("context", {})

        # Get the appropriate exception class
        error_class = self.ERROR_CLASS_MAP.get(error_type, LLMCliRunnerError)

        # Create exception instance based on type
        if error_type == "ProviderError":
            provider = error_data.get("provider", "unknown")
            return ProviderError(
                error_message,
                provider=provider,
                error_code=error_code,
                context=context,
            )
        if error_type == "RateLimitError":
            provider = error_data.get("provider", "unknown")
            retry_after = error_data.get("retry_after", 0)
            return RateLimitError(
                error_message,
                provider=provider,
                retry_after=retry_after,
                error_code=error_code,
                context=context,
            )
        return error_class(error_message, error_code=error_code, context=context)

    def _get_error_code(self, error: LLMCliRunnerError) -> str:
        """Extract or generate error code for the exception.

        Args:
            error: The exception to get error code for

        Returns:
            Error code string

        """
        # If error has explicit error_code, use it
        if hasattr(error, "error_code") and error.error_code:
            return error.error_code

        # Generate default error code based on error type
        error_type = error.__class__.__name__
        prefix = self.DEFAULT_ERROR_CODES.get(error_type, "GEN")
        unique_id = str(uuid.uuid4())[:8].upper()

        return f"{prefix}{unique_id}"

    def _extract_error_data(self, error: LLMCliRunnerError) -> dict[str, Any]:
        """Extract comprehensive error data for serialization.

        Args:
            error: The exception to extract data from

        Returns:
            Dictionary containing error metadata

        """
        error_data: dict[str, Any] = {
            "error_type": error.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
        }

        # Add context if available
        if hasattr(error, "context") and error.context:
            error_data["context"] = self._serialize_context(error.context)
        else:
            error_data["context"] = {}

        # Add provider-specific data
        if isinstance(error, ProviderError):
            error_data["provider"] = error.provider

        if isinstance(error, RateLimitError):
            error_data["retry_after"] = error.retry_after

        # Add error chaining information
        if error.__cause__:
            error_data["cause"] = (
                f"{error.__cause__.__class__.__name__}: {error.__cause__!s}"
            )

        return error_data

    def _serialize_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Serialize context data, handling special types.

        Args:
            context: Raw context dictionary

        Returns:
            JSON-serializable context dictionary

        """
        serialized = {}

        for key, value in context.items():
            try:
                # Handle datetime objects
                if isinstance(value, datetime):
                    serialized[key] = value.isoformat()
                # Handle other basic types
                elif isinstance(value, str | int | float | bool | list | dict):
                    serialized[key] = value
                else:
                    # Convert other types to string
                    serialized[key] = str(value)
            except Exception:
                # If serialization fails, convert to string
                serialized[key] = str(value)

        return serialized

    def to_json(self, error: LLMCliRunnerError) -> str:
        """Serialize exception to JSON string.

        Args:
            error: The exception to serialize

        Returns:
            JSON string in MCP protocol format

        """
        return json.dumps(self.serialize(error), indent=2)

    def from_json(self, json_str: str) -> LLMCliRunnerError:
        """Deserialize exception from JSON string.

        Args:
            json_str: JSON string in MCP protocol format

        Returns:
            Exception object reconstructed from JSON

        """
        mcp_error = json.loads(json_str)
        return self.deserialize(mcp_error)
