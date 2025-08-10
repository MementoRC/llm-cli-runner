"""Custom exceptions for MCP Server Cheap LLM.

This module defines custom exception classes following atomic design principles.
Provides clear error hierarchies and actionable error messages.

Key exceptions:
    CheapLLMError: Base exception class
    ConfigurationError: Configuration-related errors
    ProviderError: Provider-specific errors
    ValidationError: Input validation errors

Example:
    >>> raise ProviderError("Gemini API key invalid", provider="gemini")
    >>> raise ConfigurationError("Missing required field: model_name")
"""

from typing import Any


class CheapLLMError(Exception):
    """Base exception class for MCP Server Cheap LLM.

    All custom exceptions inherit from this base class to provide
    consistent error handling and debugging information.

    Attributes:
        message: Human-readable error message
        error_code: Optional error code for programmatic handling
        context: Additional context information

    Example:
        >>> try:
        ...     raise CheapLLMError("Something went wrong", error_code="E001")
        ... except CheapLLMError as e:
        ...     print(f"Error {e.error_code}: {e.message}")
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize the exception.

        Args:
            message: Human-readable error description
            error_code: Optional error code for categorization
            context: Additional context for debugging
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging.

        Returns:
            Dictionary representation of the exception
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
        }


class ConfigurationError(CheapLLMError):
    """Raised when configuration is invalid or missing.

    This exception indicates problems with configuration files,
    environment variables, or provider settings.

    Example:
        >>> raise ConfigurationError(
        ...     "Invalid provider configuration",
        ...     error_code="CFG001",
        ...     context={"provider": "gemini", "field": "api_key"}
        ... )
    """


class ProviderError(CheapLLMError):
    """Raised when provider operations fail.

    This exception covers API errors, authentication failures,
    rate limiting, and other provider-specific issues.

    Attributes:
        provider: Name of the provider that failed

    Example:
        >>> raise ProviderError(
        ...     "API rate limit exceeded",
        ...     error_code="PRV001",
        ...     context={"provider": "gemini", "retry_after": 60}
        ... )
    """

    def __init__(
        self,
        message: str,
        provider: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize provider error.

        Args:
            message: Error description
            provider: Name of the failing provider
            error_code: Optional error code
            context: Additional context information
        """
        context = context or {}
        context["provider"] = provider
        super().__init__(message, error_code, context)
        self.provider = provider


class ValidationError(CheapLLMError):
    """Raised when input validation fails.

    This exception indicates problems with user input,
    request parameters, or data format validation.

    Example:
        >>> raise ValidationError(
        ...     "Prompt too long",
        ...     error_code="VAL001",
        ...     context={"max_length": 10000, "actual_length": 15000}
        ... )
    """


class RateLimitError(ProviderError):
    """Raised when provider rate limits are exceeded.

    This specialized provider error includes rate limit
    specific information for retry logic.

    Attributes:
        retry_after: Seconds to wait before retrying

    Example:
        >>> raise RateLimitError(
        ...     "Rate limit exceeded",
        ...     provider="gemini",
        ...     retry_after=60
        ... )
    """

    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: int,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize rate limit error.

        Args:
            message: Error description
            provider: Name of the provider
            retry_after: Seconds to wait before retrying
            error_code: Optional error code
            context: Additional context information
        """
        context = context or {}
        context["retry_after"] = retry_after
        super().__init__(message, provider, error_code, context)
        self.retry_after = retry_after


class SecurityError(CheapLLMError):
    """Raised when security violations are detected.

    This exception indicates potential security issues
    such as command injection attempts or unsafe operations.

    Example:
        >>> raise SecurityError(
        ...     "Unsafe command detected",
        ...     error_code="SEC001",
        ...     context={"command": "rm -rf /", "source": "user_input"}
        ... )
    """
