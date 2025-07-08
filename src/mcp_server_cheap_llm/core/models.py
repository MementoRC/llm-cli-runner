"""Core data models for MCP Server Cheap LLM.

This module defines Pydantic models for configuration, requests, and responses.
Follows atomic design patterns with clear data structures (200-300 lines).

Key models:
    ProviderConfig: Configuration for LLM providers
    LLMRequest: Standardized request format
    LLMResponse: Standardized response format
    ProviderStatus: Runtime status information

Example:
    >>> config = ProviderConfig(name="gemini", enabled=True)
    >>> request = LLMRequest(prompt="Hello", provider="gemini")
    >>> response = LLMResponse(content="Hi there!", provider="gemini")
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    GEMINI = "gemini"
    CODEX = "codex"
    LLAMA = "llama"


class ProviderStatus(str, Enum):
    """Provider operational status."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class LLMRequest(BaseModel):
    """Standardized request format for LLM providers.

    Attributes:
        prompt: The text prompt to send to the LLM
        provider: Target provider (optional, uses default if not specified)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)
        system_prompt: Optional system prompt for context
        metadata: Additional provider-specific parameters

    Example:
        >>> request = LLMRequest(
        ...     prompt="Explain Python decorators",
        ...     provider="gemini",
        ...     max_tokens=500,
        ...     temperature=0.7
        ... )
    """

    prompt: str = Field(..., min_length=1, max_length=10000)
    provider: ProviderType | None = None
    max_tokens: int = Field(default=1000, ge=1, le=8000)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    system_prompt: str | None = Field(None, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @validator("metadata")
    def validate_metadata(cls, v):
        """Ensure metadata doesn't exceed reasonable size."""
        if len(str(v)) > 5000:
            raise ValueError("Metadata too large (max 5000 characters)")
        return v


class LLMResponse(BaseModel):
    """Standardized response format from LLM providers.

    Attributes:
        content: The generated text response
        provider: Which provider generated the response
        success: Whether the request succeeded
        error_message: Error details if success=False
        tokens_used: Number of tokens consumed
        response_time_ms: Response time in milliseconds
        metadata: Provider-specific response data

    Example:
        >>> response = LLMResponse(
        ...     content="Decorators are a way to modify functions...",
        ...     provider="gemini",
        ...     success=True,
        ...     tokens_used=45,
        ...     response_time_ms=1250
        ... )
    """

    content: str = ""
    provider: ProviderType
    success: bool = True
    error_message: str | None = None
    tokens_used: int = Field(default=0, ge=0)
    response_time_ms: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    def to_debug_dict(self) -> dict[str, Any]:
        """Convert to dictionary for debugging purposes.

        Returns:
            Dictionary containing all response data for logging
        """
        return {
            "provider": self.provider,
            "success": self.success,
            "content_length": len(self.content),
            "tokens_used": self.tokens_used,
            "response_time_ms": self.response_time_ms,
            "has_error": self.error_message is not None,
            "created_at": self.created_at.isoformat(),
        }


class ProviderConfig(BaseModel):
    """Configuration for individual LLM providers.

    Attributes:
        name: Unique provider identifier
        provider_type: Type of provider (gemini/codex/llama)
        enabled: Whether provider is active
        api_key: Authentication key (optional)
        endpoint_url: Custom endpoint URL (optional)
        model_name: Specific model to use
        default_max_tokens: Default token limit
        default_temperature: Default sampling temperature
        rate_limit_per_minute: Requests per minute limit
        timeout_seconds: Request timeout
        provider_specific: Provider-specific configuration

    Example:
        >>> config = ProviderConfig(
        ...     name="my_gemini",
        ...     provider_type=ProviderType.GEMINI,
        ...     enabled=True,
        ...     model_name="gemini-pro"
        ... )
    """

    name: str = Field(..., min_length=1, max_length=50)
    provider_type: ProviderType
    enabled: bool = True
    api_key: str | None = Field(None, min_length=1)
    endpoint_url: str | None = None
    model_name: str = Field(..., min_length=1)
    default_max_tokens: int = Field(default=1000, ge=1, le=8000)
    default_temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    provider_specific: dict[str, Any] = Field(default_factory=dict)

    @validator("name")
    def validate_name(cls, v):
        """Ensure name contains only valid characters."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Name must contain only letters, numbers, hyphens, and underscores"
            )
        return v


class ProviderStatusInfo(BaseModel):
    """Runtime status information for providers.

    Attributes:
        name: Provider name
        status: Current operational status
        last_request_time: When last request was made
        total_requests: Total requests processed
        failed_requests: Number of failed requests
        average_response_time_ms: Average response time
        rate_limit_remaining: Remaining rate limit tokens
        error_details: Details of last error (if any)

    Example:
        >>> status = ProviderStatusInfo(
        ...     name="gemini",
        ...     status=ProviderStatus.AVAILABLE,
        ...     total_requests=145,
        ...     average_response_time_ms=1200
        ... )
    """

    name: str
    status: ProviderStatus
    last_request_time: datetime | None = None
    total_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    average_response_time_ms: float = Field(default=0.0, ge=0.0)
    rate_limit_remaining: int | None = Field(None, ge=0)
    error_details: str | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage.

        Returns:
            Success rate between 0.0 and 100.0
        """
        if self.total_requests == 0:
            return 100.0
        return (
            (self.total_requests - self.failed_requests) / self.total_requests
        ) * 100.0

    def to_debug_dict(self) -> dict[str, Any]:
        """Convert to dictionary for debugging.

        Returns:
            Dictionary with status information for logging
        """
        return {
            "name": self.name,
            "status": self.status,
            "total_requests": self.total_requests,
            "success_rate": round(self.success_rate, 2),
            "avg_response_ms": round(self.average_response_time_ms, 2),
            "has_error": self.error_details is not None,
            "last_request": (
                self.last_request_time.isoformat() if self.last_request_time else None
            ),
        }
