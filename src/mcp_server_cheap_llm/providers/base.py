"""Abstract base class for LLM providers.

This module defines the core interface that all LLM providers must implement.
Includes circuit breaker integration and comprehensive error handling.

Key classes:
    LLMProvider: Abstract base class with core interface
    ProviderCapabilities: Supported feature enumeration

Example:
    >>> class MyProvider(LLMProvider):
    ...     async def generate(self, request: LLMRequest) -> LLMResponse:
    ...         # Implementation here
    ...         pass

"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from mcp_server_cheap_llm.core.errors import ProviderError
from mcp_server_cheap_llm.core.models import (
    CostEstimate,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderType,
    QuotaStatusInfo,
    UsageStats,
)
from mcp_server_cheap_llm.utils.logging import StructuredLogger

from .circuit_breaker import ProviderCircuitBreaker


class ProviderCapabilities(str, Enum):
    """Enumeration of provider capabilities."""

    STREAMING = "streaming"
    BATCH_PROCESSING = "batch_processing"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"
    IMAGE_PROCESSING = "image_processing"
    ASYNC_GENERATION = "async_generation"


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.

    This class defines the interface that all concrete provider implementations
    must follow. It includes circuit breaker integration for reliability.

    Attributes:
        name: Unique provider identifier
        provider_type: Type of provider (from ProviderType enum)
        config: Provider configuration
        capabilities: Set of supported capabilities
        circuit_breaker: Circuit breaker for handling failures
        logger: Structured logger instance

    Example:
        >>> class GeminiProvider(LLMProvider):
        ...     def __init__(self, config: ProviderConfig):
        ...         super().__init__(config)
        ...         self.api_client = GeminiClient(config.api_key)
        ...
        ...     async def generate(self, request: LLMRequest) -> LLMResponse:
        ...         return await self._generate_with_circuit_breaker(request)

    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize provider with configuration.

        Args:
            config: Provider configuration object

        Raises:
            ProviderError: If configuration is invalid

        """
        if not self.validate_config(config):
            msg = f"Invalid configuration for provider {config.name}"
            raise ProviderError(
                msg,
                provider=config.name,
                error_code="INVALID_CONFIG",
                context={"provider": config.name, "config": config.model_dump()},
            )

        self.name = config.name
        self.config = config
        self.circuit_breaker = ProviderCircuitBreaker(provider_name=config.name)
        self.logger = StructuredLogger(__name__)

        # Set by subclasses
        self.provider_type: ProviderType = config.provider_type
        self.capabilities: set[ProviderCapabilities] = set()
        self.metadata: dict[str, Any] = {}

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM provider.

        This is the main interface method that all providers must implement.
        Should handle the actual API call to the LLM service.

        Args:
            request: Standardized request object

        Returns:
            LLMResponse: Standardized response object

        Raises:
            ProviderError: If generation fails

        """

    @abstractmethod
    def validate_config(self, config: ProviderConfig) -> bool:
        """Validate provider configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid

        """

    @abstractmethod
    async def get_usage(self) -> UsageStats:
        """Get current usage statistics.

        Returns:
            UsageStats: Current usage information

        """

    @abstractmethod
    async def check_quota(self) -> QuotaStatusInfo:
        """Check current quota status.

        Returns:
            QuotaStatusInfo: Current quota information

        """

    @abstractmethod
    async def estimate_cost(self, request: LLMRequest) -> CostEstimate:
        """Estimate cost for a request.

        Args:
            request: Request to estimate cost for

        Returns:
            CostEstimate: Cost estimation

        """

    async def _generate_with_circuit_breaker(self, request: LLMRequest) -> LLMResponse:
        """Generate response with circuit breaker protection.

        This method wraps the actual generate call with circuit breaker logic.
        Subclasses can use this for reliable generation.

        Args:
            request: Request to process

        Returns:
            LLMResponse: Response from provider

        Raises:
            ProviderError: If circuit breaker is open or generation fails

        """

        async def _internal_generate():
            """Internal function for circuit breaker."""
            return await self.generate(request)

        try:
            return await self.circuit_breaker.call(_internal_generate)
        except Exception as e:
            self.logger.exception(
                "Provider generation failed",
                extra={
                    "provider": self.name,
                    "error": str(e),
                    "request_id": getattr(request, "metadata", {}).get("request_id"),
                },
            )
            msg = f"Generation failed for provider {self.name}: {e!s}"
            raise ProviderError(
                msg,
                provider=self.name,
                error_code="GENERATION_FAILED",
                context={"provider": self.name, "error": str(e)},
            ) from e

    def supports_capability(self, capability: ProviderCapabilities) -> bool:
        """Check if provider supports a specific capability.

        Args:
            capability: Capability to check

        Returns:
            bool: True if capability is supported

        """
        return capability in self.capabilities

    def get_health_status(self) -> dict[str, Any]:
        """Get provider health status.

        Returns:
            Dict containing health status information

        """
        circuit_status = self.circuit_breaker.get_health_status()
        return {
            "provider": self.name,
            "provider_type": self.provider_type,
            "circuit_breaker": circuit_status,
            "capabilities": list(self.capabilities),
            "config_valid": self.validate_config(self.config),
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """String representation of provider."""
        return f"{self.__class__.__name__}(name={self.name}, type={self.provider_type})"

    def __repr__(self) -> str:
        """Detailed representation of provider."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"type={self.provider_type}, "
            f"capabilities={self.capabilities}, "
            f"circuit_state={self.circuit_breaker.get_state()}"
            f")"
        )
