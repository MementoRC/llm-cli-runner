"""Abstract base class and interfaces for LLM providers.

This module defines the abstract LLMProvider interface that all provider
implementations must follow, along with supporting data classes for
capabilities, metadata, and request/response handling.
"""

import abc
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from mcp_server_cheap_llm.core.errors import ValidationError
from mcp_server_cheap_llm.core.models import LLMRequest, LLMResponse, ProviderStatus


@dataclass
class ProviderCapabilities:
    """Defines capabilities supported by a provider.

    Attributes:
        streaming: Whether provider supports streaming responses
        batch: Whether provider supports batch processing
        embeddings: Whether provider supports text embeddings
        function_calling: Whether provider supports function/tool calling
        vision: Whether provider supports image inputs
        audio: Whether provider supports audio inputs
    """

    streaming: bool = False
    batch: bool = False
    embeddings: bool = False
    function_calling: bool = False
    vision: bool = False
    audio: bool = False


@dataclass
class ProviderMetadata:
    """Metadata and configuration for a provider.

    Attributes:
        name: Provider name (e.g., "gemini", "openai")
        cost_per_token: Cost per token in USD
        rate_limits: Rate limits as requests per minute/hour
        model_variants: Available model variants
        max_tokens: Maximum tokens per request
        supported_languages: Supported language codes
    """

    name: str
    cost_per_token: Decimal
    rate_limits: dict[
        str, int
    ]  # {"requests_per_minute": 60, "tokens_per_hour": 100000}
    model_variants: list[str]
    max_tokens: int
    supported_languages: set[str] | None = None

    def __post_init__(self):
        """Initialize default supported languages if not provided."""
        if self.supported_languages is None:
            self.supported_languages = {"en"}


@dataclass
class UsageStats:
    """Usage statistics for a provider.

    Attributes:
        requests_made: Total requests made
        tokens_used: Total tokens consumed
        cost_incurred: Total cost in USD
        error_rate: Error rate as percentage
        avg_response_time: Average response time in seconds
    """

    requests_made: int = 0
    tokens_used: int = 0
    cost_incurred: Decimal = Decimal("0.00")
    error_rate: float = 0.0
    avg_response_time: float = 0.0


@dataclass
class QuotaStatus:
    """Current quota status for a provider.

    Attributes:
        remaining_requests: Requests remaining in current period
        remaining_tokens: Tokens remaining in current period
        reset_time: When quota resets (Unix timestamp)
        is_exhausted: Whether quota is currently exhausted
    """

    remaining_requests: int
    remaining_tokens: int
    reset_time: int
    is_exhausted: bool = False


@dataclass
class CostEstimate:
    """Cost estimate for a request.

    Attributes:
        estimated_tokens: Estimated tokens for the request
        estimated_cost: Estimated cost in USD
        breakdown: Cost breakdown by component
    """

    estimated_tokens: int
    estimated_cost: Decimal
    breakdown: dict[str, Decimal]  # {"input": 0.001, "output": 0.002}


class LLMProvider(abc.ABC):
    """Abstract base class for all LLM providers.

    This class defines the standard interface that all provider implementations
    must follow. It includes methods for generation, configuration validation,
    usage tracking, quota management, and cost estimation.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize provider with configuration.

        Args:
            config: Provider-specific configuration dictionary

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self._validate_init_config()

    @property
    @abc.abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities.

        Returns:
            ProviderCapabilities: Supported features
        """
        pass

    @property
    @abc.abstractmethod
    def metadata(self) -> ProviderMetadata:
        """Get provider metadata.

        Returns:
            ProviderMetadata: Provider information and limits
        """
        pass

    @abc.abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response for the given request.

        Args:
            request: The LLM request to process

        Returns:
            LLMResponse: Generated response

        Raises:
            ProviderError: If generation fails
            ValidationError: If request is invalid
        """
        pass

    @abc.abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration.

        Returns:
            bool: True if configuration is valid

        Raises:
            ValidationError: If configuration is invalid
        """
        pass

    @abc.abstractmethod
    def get_usage(self) -> UsageStats:
        """Get current usage statistics.

        Returns:
            UsageStats: Current usage information
        """
        pass

    @abc.abstractmethod
    def check_quota(self) -> QuotaStatus:
        """Check current quota status.

        Returns:
            QuotaStatus: Current quota information

        Raises:
            ProviderError: If quota check fails
        """
        pass

    @abc.abstractmethod
    def estimate_cost(self, request: LLMRequest) -> CostEstimate:
        """Estimate cost for a request.

        Args:
            request: The request to estimate cost for

        Returns:
            CostEstimate: Estimated cost information
        """
        pass

    @abc.abstractmethod
    def get_status(self) -> ProviderStatus:
        """Get current provider status.

        Returns:
            ProviderStatus: Current operational status
        """
        pass

    def _validate_init_config(self) -> None:
        """Validate configuration during initialization.

        Raises:
            ValidationError: If required config is missing
        """
        if not isinstance(self.config, dict):
            raise ValidationError("Provider config must be a dictionary")

        required_keys = self._get_required_config_keys()
        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            raise ValidationError(
                f"Missing required configuration keys: {missing_keys}"
            )

    @abc.abstractmethod
    def _get_required_config_keys(self) -> list[str]:
        """Get list of required configuration keys.

        Returns:
            List[str]: Required configuration keys
        """
        pass

    def __str__(self) -> str:
        """String representation of provider."""
        return f"{self.__class__.__name__}(name={self.metadata.name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.metadata.name}, "
            f"status={self.get_status()}, "
            f"capabilities={self.capabilities})"
        )
