"""OpenAI provider implementation for the MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from mcp_server_cheap_llm.core.errors import ValidationError
from mcp_server_cheap_llm.core.models import (
    CostEstimate,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderType,
    QuotaStatusInfo,
)
from mcp_server_cheap_llm.utils.logging import get_logger

if TYPE_CHECKING:
    pass  # All needed imports are above

logger = get_logger(__name__)


class OpenAIProvider:
    """OpenAI provider for cheap LLM access."""

    def __init__(self, config: ProviderConfig):
        """Initialize the OpenAI provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self.name = config.name
        self.provider_type = ProviderType.OPENAI
        self.client: AsyncOpenAI | None = None
        self._initialized = False

        # Extract API key from config
        self.api_key = config.api_key
        if not self.api_key:
            logger.warning(f"No API key provided for OpenAI provider '{self.name}'")

        logger.info(f"Initialized OpenAI provider: {self.name}")

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self._initialized:
            return

        try:
            if not self.api_key:
                logger.error(
                    f"Cannot initialize OpenAI provider '{self.name}': No API key"
                )
                return

            self.client = AsyncOpenAI(api_key=self.api_key)
            self._initialized = True
            logger.info(f"OpenAI provider '{self.name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider '{self.name}': {e}")
            self._initialized = False

    async def close(self) -> None:
        """Close the OpenAI client."""
        if self.client:
            await self.client.close()
            self.client = None
            self._initialized = False
            logger.info(f"OpenAI provider '{self.name}' closed")

    def _convert_to_openai_messages(
        self, prompt: str, system_prompt: str | None = None
    ) -> list[ChatCompletionMessageParam]:
        """Convert prompt to OpenAI message format.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            List of OpenAI message dictionaries
        """
        messages: list[ChatCompletionMessageParam] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return messages

    async def generate(
        self,
        request: LLMRequest | None = None,
        *,
        prompt: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using OpenAI.

        Args:
            request: Optional LLMRequest object
            prompt: Text prompt (if request not provided)
            model: Model name override
            max_tokens: Maximum tokens to generate
            temperature: Temperature for randomness
            system_prompt: System prompt to use
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated text or error
        """
        if request is None:
            # Only pass non-None values to avoid overriding Pydantic defaults
            request_kwargs: dict[str, Any] = {"prompt": prompt or ""}
            if model is not None:
                request_kwargs["model"] = model
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if system_prompt is not None:
                request_kwargs["system_prompt"] = system_prompt
            request_kwargs.update(kwargs)
            request = LLMRequest(**request_kwargs)

        # Security fix: Replace assert with proper validation
        if not isinstance(request, LLMRequest):
            raise ValidationError(
                "Invalid request type: expected LLMRequest",
                details={"received_type": type(request).__name__},
            )

        if not self._initialized:
            await self.initialize()

        if not self.client:
            return LLMResponse(
                content="",
                provider=self.name,
                success=False,
                error_message="Provider not initialized",
            )

        try:
            # Use configured model or default
            model_name = request.model or (
                self.config.models[0] if self.config.models else "gpt-3.5-turbo"
            )
            max_tokens = request.max_tokens or self.config.max_tokens
            temperature = request.temperature or 0.7

            # Convert to OpenAI message format
            messages = self._convert_to_openai_messages(
                request.prompt, request.system_prompt
            )

            # Make the API call
            logger.debug(f"Making OpenAI API call with model: {model_name}")

            completion = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Extract response content
            content = ""
            if completion.choices and completion.choices[0].message.content:
                content = completion.choices[0].message.content

            # Calculate tokens used if available
            tokens_used = 0
            if completion.usage:
                tokens_used = completion.usage.total_tokens or 0

            return LLMResponse(
                content=content,
                provider=self.name,
                model=model_name,
                success=True,
                tokens_used=tokens_used,
            )

        except Exception as e:
            error_msg = f"OpenAI API error: {e}"
            logger.error(error_msg)

            return LLMResponse(
                content="",
                provider=self.name,
                success=False,
                error_message=error_msg,
            )

    async def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of available model names
        """
        if not self._initialized:
            await self.initialize()

        if not self.client:
            logger.warning(f"OpenAI provider '{self.name}' not initialized")
            return []

        try:
            models = await self.client.models.list()
            model_names = [model.id for model in models.data]
            logger.debug(f"Retrieved {len(model_names)} models from OpenAI")
            return model_names

        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}")
            return []

    async def get_cost_estimate(
        self,
        request: LLMRequest | None = None,
        *,
        prompt: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CostEstimate:
        """Get cost estimate for a request.

        Args:
            request: Optional LLMRequest object
            prompt: Text prompt (if request not provided)
            model: Model name override
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Cost estimate information
        """
        if request is None:
            # Only pass non-None values to avoid overriding Pydantic defaults
            request_kwargs: dict[str, Any] = {"prompt": prompt or ""}
            if model is not None:
                request_kwargs["model"] = model
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            request_kwargs.update(kwargs)
            request = LLMRequest(**request_kwargs)

        # Security fix: Replace assert with proper validation
        if not isinstance(request, LLMRequest):
            raise ValidationError(
                "Invalid request type: expected LLMRequest",
                details={"received_type": type(request).__name__},
            )

        model = request.model or (
            self.config.models[0] if self.config.models else "gpt-3.5-turbo"
        )
        max_tokens = request.max_tokens or self.config.max_tokens

        # Rough token estimation (4 characters per token average)
        estimated_input_tokens = len(request.prompt) // 4
        estimated_output_tokens = max_tokens or 1000

        # Rough cost estimation based on OpenAI pricing (as of 2024)
        # These are approximate rates - actual rates may vary
        cost_per_1k_input = 0.0015  # $0.0015 per 1K input tokens for GPT-3.5
        cost_per_1k_output = 0.002  # $0.002 per 1K output tokens for GPT-3.5

        if "gpt-4" in model.lower():
            cost_per_1k_input = 0.03  # $0.03 per 1K input tokens for GPT-4
            cost_per_1k_output = 0.06  # $0.06 per 1K output tokens for GPT-4

        estimated_input_cost = (estimated_input_tokens / 1000) * cost_per_1k_input
        estimated_output_cost = (estimated_output_tokens / 1000) * cost_per_1k_output
        total_cost = estimated_input_cost + estimated_output_cost

        total_tokens = estimated_input_tokens + estimated_output_tokens
        # Calculate average cost per token for the estimate
        avg_cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0.0

        return CostEstimate(
            provider_name=self.name,
            estimated_tokens=total_tokens,
            cost_per_token=avg_cost_per_token,
            estimated_cost_usd=total_cost,
            cost_breakdown={
                "model": model,
                "input_tokens": estimated_input_tokens,
                "output_tokens": estimated_output_tokens,
                "currency": "USD",
            },
        )

    async def get_quota_status(self) -> QuotaStatusInfo:
        """Get quota status information.

        Returns:
            Quota status information
        """
        # OpenAI doesn't provide direct quota API access
        # Return basic status based on initialization
        return QuotaStatusInfo(
            provider_name=self.name,
            quota_type="requests",
            current_usage=0,
            quota_limit=float("inf"),
            quota_remaining=float("inf"),
            reset_time=None,
        )

    async def health_check(self) -> bool:
        """Check provider health.

        Returns:
            True if provider is healthy, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        if not self.client:
            return False

        try:
            # Try to list models as a health check
            await self.client.models.list()
            return True

        except Exception as e:
            logger.warning(f"OpenAI provider '{self.name}' health check failed: {e}")
            return False

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider information.

        Returns:
            Dictionary containing provider information
        """
        return {
            "name": self.name,
            "type": self.provider_type.value,
            "initialized": self._initialized,
            "models": self.config.models,
            "max_tokens": self.config.max_tokens,
            "supports_streaming": False,  # OpenAI streaming not yet implemented
            "supports_system_prompt": True,
            "supports_temperature": True,
        }
