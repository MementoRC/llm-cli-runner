"""OpenAI provider implementation for the MCP server."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from mcp_server_cheap_llm.core.models import (
    CostEstimate,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderType,
    QuotaStatusInfo,
    UsageStats,
)
from mcp_server_cheap_llm.utils.logging import get_logger

from .base import LLMProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize the OpenAI provider."""
        if config is None:
            # Create default config without api_key for testing
            config = ProviderConfig(
                name="openai",
                provider_type=ProviderType.OPENAI,
                enabled=True,
                models=[
                    "gpt-3.5-turbo",
                    "gpt-4",
                    "gpt-4-turbo",
                    "gpt-4o",
                    "gpt-4o-mini",
                ],
            )

        # Store config before calling super to handle validation properly
        self._temp_config = config

        # Use the config name (allows for different OpenAI instances)
        self.name = config.name
        self.provider_type = ProviderType.OPENAI
        self.config = config

        # Initialize circuit breaker and logger manually to avoid base class validation
        from mcp_server_cheap_llm.providers.circuit_breaker import (
            ProviderCircuitBreaker,
        )
        from mcp_server_cheap_llm.utils.logging import StructuredLogger

        self.circuit_breaker = ProviderCircuitBreaker(provider_name=config.name)
        self.logger = StructuredLogger(__name__)
        self.capabilities: set = set()
        self.metadata: dict = {}

        self.client: AsyncOpenAI | None = None
        self._initialized = False
        self._usage_stats = UsageStats(provider_name=self.name)

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self._initialized:
            return

        try:
            # Initialize OpenAI client
            api_key = self.config.api_key or self._get_env_var("OPENAI_API_KEY")
            if not api_key:
                msg = "OpenAI API key is required"
                raise ValueError(msg)

            base_url = self.config.base_url or self._get_env_var("OPENAI_BASE_URL")

            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout,
            )

            self._initialized = True
            logger.info("OpenAI provider initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize OpenAI provider: {e}")
            raise

    async def generate(
        self,
        request: LLMRequest | str | None = None,
        *,
        prompt: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using OpenAI API.

        Supports both LLMRequest objects and keyword arguments for backwards compatibility.
        """
        # Handle backwards compatibility
        if isinstance(request, str):
            prompt = request
            request = None

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

        # At this point, request is always an LLMRequest
        assert isinstance(request, LLMRequest), (
            "Request should be LLMRequest at this point"
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

        start_time = datetime.now()
        self._usage_stats.total_requests += 1

        try:
            # Use provided model or default
            model = request.model or (
                self.config.models[0] if self.config.models else "gpt-3.5-turbo"
            )
            max_tokens = request.max_tokens or self.config.max_tokens
            temperature = request.temperature

            # Create messages format for chat completion
            messages: list[ChatCompletionMessageParam] = [
                {"role": "user", "content": request.prompt},
            ]

            # Add system prompt if provided
            if request.system_prompt:
                messages.insert(
                    0,
                    {"role": "system", "content": request.system_prompt},
                )

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **request.metadata,
            )

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            # Extract response content
            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0

            # Update usage stats
            self._usage_stats.successful_requests += 1
            self._usage_stats.total_tokens += tokens_used
            self._update_response_time(response_time)

            return LLMResponse(
                content=content,
                provider=self.name,
                model=model,
                success=True,
                tokens_used=tokens_used,
                response_time_ms=int(response_time),
                metadata={
                    "completion_tokens": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "prompt_tokens": response.usage.prompt_tokens
                    if response.usage
                    else 0,
                },
            )

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.exception(f"OpenAI generation failed: {e}")

            # Update failure stats
            self._usage_stats.failed_requests += 1
            self._update_response_time(response_time)

            return LLMResponse(
                content="",
                provider=self.name,
                model=request.model or "unknown",
                success=False,
                error_message=str(e),
                response_time_ms=int(response_time),
                tokens_used=0,
            )

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Generate a streaming response using OpenAI API."""
        if not self._initialized:
            await self.initialize()

        if not self.client:
            yield self._create_error_response("Provider not initialized")
            return

        try:
            # Use provided model or default
            model = model or (
                self.config.models[0] if self.config.models else "gpt-3.5-turbo"
            )
            max_tokens = max_tokens or self.config.max_tokens
            temperature = temperature or kwargs.get("temperature", 0.7)

            # Create messages format for chat completion
            messages: list[ChatCompletionMessageParam] = [
                {"role": "user", "content": prompt}
            ]

            # Add system prompt if provided
            if kwargs.get("system_prompt"):
                messages.insert(
                    0,
                    {"role": "system", "content": kwargs["system_prompt"]},
                )

            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["system_prompt", "temperature"]
                },
            )

            chunk_index = 0
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield {
                        "content": chunk.choices[0].delta.content,
                        "provider": self.name,
                        "model": model,
                        "success": True,
                        "is_final": False,
                        "chunk_index": chunk_index,
                    }
                    chunk_index += 1

            # Send final chunk
            yield {
                "content": "",
                "provider": self.name,
                "model": model,
                "success": True,
                "is_final": True,
                "chunk_index": chunk_index,
            }

        except Exception as e:
            logger.exception(f"OpenAI streaming failed: {e}")
            yield {
                "content": "",
                "provider": self.name,
                "model": model or "unknown",
                "success": False,
                "error_message": str(e),
                "is_final": True,
            }

    def _get_env_var(self, var_name: str) -> str | None:
        """Get environment variable."""
        import os

        return os.getenv(var_name)

    def _create_error_response(self, error_message: str) -> dict[str, Any]:
        """Create a standardized error response."""
        return {
            "content": "",
            "provider": self.name,
            "success": False,
            "error_message": error_message,
            "tokens_used": 0,
            "response_time_ms": 0,
        }

    def _update_response_time(self, response_time: float) -> None:
        """Update average response time."""
        total_requests = self._usage_stats.total_requests
        if total_requests == 1:
            self._usage_stats.average_response_time = response_time
        else:
            # Calculate running average
            current_avg = self._usage_stats.average_response_time
            self._usage_stats.average_response_time = (
                (current_avg * (total_requests - 1)) + response_time
            ) / total_requests

    def get_available_models(self) -> list[str]:
        """Get list of available models."""
        return self.config.models or [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
        ]

    async def get_model_info(self, model: str) -> dict[str, Any]:
        """Get information about a specific model."""
        # Check against both configured models and standard OpenAI models
        available_models = self.get_available_models()
        standard_models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
        ]

        # Model is available if it's in config OR it's a standard OpenAI model
        is_available = model in available_models or model in standard_models

        if not is_available:
            return {
                "model": model,
                "available": False,
                "error": "Model not available",
            }

        # Basic model information
        model_info = {
            "model": model,
            "available": True,
            "provider": self.name,
            "type": "chat",
        }

        # Add model-specific details
        if "gpt-4" in model:
            model_info.update(
                {
                    "context_window": 8192 if "turbo" not in model else 128000,
                    "max_tokens": 4096,
                    "description": "Advanced language model with high reasoning capabilities",
                },
            )
        elif "gpt-3.5" in model:
            model_info.update(
                {
                    "context_window": 4096,
                    "max_tokens": 4096,
                    "description": "Fast and efficient language model",
                },
            )

        return model_info

    def validate_config(self, config: ProviderConfig) -> bool:
        """Validate OpenAI provider configuration."""
        try:
            if config.provider_type != ProviderType.OPENAI:
                return False
            if not config.api_key:
                return False
            # Check if models list has at least one model
            return not (not config.models or len(config.models) == 0)
        except Exception:
            return False

    async def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            if not self._initialized:
                await self.initialize()

            if not self.client:
                return False

            # Simple test request
            await self.client.models.list()
            return True

        except Exception as e:
            logger.warning(f"OpenAI availability check failed: {e}")
            return False

    async def get_usage(self) -> UsageStats:
        """Get current usage statistics."""
        self._usage_stats.last_updated = datetime.now()
        return self._usage_stats

    async def check_quota(self) -> QuotaStatusInfo:
        """Check quota status."""
        try:
            # Basic quota check - could be enhanced with actual API quota checks
            current_usage = self._usage_stats.total_requests
            quota_limit = 1000  # Default limit, should be configured

            return QuotaStatusInfo(
                provider_name=self.name,
                quota_type="requests",
                current_usage=current_usage,
                quota_limit=quota_limit,
                quota_remaining=max(0, quota_limit - current_usage),
            )

        except Exception as e:
            logger.warning(f"Failed to check OpenAI quota: {e}")
            return QuotaStatusInfo(
                provider_name=self.name,
                quota_type="requests",
                current_usage=0,
                quota_limit=1000,
                quota_remaining=1000,
            )

    async def estimate_cost(
        self,
        request: LLMRequest | str | None = None,
        *,
        prompt: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CostEstimate:
        """Estimate the cost of a request.

        Supports both LLMRequest objects and keyword arguments for backwards compatibility.
        """
        # Handle backwards compatibility
        if isinstance(request, str):
            prompt = request
            request = None

        if request is None:
            # Only pass non-None values to avoid overriding Pydantic defaults
            request_kwargs: dict[str, Any] = {"prompt": prompt or ""}
            if model is not None:
                request_kwargs["model"] = model
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            request_kwargs.update(kwargs)
            request = LLMRequest(**request_kwargs)

        # At this point, request is always an LLMRequest
        assert isinstance(request, LLMRequest), (
            "Request should be LLMRequest at this point"
        )

        model = request.model or (
            self.config.models[0] if self.config.models else "gpt-3.5-turbo"
        )
        max_tokens = request.max_tokens or self.config.max_tokens

        # Rough token estimation (4 characters per token average)
        estimated_input_tokens = len(request.prompt) // 4
        estimated_output_tokens = max_tokens or 1000
        total_tokens = estimated_input_tokens + estimated_output_tokens

        # Basic pricing (as of 2024, prices may vary)
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }

        if model not in pricing:
            model = "gpt-3.5-turbo"  # fallback

        rates = pricing[model]
        estimated_cost = (estimated_input_tokens / 1000) * rates["input"] + (
            estimated_output_tokens / 1000
        ) * rates["output"]

        # Calculate average cost per token for the estimate
        cost_per_token = estimated_cost / total_tokens if total_tokens > 0 else 0

        return CostEstimate(
            provider_name=self.name,
            estimated_tokens=total_tokens,
            cost_per_token=cost_per_token,
            estimated_cost_usd=round(estimated_cost, 6),
            confidence_score=0.8,
            estimation_method="token_based",
            cost_breakdown={
                "model": model,
                "estimated_input_tokens": estimated_input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "input_cost_per_1k": rates["input"],
                "output_cost_per_1k": rates["output"],
            },
        )

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on the provider."""
        health_status = {
            "provider": self.name,
            "initialized": self._initialized,
            "available": False,
            "models_configured": len(self.config.models) if self.config.models else 0,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            health_status["available"] = await self.is_available()

            if health_status["available"]:
                # Test a simple generation
                test_request = LLMRequest(
                    prompt="Hello", max_tokens=5, system_prompt=None
                )
                test_response = await self.generate(test_request)
                health_status["test_generation"] = test_response.success
            else:
                health_status["test_generation"] = False

        except Exception as e:
            health_status["error"] = str(e)
            health_status["test_generation"] = False

        return health_status

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.client:
            await self.client.close()
            self.client = None
        self._initialized = False
        logger.info("OpenAI provider cleaned up")
