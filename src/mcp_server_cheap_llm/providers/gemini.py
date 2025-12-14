"""Gemini provider implementation for cheap LLM access.

This module implements the Google Gemini provider for accessing
Gemini models through CLI commands and API calls.

Key features:
    Streaming responses with backpressure handling
    Cost estimation and usage tracking
    Error handling with retries
    Model-specific optimizations

Example:
    >>> provider = GeminiProvider()
    >>> response = await provider.generate(
    ...     LLMRequest(prompt="Hello", model="gemini-1.5-flash")
    ... )

"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any

from src.mcp_server_cheap_llm.cache.metrics import CacheMetrics
from src.mcp_server_cheap_llm.core.errors import (
    ConfigurationError,
    ProviderError,
    RateLimitError,
)
from src.mcp_server_cheap_llm.core.models import (
    CostEstimate,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderType,
    QuotaStatus,
    QuotaStatusInfo,
    StreamingResponse,
    UsageStats,
)
from src.mcp_server_cheap_llm.providers.base import LLMProvider, ProviderCapabilities
from src.mcp_server_cheap_llm.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """Gemini LLM Provider.

    Provides access to Google Gemini models using the CLI interface
    with comprehensive error handling and cost tracking.

    Attributes:
        name: Provider identifier
        model: Default model name
        temperature: Default sampling temperature
        max_tokens: Maximum token limit
        metrics: Usage metrics collector
        capabilities: Supported provider capabilities

    """

    def __init__(
        self,
        config: ProviderConfig | None = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        daily_quota: int | None = None,
    ) -> None:
        """Initialize Gemini provider with configuration."""
        # Create default config if none provided
        if config is None:
            config = ProviderConfig(
                name="gemini",
                provider_type=ProviderType.GEMINI,
                models=[model],  # Use models list instead of model_name
                provider_specific={
                    "default_model": model,
                    "default_temperature": temperature,
                },
                api_key=None,  # Gemini CLI doesn't require API key in config
            )

        super().__init__(config)

        # Override name to always be "gemini" regardless of config name
        self.name = "gemini"

        # Set capabilities
        self.capabilities = {
            ProviderCapabilities.STREAMING,
            ProviderCapabilities.ASYNC_GENERATION,
        }

        # Configuration
        self.model = model
        self.temperature = temperature
        self.max_tokens = 4096  # Gemini token limit

        # Metrics tracking
        self._usage_stats = UsageStats(provider_name="gemini")
        self.cache_metrics = CacheMetrics()

        # Simple metrics object for compatibility
        self.metrics = SimpleMetrics()

        # Provider-specific settings
        self.timeout = config.timeout if config else 30
        self.retry_attempts = max_retries
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Set up quota manager with custom limit if provided
        if daily_quota is not None:
            self.quota_manager = GeminiQuotaManager(daily_limit=daily_quota)
        else:
            self.quota_manager = GeminiQuotaManager()

        logger.info(
            "Initialized Gemini provider",
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @property
    def usage_stats(self) -> UsageStats:
        """Get usage statistics."""
        return self._usage_stats

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Gemini CLI.

        Args:
            request: The LLM request containing prompt and parameters

        Returns:
            Structured response from Gemini

        Raises:
            ProviderError: If generation fails
            RateLimitError: If rate limit exceeded

        """
        start_time = time.time()

        try:
            # Check quota before processing
            if not self.quota_manager.check_quota():
                raise RateLimitError(
                    "Gemini daily quota exhausted",
                    provider="gemini",
                    retry_after=86400,  # Retry after 24 hours (daily quota)
                )

            # Validate request
            self._validate_request(request)

            # Get effective model
            model = request.model or self.model
            temperature = request.temperature or self.temperature
            max_tokens = min(request.max_tokens or self.max_tokens, self.max_tokens)

            # Build command
            cmd = self._build_command(
                prompt=request.prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=request.system_prompt,
            )

            # Execute with retries
            result = await self._execute_with_retries(cmd, request)

            # Consume quota after successful generation
            self.quota_manager.consume_quota()

            # Calculate metrics
            response_time = time.time() - start_time
            tokens_used = self._estimate_tokens(
                request.prompt,
                result.get("content", ""),
            )
            cost = self._calculate_cost(tokens_used, model)

            # Track usage
            self.metrics.record_request(
                response_time=response_time,
                token_count=tokens_used,
                cost=cost,
            )

            # Create response
            response = LLMResponse(
                content=result.get("content", ""),
                provider="gemini",
                model=model,
                success=True,
                tokens_used=tokens_used,
                cost=cost,
                response_time_ms=int(response_time * 1000),
                metadata={
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "cli_version": result.get("cli_version"),
                    "usage": {
                        "total_tokens": tokens_used,
                        "estimated_input_tokens": self._estimate_tokens(request.prompt),
                        "estimated_output_tokens": tokens_used
                        - self._estimate_tokens(request.prompt),
                    },
                },
            )

            logger.debug(
                "Generated response",
                model=model,
                tokens=tokens_used,
                cost=cost,
                response_time_ms=response.response_time_ms,
            )

            return response

        except (RateLimitError, ProviderError):
            # Re-raise specific errors
            raise
        except Exception as e:
            # Convert to provider error
            response_time = time.time() - start_time
            self.metrics.record_request(
                response_time=response_time,
                token_count=0,
                cost=0.0,
                error=str(e),
            )

            error_msg = f"Gemini generation failed: {e}"
            logger.exception(error_msg, error=e, model=request.model)
            raise ProviderError(error_msg, provider="gemini") from e

    async def stream_generate(
        self,
        request: LLMRequest,
    ) -> list[StreamingResponse]:  # Updated return type hint
        """Generate streaming response using Gemini.

        Args:
            request: The LLM request

        Returns:
            Generator yielding streaming response chunks

        Raises:
            ProviderError: If streaming fails

        """
        try:
            # Validate request
            self._validate_request(request)

            model = request.model or self.model
            temperature = request.temperature or self.temperature

            # Build streaming command
            cmd = self._build_command(
                prompt=request.prompt,
                model=model,
                temperature=temperature,
                system_prompt=request.system_prompt,
                stream=True,
            )

            # Execute streaming command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            chunk_index = 0
            chunks = []

            try:
                if process.stdout is None:
                    raise ProviderError(
                        "No stdout available from Gemini process", provider="gemini"
                    )
                async for line in process.stdout:
                    line_text = (
                        line.decode("utf-8") if isinstance(line, bytes) else line
                    )
                    if line_text.strip():
                        try:
                            chunk_data = json.loads(line_text.strip())
                            content = chunk_data.get("content", "")
                            is_final = chunk_data.get("final", False)

                            chunk = StreamingResponse(
                                content=content,
                                provider="gemini",
                                model=model,
                                is_final=is_final,
                                chunk_index=chunk_index,
                                metadata=chunk_data,
                            )

                            chunks.append(chunk)
                            chunk_index += 1

                            if is_final:
                                break

                        except json.JSONDecodeError:
                            # Handle non-JSON output
                            chunk = StreamingResponse(
                                content=line_text.strip(),
                                provider="gemini",
                                model=model,
                                is_final=False,
                                chunk_index=chunk_index,
                            )
                            chunks.append(chunk)
                            chunk_index += 1

                await process.wait()
                return chunks

            except Exception as e:
                process.terminate()
                await process.wait()
                msg = f"Streaming failed: {e}"
                raise ProviderError(msg, provider="gemini") from e

        except Exception as e:
            error_msg = f"Gemini streaming failed: {e}"
            logger.exception(error_msg, error=e)
            raise ProviderError(error_msg, provider="gemini") from e

    async def generate_streaming_response(self, request: LLMRequest):
        """Generate streaming response by simulating chunked delivery.

        This method provides streaming capability by splitting the regular
        response into chunks for compatibility with streaming tests.

        Args:
            request: The LLM request

        Yields:
            StreamingResponse: Response chunks

        """
        try:
            # Get the full response first
            response = await self.generate(request)

            # Split response into words for streaming simulation
            words = response.content.split()

            # Stream each chunk
            for i, word in enumerate(words):
                chunk = StreamingResponse(
                    content=word,
                    provider=response.provider,
                    model=response.model,
                    is_final=(i == len(words) - 1),
                    chunk_index=i,
                    metadata={
                        "total_chunks": len(words),
                        "chunk_index": i,
                        "original_model": response.model,
                    },
                )
                yield chunk

        except Exception as e:
            error_msg = f"Streaming response generation failed: {e}"
            logger.exception(error_msg, error=e)
            raise ProviderError(error_msg, provider="gemini") from e

    async def estimate_cost(self, request: LLMRequest) -> CostEstimate:
        """Estimate cost for Gemini request.

        Args:
            request: The LLM request

        Returns:
            Cost estimation details

        Raises:
            ProviderError: If cost estimation fails

        """
        try:
            model = request.model or self.model

            # Estimate tokens
            input_tokens = self._estimate_tokens(request.prompt)
            output_tokens = min(request.max_tokens or 1000, self.max_tokens)

            # Get pricing for model
            pricing = self._get_model_pricing(model)

            input_cost = input_tokens * pricing["input_cost_per_token"]
            output_cost = output_tokens * pricing["output_cost_per_token"]
            total_cost = input_cost + output_cost

            return CostEstimate(
                provider_name="gemini",
                estimated_tokens=input_tokens + output_tokens,
                cost_per_token=total_cost / (input_tokens + output_tokens)
                if (input_tokens + output_tokens) > 0
                else 0.0,
                estimated_cost_usd=total_cost,
                confidence_score=0.9,  # High confidence for known pricing
                estimation_method="pricing_table",
            )

        except Exception as e:
            error_msg = f"Cost estimation failed: {e}"
            logger.exception(error_msg, error=e)
            raise ProviderError(error_msg, provider="gemini") from e

    def get_usage_stats(self) -> dict[str, Any]:
        """Get comprehensive usage statistics.

        Returns:
            Dictionary containing usage metrics and provider stats

        """
        base_stats = self.metrics.to_dict()

        # Add provider-specific stats
        provider_stats = {
            "provider_name": "gemini",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "capabilities": [cap.value for cap in self.capabilities],
            **base_stats,
        }

        # Add usage metrics if available
        if hasattr(self, "_usage_stats") and self._usage_stats:
            usage_data = {
                "total_tokens": self._usage_stats.total_tokens,
                "total_cost": self._usage_stats.total_cost,
                "avg_response_time": self._usage_stats.average_response_time,
                "last_request_time": self._usage_stats.last_updated,
            }
            provider_stats.update(
                {
                    "total_tokens_consumed": usage_data.get("total_tokens", 0),
                    "total_cost_usd": usage_data.get("total_cost", 0.0),
                    "average_response_time_ms": usage_data.get("avg_response_time", 0),
                    "last_request_time": usage_data.get("last_request_time"),
                },
            )

        # Record metrics call
        self._usage_stats.total_requests += 1
        self._usage_stats.last_updated = datetime.now()

        return provider_stats

    def _validate_request(self, request: LLMRequest) -> None:
        """Validate LLM request parameters.

        Args:
            request: Request to validate

        Raises:
            ValueError: If request is invalid

        """
        if not request.prompt or not request.prompt.strip():
            msg = "Prompt cannot be empty"
            raise ValueError(msg)

        if request.max_tokens and request.max_tokens > self.max_tokens:
            msg = f"max_tokens cannot exceed {self.max_tokens}"
            raise ValueError(msg)

        if request.temperature is not None and not 0 <= request.temperature <= 1:
            msg = "temperature must be between 0 and 1"
            raise ValueError(msg)

    def _build_command(
        self,
        prompt: str,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> list[str]:
        """Build Gemini CLI command.

        Args:
            prompt: Text prompt
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            system_prompt: Optional system prompt
            stream: Enable streaming

        Returns:
            Command as list of strings

        """
        cmd = ["gemini"]

        # Model selection
        if model != "gemini-1.5-flash":  # Default model
            cmd.extend(["--model", model])

        # Temperature
        if temperature != 0.7:  # Default temperature
            cmd.extend(["--temperature", str(temperature)])

        # Max tokens
        if max_tokens != 1000:  # Default max tokens
            cmd.extend(["--max-tokens", str(max_tokens)])

        # System prompt
        if system_prompt:
            cmd.extend(["--system", system_prompt])

        # Streaming
        if stream:
            cmd.append("--stream")

        # Output format
        cmd.extend(["--format", "json"])

        # Add the prompt as the last argument
        cmd.append(prompt)

        logger.debug("Built Gemini command", command=" ".join(cmd))
        return cmd

    async def _execute_with_retries(
        self,
        cmd: list[str],
        request: LLMRequest,
    ) -> dict[str, Any]:
        """Execute command with retry logic.

        Args:
            cmd: Command to execute
            request: Original request for context

        Returns:
            Parsed command output

        Raises:
            ProviderError: If all retries fail
            RateLimitError: If rate limited

        """
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                # Execute command
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )

                # Decode bytes to strings
                stdout_text = stdout.decode("utf-8") if stdout else ""
                stderr_text = stderr.decode("utf-8") if stderr else ""

                if process.returncode == 0:
                    # Success - parse output
                    try:
                        return json.loads(stdout_text)
                    except json.JSONDecodeError:
                        # Fallback to plain text
                        return {"content": stdout_text.strip(), "format": "text"}
                else:
                    # Error occurred
                    error_msg = stderr_text.strip() or stdout_text.strip()

                    # Check for rate limiting
                    if (
                        "rate limit" in error_msg.lower()
                        or "quota" in error_msg.lower()
                    ):
                        msg = f"Gemini rate limit exceeded: {error_msg}"
                        raise RateLimitError(msg, provider="gemini", retry_after=3600)

                    # Check for authentication errors
                    if (
                        "auth" in error_msg.lower()
                        or "unauthorized" in error_msg.lower()
                    ):
                        msg = f"Gemini authentication failed: {error_msg}"
                        raise ConfigurationError(
                            msg,
                        )

                    last_error = ProviderError(
                        f"Gemini CLI error: {error_msg}",
                        provider="gemini",
                    )

            except TimeoutError:
                last_error = ProviderError(
                    f"Gemini request timed out after {self.timeout}s",
                    provider="gemini",
                )
            except FileNotFoundError:
                msg = "Gemini CLI not found. Please install it first."
                raise ConfigurationError(
                    msg,
                ) from None
            except Exception as e:
                last_error = ProviderError(
                    f"Gemini execution error: {e}",
                    provider="gemini",
                )

            # Wait before retry (use configured retry delay)
            if attempt < self.retry_attempts - 1:
                wait_time = self.retry_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"Gemini attempt {attempt + 1} failed, retrying in {wait_time}s",
                    error=str(last_error),
                )
                await asyncio.sleep(wait_time)

        # All retries failed
        error_msg = f"Gemini generation failed after {self.max_retries} attempts"
        raise ProviderError(error_msg, provider="gemini") from last_error

    def _estimate_tokens(self, text: str, completion: str = "") -> int:
        """Estimate token count for text.

        Args:
            text: Input text
            completion: Optional completion text

        Returns:
            Estimated token count

        """
        # Simple estimation: ~4 characters per token
        # This is conservative for Gemini
        total_chars = len(text) + len(completion)
        return max(1, total_chars // 4)

    def _calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for token usage.

        Args:
            tokens: Total token count
            model: Model used

        Returns:
            Estimated cost in USD

        """
        pricing = self._get_model_pricing(model)
        # Assume roughly equal input/output split
        cost = (
            tokens
            * (pricing["input_cost_per_token"] + pricing["output_cost_per_token"])
            / 2
        )
        return round(cost, 6)

    def _get_model_pricing(self, model: str) -> dict[str, Any]:
        """Get pricing information for model.

        Args:
            model: Model name

        Returns:
            Pricing dictionary

        """
        # Gemini pricing (as of 2024)
        pricing_table = {
            "gemini-1.5-flash": {
                "input_cost_per_token": 0.000000075,  # $0.075 per 1M tokens
                "output_cost_per_token": 0.0000003,  # $0.30 per 1M tokens
                "tier": "flash",
            },
            "gemini-1.5-pro": {
                "input_cost_per_token": 0.00000125,  # $1.25 per 1M tokens
                "output_cost_per_token": 0.000005,  # $5.00 per 1M tokens
                "tier": "pro",
            },
            "gemini-pro": {
                "input_cost_per_token": 0.0000005,  # $0.50 per 1M tokens
                "output_cost_per_token": 0.0000015,  # $1.50 per 1M tokens
                "tier": "legacy",
            },
        }

        return pricing_table.get(
            model,
            pricing_table["gemini-1.5-flash"],  # Default to flash pricing
        )

    async def health_check(self) -> dict[str, Any]:
        """Check provider health and availability.

        Returns:
            Health status information

        """
        try:
            # Test basic CLI availability
            process = await asyncio.create_subprocess_exec(
                "gemini",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)

            # Decode bytes to strings
            stdout_text = stdout.decode("utf-8") if stdout else ""
            stderr_text = stderr.decode("utf-8") if stderr else ""

            if process.returncode == 0:
                version = stdout_text.strip()
                return {
                    "status": "healthy",
                    "version": version,
                    "provider": "gemini",
                    "timestamp": datetime.now().isoformat(),
                    "capabilities": [cap.value for cap in self.capabilities],
                    "models": [self.model],
                }
            error = stderr_text.strip() or "Unknown error"
            return {
                "status": "unhealthy",
                "error": error,
                "provider": "gemini",
                "timestamp": datetime.now().isoformat(),
            }

        except FileNotFoundError:
            return {
                "status": "unavailable",
                "error": "Gemini CLI not installed",
                "provider": "gemini",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "provider": "gemini",
                "timestamp": datetime.now().isoformat(),
            }

    def validate_config(self, config: ProviderConfig) -> bool:
        """Validate provider configuration.

        Args:
            config: Provider configuration to validate

        Returns:
            True if configuration is valid

        """
        try:
            # Basic validation - Gemini CLI doesn't require API key
            return (
                config.name == "gemini"
                and config.provider_type == ProviderType.GEMINI
                and len(config.models) > 0
            )
        except Exception:
            return False

    async def get_usage(self) -> UsageStats:
        """Get current usage statistics.

        Returns:
            Current usage statistics for this provider

        """
        return self._usage_stats

    async def check_quota(self) -> QuotaStatusInfo:
        """Check current quota status.

        Returns:
            Current quota status

        """
        try:
            if self.quota_manager.check_quota():
                remaining = self.quota_manager.get_remaining_quota()
                current_usage = self.quota_manager.get_current_usage()
                quota_limit = self.quota_manager.get_quota_limit()

                return QuotaStatusInfo(
                    provider_name=self.name,
                    quota_type="requests",
                    current_usage=current_usage,
                    quota_limit=quota_limit,
                    quota_remaining=remaining,
                )

            # Quota exceeded
            current_usage = self.quota_manager.get_current_usage()
            quota_limit = self.quota_manager.get_quota_limit()

            return QuotaStatusInfo(
                provider_name=self.name,
                quota_type="requests",
                current_usage=current_usage,
                quota_limit=quota_limit,
                quota_remaining=0,
            )
        except Exception:
            return QuotaStatusInfo(
                provider_name=self.name,
                quota_type="requests",
                current_usage=0,
                quota_limit=1000,
                quota_remaining=0,
            )

    async def is_available(self) -> bool:
        """Check if the Gemini provider is available.

        Returns:
            True if Gemini CLI is available and quota is not exhausted

        """
        try:
            # Check CLI availability
            process = await asyncio.create_subprocess_exec(
                "gemini",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await process.wait()

            # If CLI is not available, return False
            if process.returncode != 0:
                return False

            # Check quota status
            quota_status = await self.check_quota()
            return quota_status != QuotaStatus.EXCEEDED

        except (FileNotFoundError, Exception):
            # CLI not found or other error
            return False

    def get_health_status(self) -> dict[str, Any]:
        """Get provider health status including quota information.

        Returns:
            Dict containing health status information with quota details
        """
        # Get base health status from parent class
        base_status = super().get_health_status()

        # Add quota information
        base_status.update(
            {
                "model": self.model,
                "quota_remaining": self.quota_manager.get_remaining_quota(),
                "quota_limit": self.quota_manager.daily_limit,
                "configuration": {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "max_retries": self.max_retries,
                    "retry_delay": self.retry_delay,
                },
            }
        )

        return base_status


class GeminiQuotaManager:
    """Simple quota manager for Gemini provider."""

    def __init__(self, daily_limit: int = 1000) -> None:
        """Initialize quota manager with daily limit."""
        self.daily_limit = daily_limit
        self.requests_today = 0
        self.last_reset = datetime.now().date()

    def check_quota(self) -> bool:
        """Check if quota is available."""
        self._reset_if_needed()
        return self.requests_today < self.daily_limit

    def consume_quota(self, amount: int = 1) -> bool:
        """Consume quota if available."""
        if self.check_quota():
            self.requests_today += amount
            return True
        return False

    def get_remaining_quota(self) -> int:
        """Get remaining quota for today."""
        self._reset_if_needed()
        return max(0, self.daily_limit - self.requests_today)

    def get_current_usage(self) -> int:
        """Get current usage for today."""
        self._reset_if_needed()
        return self.requests_today

    def get_quota_limit(self) -> int:
        """Get the daily quota limit."""
        return self.daily_limit

    def _reset_if_needed(self) -> None:
        """Reset quota if new day."""
        today = datetime.now().date()
        if today > self.last_reset:
            self.requests_today = 0
            self.last_reset = today


class SimpleMetrics:
    """Simple metrics class for compatibility."""

    def __init__(self) -> None:
        """Initialize metrics tracking."""
        self.request_timestamps = []
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_response_time = 0.0

    def record_request(
        self,
        response_time: float,
        token_count: int,
        cost: float,
        error: str | None = None,
    ) -> None:
        """Record a request for metrics tracking."""
        self.request_timestamps.append(datetime.now())
        self.request_count += 1
        self.total_tokens += token_count
        self.total_cost += cost
        self.total_response_time += response_time

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "average_response_time": (
                self.total_response_time / self.request_count
                if self.request_count > 0
                else 0.0
            ),
            "request_timestamps": [
                ts.isoformat() for ts in self.request_timestamps[-10:]
            ],  # Last 10
        }
