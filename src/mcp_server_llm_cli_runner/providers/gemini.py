"""Gemini provider implementation for LLM CLI runner access.

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
    ...     LLMRequest(prompt="Hello", model="gemini-2.5-flash-lite")
    ... )

"""

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Any

from mcp_server_llm_cli_runner.cache.metrics import CacheMetrics
from mcp_server_llm_cli_runner.core.errors import (
    ConfigurationError,
    ProviderError,
    RateLimitError,
)
from mcp_server_llm_cli_runner.core.models import (
    CostEstimate,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderType,
    QuotaStatusInfo,
    StreamingResponse,
    UsageStats,
)
from mcp_server_llm_cli_runner.providers.base import LLMProvider, ProviderCapabilities
from mcp_server_llm_cli_runner.utils.logging import get_logger

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

    PROVIDER_TYPE = ProviderType.GEMINI

    def __init__(
        self,
        config: ProviderConfig | None = None,
        model: str = "gemini-2.5-flash-lite",
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
        # CLI-based LLMs need longer timeout (reads project context on startup)
        self.timeout = config.timeout if config else 120
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

            # Extract response content (Gemini CLI uses "response" field, not "content")
            response_content = result.get("response", "")

            # Extract actual token counts from stats if available
            # Structure: stats.models.[model_name].tokens.{prompt, candidates, total}
            stats = result.get("stats", {})
            models_stats = stats.get("models", {})

            # Find the model stats (could be under any model name)
            actual_tokens = None
            for model_stats in models_stats.values():
                tokens_info = model_stats.get("tokens", {})
                if tokens_info:
                    actual_tokens = {
                        "prompt": tokens_info.get("prompt", 0),
                        "completion": tokens_info.get("candidates", 0),
                        "total": tokens_info.get("total", 0),
                    }
                    break

            # Use actual tokens if available, otherwise estimate
            if actual_tokens:
                tokens_used = actual_tokens["total"]
            else:
                tokens_used = self._estimate_tokens(request.prompt, response_content)

            cost = self._calculate_cost(tokens_used, model)

            # Track usage
            self.metrics.record_request(
                response_time=response_time,
                token_count=tokens_used,
                cost=cost,
            )

            # Create response
            response = LLMResponse(
                content=response_content,
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
                    "session_id": result.get("session_id"),
                    "usage": actual_tokens
                    or {
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

        except RateLimitError:
            # Re-raise rate limit errors without modification
            raise
        except ProviderError as e:
            # Re-raise provider errors with retry information
            response_time = time.time() - start_time
            self.metrics.record_request(
                response_time=response_time,
                token_count=0,
                cost=0.0,
                error=str(e),
            )

            # If the error has retry attempts, log them
            if hasattr(e, "retry_attempts") and e.retry_attempts:
                logger.error(
                    f"Gemini generation failed: {e}",
                    error=e,
                    model=request.model,
                    retry_attempts=e.retry_attempts,
                )
            else:
                logger.exception(
                    f"Gemini generation failed: {e}", error=e, model=request.model
                )
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
    ) -> list[StreamingResponse]:
        """Generate streaming response using Gemini.

        Args:
            request: The LLM request

        Returns:
            List of streaming response chunks

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

            # Regex to find JSON objects in lines (handles mixed output)
            json_pattern = re.compile(r"({.*})")

            try:
                if process.stdout is None:
                    raise ProviderError(
                        "No stdout available from Gemini process", provider="gemini"
                    )
                async for line in process.stdout:
                    line_text = (
                        line.decode("utf-8") if isinstance(line, bytes) else line
                    ).strip()

                    if not line_text:
                        continue

                    # Try to find JSON in the line
                    match = json_pattern.search(line_text)
                    if match:
                        json_str = match.group(1)
                        try:
                            chunk_data = json.loads(json_str)
                            event_type = chunk_data.get("type", "")

                            content = ""
                            is_final = False

                            if event_type == "message":
                                content = chunk_data.get("content", "")
                            elif event_type == "result":
                                content = chunk_data.get("response", "")
                                is_final = True
                            elif event_type == "error":
                                error_msg = chunk_data.get("error", {})
                                content = f"Error: {error_msg}"
                                is_final = True
                            elif event_type == "init":
                                continue

                            chunk = StreamingResponse(
                                content=content,
                                provider="gemini",
                                model=model,
                                is_final=is_final,
                                chunk_index=chunk_index,
                                metadata={"event_type": event_type, **chunk_data},
                            )

                            chunks.append(chunk)
                            chunk_index += 1

                            if is_final:
                                break
                        except json.JSONDecodeError:
                            # If extraction failed, treat as plain text
                            pass
                    else:
                        # Non-JSON output - treat as raw content chunk
                        chunk = StreamingResponse(
                            content=line_text,
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
        """Estimate cost for Gemini request."""
        try:
            model = request.model or self.model
            input_tokens = self._estimate_tokens(request.prompt)
            output_tokens = min(request.max_tokens or 1000, self.max_tokens)
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
                confidence_score=0.9,
                estimation_method="pricing_table",
            )
        except Exception as e:
            error_msg = f"Cost estimation failed: {e}"
            logger.exception(error_msg, error=e)
            raise ProviderError(error_msg, provider="gemini") from e

    def get_usage_stats(self) -> dict[str, Any]:
        """Get comprehensive usage statistics."""
        base_stats = self.metrics.to_dict()
        provider_stats = {
            "provider_name": "gemini",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "capabilities": [cap.value for cap in self.capabilities],
            **base_stats,
        }

        if hasattr(self, "_usage_stats") and self._usage_stats:
            provider_stats.update(
                {
                    "total_tokens_consumed": self._usage_stats.total_tokens,
                    "total_cost_usd": self._usage_stats.total_cost,
                    "average_response_time_ms": self._usage_stats.average_response_time,
                    "last_request_time": self._usage_stats.last_updated,
                },
            )

        self._usage_stats.total_requests += 1
        self._usage_stats.last_updated = datetime.now()
        return provider_stats

    def _validate_request(self, request: LLMRequest) -> None:
        """Validate LLM request parameters."""
        if not request.prompt or not request.prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if request.max_tokens and request.max_tokens > self.max_tokens:
            raise ValueError(f"max_tokens cannot exceed {self.max_tokens}")

        if request.temperature is not None and not 0 <= request.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")

    def _build_command(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> list[str]:
        """Build Gemini CLI command."""

        # Warn about ignored parameters
        if temperature != 0.7:
            logger.warning(
                "Gemini CLI does not support 'temperature' parameter. It will be ignored."
            )
        if max_tokens != 1000 and max_tokens != 4096:
            logger.warning(
                "Gemini CLI does not support 'max_tokens' parameter. It will be ignored."
            )
        if system_prompt:
            logger.warning(
                "Gemini CLI does not support 'system_prompt'. It will be ignored."
            )

        cmd = ["gemini"]
        if model != "gemini-2.5-flash-lite":
            cmd.extend(["--model", model])

        if stream:
            cmd.extend(["--output-format", "stream-json"])
        else:
            cmd.extend(["--output-format", "json"])

        cmd.extend(["-p", prompt])
        logger.debug("Built Gemini command", command=" ".join(cmd))
        return cmd

    async def _execute_with_retries(
        self,
        cmd: list[str],
        request: LLMRequest,
    ) -> dict[str, Any]:
        """Execute command with retry logic."""
        last_error = None
        retry_attempts_log: list[dict[str, Any]] = []

        for attempt in range(self.retry_attempts):
            attempt_start_time = time.time()

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )

                stdout_text = stdout.decode("utf-8") if stdout else ""
                stderr_text = stderr.decode("utf-8") if stderr else ""
                attempt_duration_ms = int((time.time() - attempt_start_time) * 1000)

                if process.returncode == 0:
                    try:
                        # Handle potential mixed output in non-streaming as well
                        json_match = re.search(r"({.*})", stdout_text, re.DOTALL)
                        if json_match:
                            result: dict[str, Any] = json.loads(json_match.group(1))
                        else:
                            result = {"response": stdout_text.strip(), "format": "text"}
                    except json.JSONDecodeError:
                        result = {"response": stdout_text.strip(), "format": "text"}

                    if retry_attempts_log:
                        result["retry_attempts"] = retry_attempts_log
                    return result
                else:
                    error_msg = stderr_text.strip() or stdout_text.strip()
                    retry_attempts_log.append(
                        {
                            "attempt": attempt + 1,
                            "error": error_msg,
                            "duration_ms": attempt_duration_ms,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                    if (
                        "rate limit" in error_msg.lower()
                        or "quota" in error_msg.lower()
                    ):
                        raise RateLimitError(
                            f"Gemini rate limit exceeded: {error_msg}",
                            provider="gemini",
                            retry_after=3600,
                        )

                    if (
                        "auth" in error_msg.lower()
                        or "unauthorized" in error_msg.lower()
                    ):
                        raise ConfigurationError(
                            f"Gemini authentication failed: {error_msg}"
                        )

                    last_error = ProviderError(
                        f"Gemini CLI error: {error_msg}", provider="gemini"
                    )

            except TimeoutError:
                timeout_error = f"Gemini request timed out after {self.timeout}s"
                retry_attempts_log.append(
                    {
                        "attempt": attempt + 1,
                        "error": timeout_error,
                        "duration_ms": int((time.time() - attempt_start_time) * 1000),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                last_error = ProviderError(timeout_error, provider="gemini")
            except FileNotFoundError as err:
                raise ConfigurationError(
                    "Gemini CLI not found. Please install it first."
                ) from err
            except Exception as e:
                error_msg = f"Gemini execution error: {e}"
                retry_attempts_log.append(
                    {
                        "attempt": attempt + 1,
                        "error": error_msg,
                        "duration_ms": int((time.time() - attempt_start_time) * 1000),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                last_error = ProviderError(error_msg, provider="gemini")

            if attempt < self.retry_attempts - 1:
                wait_time = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Gemini attempt {attempt + 1} failed, retrying in {wait_time}s",
                    error=str(last_error),
                )
                await asyncio.sleep(wait_time)

        error = ProviderError(
            f"Gemini generation failed after {self.max_retries} attempts",
            provider="gemini",
        )
        error.retry_attempts = retry_attempts_log
        raise error from last_error

    def _estimate_tokens(self, text: str, completion: str = "") -> int:
        """Estimate token count for text."""
        total_chars = len(text) + len(completion)
        return max(1, total_chars // 4)

    def _calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for token usage."""
        pricing = self._get_model_pricing(model)
        cost = (
            tokens
            * (pricing["input_cost_per_token"] + pricing["output_cost_per_token"])
            / 2
        )
        return round(cost, 6)

    def _get_model_pricing(self, model: str) -> dict[str, Any]:
        """Get pricing information for model."""
        pricing_table = {
            "gemini-2.5-flash-lite": {
                "input_cost_per_token": 0.000000075,
                "output_cost_per_token": 0.0000003,
                "tier": "flash-lite",
            },
            "gemini-2.5-flash": {
                "input_cost_per_token": 0.000000075,
                "output_cost_per_token": 0.0000003,
                "tier": "flash",
            },
            "gemini-1.5-pro": {
                "input_cost_per_token": 0.00000125,
                "output_cost_per_token": 0.000005,
                "tier": "pro",
            },
            "gemini-pro": {
                "input_cost_per_token": 0.0000005,
                "output_cost_per_token": 0.0000015,
                "tier": "legacy",
            },
        }
        return pricing_table.get(model, pricing_table["gemini-2.5-flash-lite"])

    async def health_check(self) -> dict[str, Any]:
        """Check provider health and availability."""
        try:
            process = await asyncio.create_subprocess_exec(
                "gemini",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            if process.returncode == 0:
                return {
                    "status": "healthy",
                    "version": stdout.decode("utf-8").strip(),
                    "provider": "gemini",
                    "timestamp": datetime.now().isoformat(),
                    "capabilities": [cap.value for cap in self.capabilities],
                    "models": [self.model],
                }
            return {
                "status": "unhealthy",
                "error": stderr.decode("utf-8").strip() or "Unknown error",
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
        """Validate provider configuration."""
        try:
            return (
                config.name == "gemini"
                and config.provider_type == ProviderType.GEMINI
                and len(config.models) > 0
            )
        except Exception:
            return False

    async def get_usage(self) -> UsageStats:
        """Get current usage statistics."""
        return self._usage_stats

    async def check_quota(self) -> QuotaStatusInfo:
        """Check current quota status."""
        try:
            if self.quota_manager.check_quota():
                return QuotaStatusInfo(
                    provider_name=self.name,
                    quota_type="requests",
                    current_usage=self.quota_manager.get_current_usage(),
                    quota_limit=self.quota_manager.get_quota_limit(),
                    quota_remaining=self.quota_manager.get_remaining_quota(),
                )
            return QuotaStatusInfo(
                provider_name=self.name,
                quota_type="requests",
                current_usage=self.quota_manager.get_current_usage(),
                quota_limit=self.quota_manager.get_quota_limit(),
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
        """Check if the Gemini provider is available."""
        try:
            process = await asyncio.create_subprocess_exec(
                "gemini",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.wait()
            if process.returncode != 0:
                return False
            quota_status = await self.check_quota()
            return quota_status.quota_remaining > 0
        except (FileNotFoundError, Exception):
            return False

    def get_health_status(self) -> dict[str, Any]:
        """Get provider health status including quota information."""
        base_status = super().get_health_status()
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
        """Initialize quota manager with daily request limit."""
        self.daily_limit = daily_limit
        self.requests_today = 0
        self.last_reset = datetime.now().date()

    def check_quota(self) -> bool:
        """Check if quota is available for requests."""
        self._reset_if_needed()
        return self.requests_today < self.daily_limit

    def consume_quota(self, amount: int = 1) -> bool:
        """Consume quota and return True if successful."""
        if self.check_quota():
            self.requests_today += amount
            return True
        return False

    def get_remaining_quota(self) -> int:
        """Get remaining quota for today."""
        self._reset_if_needed()
        return max(0, self.daily_limit - self.requests_today)

    def get_current_usage(self) -> int:
        """Get current usage count for today."""
        self._reset_if_needed()
        return self.requests_today

    def get_quota_limit(self) -> int:
        """Get the daily quota limit."""
        return self.daily_limit

    def _reset_if_needed(self) -> None:
        today = datetime.now().date()
        if today > self.last_reset:
            self.requests_today = 0
            self.last_reset = today


class SimpleMetrics:
    """Simple metrics class for compatibility."""

    def __init__(self) -> None:
        """Initialize metrics with zeroed counters."""
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
        """Record metrics for a completed request."""
        self.request_timestamps.append(datetime.now())
        self.request_count += 1
        self.total_tokens += token_count
        self.total_cost += cost
        self.total_response_time += response_time

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary representation."""
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
            ],
        }
