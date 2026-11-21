"""Agy (Google Antigravity CLI) provider implementation.

The ``agy`` CLI accepts ``-p``, ``--model`` and ``--output-format`` flags.
Because ``agy`` is an agentic CLI, it is always run with ``--mode plan`` so it
behaves as a plain text generator.

Example:
    >>> provider = AgyProvider()
    >>> response = await provider.generate(LLMRequest(prompt="Hello"))

"""

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncGenerator
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
from mcp_server_llm_cli_runner.providers.quota import QuotaManager, SimpleMetrics
from mcp_server_llm_cli_runner.utils.logging import get_logger

logger = get_logger(__name__)

AGY_BINARY_ENV = "AGY_CLI_PATH"

_AUTH_MARKERS = (
    "not eligible",
    "eligibility check failed",
    "verify your account",
    "sign in",
    "unauthorized",
    "auth",
)
_RATE_LIMIT_MARKERS = ("rate limit", "quota")


class AgyProvider(LLMProvider):
    """Provider for Google's Antigravity CLI (``agy``)."""

    PROVIDER_TYPE = ProviderType.AGY
    PROBE_TIMEOUT = 10.0  # seconds for ``agy --version`` probes

    def __init__(
        self,
        config: ProviderConfig | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        daily_quota: int | None = None,
    ) -> None:
        """Initialize Agy provider; an empty model lets agy pick its default.

        Explicit ``model``/``temperature`` win, else the values from ``config``
        (``model_name`` and ``provider_specific["default_temperature"]``), else
        the defaults (empty model, temperature 0.7).
        """
        if model is None:
            model = (config.model_name if config is not None else None) or ""
        if temperature is None:
            configured = (
                config.provider_specific.get("default_temperature")
                if config is not None
                else None
            )
            temperature = float(configured) if configured is not None else 0.7
        if config is None:
            config = ProviderConfig(
                name="agy",
                provider_type=ProviderType.AGY,
                models=[model] if model else [],
                provider_specific={
                    "default_model": model,
                    "default_temperature": temperature,
                },
            )

        super().__init__(config)
        self.name = "agy"

        self.capabilities = {
            ProviderCapabilities.STREAMING,
            ProviderCapabilities.ASYNC_GENERATION,
        }

        self.model = model
        self.temperature = temperature
        self.max_tokens = 4096

        self._usage_stats = UsageStats(provider_name="agy")
        self.cache_metrics = CacheMetrics()
        self.metrics = SimpleMetrics()

        # CLI-based LLMs need longer timeout (reads project context on startup)
        self.timeout = config.timeout if config else 120
        self.retry_attempts = max_retries
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if daily_quota is not None:
            self.quota_manager = QuotaManager(daily_limit=daily_quota)
        else:
            self.quota_manager = QuotaManager()

        logger.info(
            f"Initialized {self._display_name()} provider",
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _cli_binary(self) -> str:
        """Return the agy executable, overridable via AGY_CLI_PATH."""
        return os.environ.get(AGY_BINARY_ENV) or "agy"

    def _display_name(self) -> str:
        """Return the name used in user-facing messages."""
        return "agy (Antigravity)"

    @property
    def usage_stats(self) -> UsageStats:
        """Get usage statistics."""
        return self._usage_stats

    def _reserve_quota_or_raise(self) -> int:
        """Reserve one request of quota or raise RateLimitError when exhausted."""
        token = self.quota_manager.reserve()
        if token is None:
            raise RateLimitError(
                f"{self._display_name()} daily quota exhausted",
                provider=self.name,
                retry_after=86400,  # Retry after 24 hours (daily quota)
            )
        return token

    def _build_command(
        self,
        prompt: str,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> list[str]:
        """Build agy CLI command (stream is ignored: stream-json is unsupported)."""
        if system_prompt:
            logger.warning("agy does not support 'system_prompt'. It will be ignored.")

        cmd = [self._cli_binary(), "--mode", "plan"]
        if model and model != "default":
            cmd.extend(["--model", model])
        cmd.extend(["--output-format", "json", "-p", prompt])
        logger.debug(
            "Built agy command",
            binary=cmd[0],
            model=model,
            prompt_chars=len(prompt),
        )
        return cmd

    def _get_model_pricing(self, model: str) -> dict[str, Any]:
        """Agy is a subscription/quota CLI with no per-token price."""
        return {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "tier": "subscription",
        }

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the agy CLI.

        Args:
            request: The LLM request containing prompt and parameters

        Returns:
            Structured response from agy

        Raises:
            ProviderError: If generation fails
            RateLimitError: If rate limit exceeded

        """
        start_time = time.time()

        try:
            # Validate request (before reserving so invalid requests use no quota)
            self._validate_request(request)

            model = request.model or self.model
            temperature = request.temperature or self.temperature
            max_tokens = min(request.max_tokens or self.max_tokens, self.max_tokens)

            cmd = self._build_command(
                prompt=request.prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=request.system_prompt,
            )

            # Atomically reserve quota; retries share this single reservation
            token = self._reserve_quota_or_raise()

            # Execute with retries; any failure (incl. cancellation) releases
            try:
                result = await self._execute_with_retries(cmd, request)
            except BaseException:
                self.quota_manager.release(token)
                raise

            response_time = time.time() - start_time
            response_content = result.get("response", "")

            # Extract actual token counts from stats if available
            # Structure: stats.models.[model_name].tokens.{prompt, candidates, total}
            stats = result.get("stats", {})
            models_stats = stats.get("models", {})

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

            if actual_tokens:
                tokens_used = actual_tokens["total"]
            else:
                tokens_used = self._estimate_tokens(request.prompt, response_content)

            cost = self._calculate_cost(tokens_used, model)

            self.metrics.record_request(
                response_time=response_time,
                token_count=tokens_used,
                cost=cost,
            )

            response = LLMResponse(
                content=response_content,
                provider=self.name,
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
            raise
        except ProviderError as e:
            response_time = time.time() - start_time
            self.metrics.record_request(
                response_time=response_time,
                token_count=0,
                cost=0.0,
                error=str(e),
            )

            if hasattr(e, "retry_attempts") and e.retry_attempts:
                logger.error(
                    f"{self._display_name()} generation failed: {e}",
                    error=e,
                    model=request.model,
                    retry_attempts=e.retry_attempts,
                )
            else:
                logger.exception(
                    f"{self._display_name()} generation failed: {e}",
                    error=e,
                    model=request.model,
                )
            raise
        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.record_request(
                response_time=response_time,
                token_count=0,
                cost=0.0,
                error=str(e),
            )

            error_msg = f"{self._display_name()} generation failed: {e}"
            logger.exception(error_msg, error=e, model=request.model)
            raise ProviderError(error_msg, provider=self.name) from e

    async def stream_generate(self, request: LLMRequest) -> list[StreamingResponse]:
        """Return the full response as a single chunk.

        agy's stream-json format is not yet supported.
        """
        response = await self.generate(request)
        return [
            StreamingResponse(
                content=response.content,
                provider="agy",
                model=response.model,
                is_final=True,
                chunk_index=0,
            )
        ]

    async def generate_streaming_response(
        self, request: LLMRequest
    ) -> AsyncGenerator[StreamingResponse, None]:
        """Yield the full response as one chunk (stream-json not yet supported)."""
        for chunk in await self.stream_generate(request):
            yield chunk

    async def estimate_cost(self, request: LLMRequest) -> CostEstimate:
        """Estimate cost for an agy request."""
        try:
            model = request.model or self.model
            input_tokens = self._estimate_tokens(request.prompt)
            output_tokens = min(request.max_tokens or 1000, self.max_tokens)
            pricing = self._get_model_pricing(model)

            input_cost = input_tokens * pricing["input_cost_per_token"]
            output_cost = output_tokens * pricing["output_cost_per_token"]
            total_cost = input_cost + output_cost

            return CostEstimate(
                provider_name=self.name,
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
            raise ProviderError(error_msg, provider=self.name) from e

    def get_usage_stats(self) -> dict[str, Any]:
        """Get comprehensive usage statistics."""
        base_stats = self.metrics.to_dict()
        provider_stats = {
            "provider_name": self.name,
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

    @staticmethod
    async def _kill_process(
        process: "asyncio.subprocess.Process | None",
    ) -> None:
        """Kill and reap a subprocess, tolerating unbound or already-exited ones."""
        if process is None:
            return
        try:
            process.kill()
        except ProcessLookupError:
            pass
        # Shield the reap so a second cancellation cannot leave a zombie child.
        waiter = asyncio.ensure_future(process.wait())
        while not waiter.done():
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                continue
            except Exception:
                break

    async def _communicate(
        self,
        process: "asyncio.subprocess.Process",
        timeout: float | None = None,
        kill_on_timeout: bool = False,
    ) -> tuple[bytes, bytes]:
        """Run process.communicate() with a timeout, reaping the child on abort.

        On timeout the caller's handler kills the process (unless
        ``kill_on_timeout``); on any other interruption (including
        cancellation) the child is killed and reaped here before the
        exception propagates.
        """
        try:
            return await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout if timeout is None else timeout,
            )
        except TimeoutError:
            if kill_on_timeout:
                await self._kill_process(process)
            raise
        except BaseException:
            await self._kill_process(process)
            raise

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

    @staticmethod
    def _parse_payload(*outputs: str) -> dict[str, Any] | None:
        """Return the first JSON object found in the given outputs, in order."""
        for text in outputs:
            match = re.search(r"({.*})", text, re.DOTALL)
            if not match:
                continue
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _raise_for_error(error_msg: str) -> None:
        """Raise auth/rate-limit errors for recognised failure text."""
        lowered = error_msg.lower()
        if any(marker in lowered for marker in _AUTH_MARKERS):
            raise ConfigurationError(f"agy authentication failed: {error_msg}")
        if any(marker in lowered for marker in _RATE_LIMIT_MARKERS):
            raise RateLimitError(
                f"agy rate limit exceeded: {error_msg}",
                provider="agy",
                retry_after=3600,
            )

    @staticmethod
    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        """Map agy JSON onto the structure ``generate`` consumes."""
        response = str(payload.get("response") or "").rstrip()
        result: dict[str, Any] = {"response": response}
        usage = payload.get("usage")
        if isinstance(usage, dict):
            prompt = usage.get("input_tokens", 0)
            completion = usage.get("output_tokens", 0)
            result["stats"] = {
                "models": {
                    "agy": {
                        "tokens": {
                            "prompt": prompt,
                            "candidates": completion,
                            "total": usage.get("total_tokens") or prompt + completion,
                        }
                    }
                }
            }
        if payload.get("conversation_id"):
            result["session_id"] = payload["conversation_id"]
        return result

    async def _execute_with_retries(
        self,
        cmd: list[str],
        request: LLMRequest,
    ) -> dict[str, Any]:
        """Execute the agy command with retry logic."""
        last_error: ProviderError | None = None
        retry_attempts_log: list[dict[str, Any]] = []

        for attempt in range(self.retry_attempts):
            attempt_start = time.time()
            process: asyncio.subprocess.Process | None = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await self._communicate(process)
                stdout_text = stdout.decode("utf-8") if stdout else ""
                stderr_text = stderr.decode("utf-8") if stderr else ""

                # JSON goes to stdout on success but to stderr on failure.
                payload = self._parse_payload(stdout_text, stderr_text)
                failed = process.returncode != 0 or (
                    payload is not None and payload.get("status") == "ERROR"
                )
                if not failed:
                    result: dict[str, Any] = (
                        self._normalize(payload)
                        if payload is not None
                        else {"response": stdout_text.strip(), "format": "text"}
                    )
                    if retry_attempts_log:
                        result["retry_attempts"] = retry_attempts_log
                    return result

                error_msg = (
                    (payload or {}).get("error")
                    or stderr_text.strip()
                    or stdout_text.strip()
                )
                self._raise_for_error(error_msg)
                last_error = ProviderError(
                    f"agy CLI error: {error_msg}", provider="agy"
                )
            except (ConfigurationError, RateLimitError):
                raise
            except FileNotFoundError as err:
                raise ConfigurationError(
                    "agy (Antigravity) CLI not installed or not found. "
                    "Please install it first."
                ) from err
            except TimeoutError:
                await self._kill_process(process)
                last_error = ProviderError(
                    f"agy request timed out after {self.timeout}s", provider="agy"
                )
            except Exception as e:
                last_error = ProviderError(f"agy execution error: {e}", provider="agy")

            retry_attempts_log.append(
                {
                    "attempt": attempt + 1,
                    "error": str(last_error),
                    "duration_ms": int((time.time() - attempt_start) * 1000),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            if attempt < self.retry_attempts - 1:
                wait_time = self.retry_delay * (2**attempt)
                logger.warning(
                    f"agy attempt {attempt + 1} failed, retrying in {wait_time}s",
                    error=str(last_error),
                )
                await asyncio.sleep(wait_time)

        error = ProviderError(
            f"agy generation failed after {self.max_retries} attempts",
            provider="agy",
        )
        error.retry_attempts = retry_attempts_log
        raise error from last_error

    async def health_check(self) -> dict[str, Any]:
        """Check agy availability via ``agy --version``."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._cli_binary(),
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await self._communicate(
                process, timeout=self.PROBE_TIMEOUT, kill_on_timeout=True
            )
            if process.returncode == 0:
                return {
                    "status": "healthy",
                    "version": stdout.decode("utf-8").strip(),
                    "provider": "agy",
                    "timestamp": datetime.now().isoformat(),
                    "capabilities": [cap.value for cap in self.capabilities],
                    "models": [self.model],
                }
            return {
                "status": "unhealthy",
                "error": stderr.decode("utf-8").strip() or "Unknown error",
                "provider": "agy",
                "timestamp": datetime.now().isoformat(),
            }
        except FileNotFoundError:
            return {
                "status": "unavailable",
                "error": f"{self._display_name()} CLI not installed",
                "provider": "agy",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "provider": "agy",
                "timestamp": datetime.now().isoformat(),
            }

    def validate_config(self, config: ProviderConfig) -> bool:
        """Validate provider configuration (a model is optional for agy)."""
        return config.name == "agy" and config.provider_type == ProviderType.AGY

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
        """Check if the agy provider is available."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._cli_binary(),
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=self.PROBE_TIMEOUT)
            except BaseException:
                await self._kill_process(process)
                raise
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
