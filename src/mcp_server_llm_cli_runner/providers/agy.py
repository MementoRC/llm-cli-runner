"""Agy (Google Antigravity CLI) provider implementation.

Reuses the Gemini CLI provider machinery, since the ``agy`` CLI accepts the
same ``-p``, ``--model`` and ``--output-format`` flags. Because ``agy`` is an
agentic CLI, it is always run with ``--mode plan`` so it behaves as a plain
text generator.

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

from mcp_server_llm_cli_runner.core.errors import (
    ConfigurationError,
    ProviderError,
    RateLimitError,
)
from mcp_server_llm_cli_runner.core.models import (
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderType,
    StreamingResponse,
    UsageStats,
)
from mcp_server_llm_cli_runner.providers.gemini import GeminiProvider
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


class AgyProvider(GeminiProvider):
    """Provider for Google's Antigravity CLI (``agy``)."""

    PROVIDER_TYPE = ProviderType.AGY

    def __init__(
        self,
        config: ProviderConfig | None = None,
        model: str = "",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        daily_quota: int | None = None,
    ) -> None:
        """Initialize Agy provider; an empty model lets agy pick its default."""
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

        super().__init__(
            config,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
            daily_quota=daily_quota,
        )
        self.name = "agy"
        self._usage_stats = UsageStats(provider_name="agy")

    def _cli_binary(self) -> str:
        """Return the agy executable, overridable via AGY_CLI_PATH."""
        return os.environ.get(AGY_BINARY_ENV) or "agy"

    def _display_name(self) -> str:
        """Return the name used in user-facing messages."""
        return "agy (Antigravity)"

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
        logger.debug("Built agy command", command=" ".join(cmd))
        return cmd

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the agy CLI."""
        response = await super().generate(request)
        response.provider = "agy"
        return response

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
        """Map agy JSON onto the structure GeminiProvider.generate consumes."""
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
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout
                )
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
        result = await super().health_check()
        result["provider"] = "agy"
        return result

    def validate_config(self, config: ProviderConfig) -> bool:
        """Validate provider configuration (a model is optional for agy)."""
        return config.name == "agy" and config.provider_type == ProviderType.AGY
