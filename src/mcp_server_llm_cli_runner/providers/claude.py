"""Claude (Claude Code CLI) provider implementation.

Wraps ``claude -p`` (print mode) as a text-only generator. Authentication is the
user's existing Claude subscription login (OAuth) handled by the CLI itself; no
API key is read, required or passed. The provider is explicit-only: it is never
auto-selected by routing because every call is billed to the user's Claude plan.

The child process is fully isolated: no hooks, CLAUDE.md, skills, plugins, MCP
servers or tools, an empty temporary working directory, and no Claude-session
environment variables.

Example:
    >>> provider = ClaudeProvider()
    >>> response = await provider.generate(LLMRequest(prompt="Hello"))

"""

import asyncio
import json
import os
import re
import shutil
import tempfile
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

CLAUDE_BINARY_ENV = "CLAUDE_CLI_PATH"
DEFAULT_MODEL = "sonnet"
VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Env vars that mark a Claude Code session; stripped so the child does not
# believe it is nested inside one.
_NESTED_ENV_EXACT = "CLAUDECODE"
_NESTED_ENV_PREFIX = "CLAUDE_CODE_"

_AUTH_MARKERS = (
    "not logged in",
    "please run /login",
    "/login",
    "log in",
    "authenticat",
    "unauthorized",
    "oauth",
    "invalid api key",
    "credentials",
)
_RATE_LIMIT_MARKERS = ("rate limit", "usage limit", "limit reached", "quota", "429")
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\[\]-]*")
_TRANSIENT_MARKERS = (
    "overloaded",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "api error: 5",
    "529",
    "500",
    "502",
    "503",
    "504",
)


class _TransientError(ProviderError):
    """A clearly transient failure that is safe to retry (spawn failure, 5xx)."""


class ClaudeProvider(LLMProvider):
    """Provider for the Claude Code CLI in print mode (subscription login)."""

    PROVIDER_TYPE = ProviderType.ANTHROPIC
    PROBE_TIMEOUT = 10.0  # seconds for ``claude --version`` probes

    def __init__(
        self,
        config: ProviderConfig | None = None,
        model: str | None = None,
        temperature: float | None = None,
        effort: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        daily_quota: int | None = None,
    ) -> None:
        """Initialize Claude provider.

        Explicit ``model``/``temperature``/``effort`` win, else the values from
        ``config`` (``model_name`` and ``provider_specific["default_temperature"]``
        / ``["effort"]``), else the defaults (model "sonnet", temperature 0.7,
        no effort flag).
        """
        specific: dict[str, Any] = config.provider_specific if config else {}
        if model is None:
            model = (config.model_name if config is not None else None) or ""
        if not model or model == "default":
            model = DEFAULT_MODEL
        if temperature is None:
            configured = specific.get("default_temperature")
            temperature = float(configured) if configured is not None else 0.7
        if effort is None:
            effort = specific.get("effort") or None
        if effort is not None and effort not in VALID_EFFORTS:
            raise ConfigurationError(
                f"Invalid claude effort '{effort}'. Valid: {', '.join(VALID_EFFORTS)}"
            )
        if config is None:
            config = ProviderConfig(
                name="claude",
                provider_type=ProviderType.ANTHROPIC,
                models=[model],
                provider_specific={
                    "default_model": model,
                    "default_temperature": temperature,
                },
            )

        super().__init__(config)
        self.name = "claude"

        self.capabilities = {
            ProviderCapabilities.STREAMING,
            ProviderCapabilities.ASYNC_GENERATION,
        }

        self.model = model
        self.temperature = temperature
        self.effort = effort
        self.max_tokens = 4096
        self._cli_path: str | None = specific.get("cli_path") or None

        self._usage_stats = UsageStats(provider_name="claude")
        self.cache_metrics = CacheMetrics()
        self.metrics = SimpleMetrics()

        self.timeout = config.timeout if config else 120
        self.retry_attempts = max_retries
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if daily_quota is not None:
            self.quota_manager = QuotaManager(daily_limit=daily_quota)
        else:
            self.quota_manager = QuotaManager()

        logger.info(
            "Initialized claude (Claude Code CLI) provider",
            model=self.model,
            effort=self.effort,
            max_tokens=self.max_tokens,
        )

    def _cli_binary(self) -> str:
        """Return the claude executable (CLAUDE_CLI_PATH, config, else PATH).

        Raises:
            ProviderError: PROVIDER_UNAVAILABLE when no executable is found.

        """
        configured = os.environ.get(CLAUDE_BINARY_ENV) or self._cli_path
        found = shutil.which(configured or "claude")
        if found is None:
            raise ProviderError(
                f"claude CLI not found ({configured or 'claude'}); install Claude "
                f"Code or set {CLAUDE_BINARY_ENV}",
                provider="claude",
                error_code="PROVIDER_UNAVAILABLE",
            )
        return found

    @staticmethod
    def _child_env() -> dict[str, str]:
        """Copy the environment minus Claude-session markers (keeps HOME/PATH)."""
        return {
            key: value
            for key, value in os.environ.items()
            if key != _NESTED_ENV_EXACT and not key.startswith(_NESTED_ENV_PREFIX)
        }

    @property
    def usage_stats(self) -> UsageStats:
        """Get usage statistics."""
        return self._usage_stats

    def _reserve_quota_or_raise(self) -> int:
        """Reserve one request of quota or raise RateLimitError when exhausted."""
        token = self.quota_manager.reserve()
        if token is None:
            raise RateLimitError(
                "claude daily quota exhausted",
                provider=self.name,
                retry_after=86400,
            )
        return token

    def _build_command(
        self,
        model: str = "",
        system_prompt: str | None = None,
        effort: str | None = None,
    ) -> list[str]:
        """Build the isolated ``claude -p`` command (the prompt goes via stdin)."""
        # NOTE: never add --bare: it forces ANTHROPIC_API_KEY auth and would
        # break the subscription (OAuth) login this provider relies on.
        # --safe-mode already disables hooks, CLAUDE.md, skills, plugins, MCP
        # servers and custom agents while keeping auth working.
        model = model or self.model
        if not _MODEL_RE.fullmatch(model):
            raise ValueError(f"Invalid claude model name: {model!r}")
        cmd = [self._cli_binary(), "-p", "--output-format", "json"]
        # --flag=value form: a value starting with "-" can never parse as a flag.
        cmd.append(f"--model={model}")
        effort = effort or self.effort
        if effort:
            cmd.extend(["--effort", effort])
        if system_prompt:
            cmd.append(f"--system-prompt={system_prompt}")
        cmd.extend(
            [
                "--safe-mode",
                "--tools",
                "",
                "--strict-mcp-config",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--permission-prompts",
                "none",
            ]
        )
        logger.debug("Built claude command", binary=cmd[0], model=model or self.model)
        return cmd

    def _get_model_pricing(self, model: str) -> dict[str, Any]:
        """Claude Code on a subscription has no per-token price."""
        return {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "tier": "subscription",
        }

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the claude CLI.

        Raises:
            ProviderError: If generation fails or the CLI is unavailable
            RateLimitError: If the local daily quota or the plan limit is hit

        """
        start_time = time.time()

        try:
            # Validate before reserving so invalid requests use no quota
            self._validate_request(request)

            model = request.model or self.model
            temperature = request.temperature or self.temperature
            max_tokens = min(request.max_tokens or self.max_tokens, self.max_tokens)

            cmd = self._build_command(model=model, system_prompt=request.system_prompt)

            # Atomically reserve quota; retries share this single reservation
            token = self._reserve_quota_or_raise()

            # Any failure (incl. cancellation) releases the reservation
            try:
                result = await self._execute_with_retries(cmd, request)
            except BaseException:
                self.quota_manager.release(token)
                raise

            response_time = time.time() - start_time
            response_content = result["response"]
            actual_tokens: dict[str, int] | None = result.get("tokens")

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
                    "total_cost_usd": result.get("total_cost_usd"),
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
                response_time_ms=response.response_time_ms,
            )
            return response

        except RateLimitError:
            raise
        except ProviderError as e:
            self.metrics.record_request(
                response_time=time.time() - start_time,
                token_count=0,
                cost=0.0,
                error=str(e),
            )
            logger.error(f"claude generation failed: {e}", error=e, model=request.model)
            raise
        except Exception as e:
            self.metrics.record_request(
                response_time=time.time() - start_time,
                token_count=0,
                cost=0.0,
                error=str(e),
            )
            error_msg = f"claude generation failed: {e}"
            logger.exception(error_msg, error=e, model=request.model)
            raise ProviderError(error_msg, provider=self.name) from e

    async def stream_generate(self, request: LLMRequest) -> list[StreamingResponse]:
        """Return the full response as a single chunk (no streaming support)."""
        response = await self.generate(request)
        return [
            StreamingResponse(
                content=response.content,
                provider="claude",
                model=response.model,
                is_final=True,
                chunk_index=0,
            )
        ]

    async def generate_streaming_response(
        self, request: LLMRequest
    ) -> AsyncGenerator[StreamingResponse, None]:
        """Yield the full response as one chunk."""
        for chunk in await self.stream_generate(request):
            yield chunk

    async def estimate_cost(self, request: LLMRequest) -> CostEstimate:
        """Estimate cost for a claude request (subscription: zero per-token cost)."""
        try:
            input_tokens = self._estimate_tokens(request.prompt)
            output_tokens = min(request.max_tokens or 1000, self.max_tokens)
            return CostEstimate(
                provider_name=self.name,
                estimated_tokens=input_tokens + output_tokens,
                cost_per_token=0.0,
                estimated_cost_usd=0.0,
                confidence_score=0.9,
                estimation_method="pricing_table",
            )
        except Exception as e:
            error_msg = f"Cost estimation failed: {e}"
            logger.exception(error_msg, error=e)
            raise ProviderError(error_msg, provider=self.name) from e

    def get_usage_stats(self) -> dict[str, Any]:
        """Get comprehensive usage statistics."""
        provider_stats = {
            "provider_name": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "capabilities": [cap.value for cap in self.capabilities],
            **self.metrics.to_dict(),
            "total_tokens_consumed": self._usage_stats.total_tokens,
            "total_cost_usd": self._usage_stats.total_cost,
            "average_response_time_ms": self._usage_stats.average_response_time,
            "last_request_time": self._usage_stats.last_updated,
        }
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
        input_data: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        """Run process.communicate() with a timeout, reaping the child on abort.

        On timeout the caller's handler kills the process (unless
        ``kill_on_timeout``); on any other interruption (including
        cancellation) the child is killed and reaped here before the
        exception propagates.
        """
        try:
            return await asyncio.wait_for(
                process.communicate(input_data),
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
        """Calculate cost for token usage (always 0.0 on a subscription)."""
        pricing = self._get_model_pricing(model)
        cost = (
            tokens
            * (pricing["input_cost_per_token"] + pricing["output_cost_per_token"])
            / 2
        )
        return round(cost, 6)

    @staticmethod
    def _parse_payload(text: str) -> dict[str, Any] | None:
        """Parse the JSON result object (or the result entry of a JSON array)."""
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(payload, list):
            results = [
                item
                for item in payload
                if isinstance(item, dict) and item.get("type") == "result"
            ]
            return results[-1] if results else None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _raise_for_error(error_msg: str) -> None:
        """Raise auth/rate-limit errors for recognised failure text."""
        lowered = error_msg.lower()
        if any(marker in lowered for marker in _AUTH_MARKERS):
            raise ConfigurationError(f"claude authentication failed: {error_msg}")
        if any(marker in lowered for marker in _RATE_LIMIT_MARKERS):
            raise RateLimitError(
                f"claude rate limit exceeded: {error_msg}",
                provider="claude",
                retry_after=3600,
            )

    @staticmethod
    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        """Map claude JSON onto what ``generate`` consumes (unknown fields ignored)."""
        result: dict[str, Any] = {"response": str(payload.get("result") or "").strip()}
        usage = payload.get("usage")
        if isinstance(usage, dict):
            prompt = int(usage.get("input_tokens") or 0)
            completion = int(usage.get("output_tokens") or 0)
            if prompt or completion:
                result["tokens"] = {
                    "prompt": prompt,
                    "completion": completion,
                    "total": prompt + completion,
                }
        if payload.get("session_id"):
            result["session_id"] = payload["session_id"]
        if isinstance(payload.get("total_cost_usd"), int | float):
            result["total_cost_usd"] = payload["total_cost_usd"]
        return result

    async def _run_once(self, cmd: list[str], prompt: str) -> dict[str, Any]:
        """Run one isolated CLI invocation and return the normalized result.

        Raises:
            ProviderError / ConfigurationError / RateLimitError on any failure.

        """
        process: asyncio.subprocess.Process | None = None
        with tempfile.TemporaryDirectory(prefix="llm-cli-runner-claude-") as workdir:
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workdir,
                    env=self._child_env(),
                )
                stdout, stderr = await self._communicate(
                    process, input_data=prompt.encode("utf-8")
                )
            except TimeoutError:  # must precede OSError (its base class)
                await self._kill_process(process)
                raise ProviderError(
                    f"claude request timed out after {self.timeout}s",
                    provider="claude",
                ) from None
            except FileNotFoundError:
                raise
            except OSError as e:
                raise _TransientError(
                    f"claude process spawn failed: {e}", provider="claude"
                ) from e

        assert process is not None  # noqa: S101 - narrowing for the type checker
        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        payload = self._parse_payload(stdout_text)

        is_error = payload is not None and bool(payload.get("is_error"))
        if process.returncode != 0 or is_error:
            error_msg = (
                (str(payload.get("result") or "") if payload else "")
                or stderr_text.strip()
                or stdout_text.strip()
                or f"exit code {process.returncode}"
            )
            self._raise_for_error(error_msg)
            error_cls = (
                _TransientError
                if any(m in error_msg.lower() for m in _TRANSIENT_MARKERS)
                else ProviderError
            )
            raise error_cls(f"claude CLI error: {error_msg}", provider="claude")

        if payload is None:
            raise ProviderError(
                f"claude returned malformed JSON output: {stdout_text.strip()[:200]!r}",
                provider="claude",
            )
        return self._normalize(payload)

    async def _execute_with_retries(
        self,
        cmd: list[str],
        request: LLMRequest,
    ) -> dict[str, Any]:
        """Execute the claude command with retry logic."""
        last_error: ProviderError | None = None
        retry_attempts_log: list[dict[str, Any]] = []

        for attempt in range(self.retry_attempts):
            attempt_start = time.time()
            try:
                result = await self._run_once(cmd, request.prompt)
                if retry_attempts_log:
                    result["retry_attempts"] = retry_attempts_log
                return result
            except (ConfigurationError, RateLimitError):
                raise
            except FileNotFoundError as err:
                raise ProviderError(
                    "claude CLI not installed or not found",
                    provider="claude",
                    error_code="PROVIDER_UNAVAILABLE",
                ) from err
            except _TransientError as e:
                last_error = e
            except ProviderError:
                # Timeouts and other CLI failures are NOT retried: the call has
                # probably already consumed the user's Claude plan quota.
                raise
            except Exception as e:
                raise ProviderError(
                    f"claude execution error: {e}", provider="claude"
                ) from e

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
                    f"claude attempt {attempt + 1} failed, retrying in {wait_time}s",
                    error=str(last_error),
                )
                await asyncio.sleep(wait_time)

        error = ProviderError(
            f"claude generation failed after {self.max_retries} attempts: {last_error}",
            provider="claude",
        )
        error.retry_attempts = retry_attempts_log
        raise error from last_error

    def _health_payload(self, status: str, **extra: Any) -> dict[str, Any]:
        return {
            "status": status,
            "provider": "claude",
            "timestamp": datetime.now().isoformat(),
            **extra,
        }

    async def health_check(self) -> dict[str, Any]:
        """Check claude availability via ``claude --version``."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._cli_binary(),
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._child_env(),
            )
            stdout, stderr = await self._communicate(
                process, timeout=self.PROBE_TIMEOUT, kill_on_timeout=True
            )
            if process.returncode == 0:
                return self._health_payload(
                    "healthy",
                    version=stdout.decode("utf-8").strip(),
                    capabilities=[cap.value for cap in self.capabilities],
                    models=[self.model],
                )
            return self._health_payload(
                "unhealthy",
                error=stderr.decode("utf-8").strip() or "Unknown error",
            )
        except (FileNotFoundError, ProviderError):
            return self._health_payload("unavailable", error="claude CLI not installed")
        except Exception as e:
            return self._health_payload("error", error=str(e))

    def validate_config(self, config: ProviderConfig) -> bool:
        """Validate provider configuration (no API key needed)."""
        return (
            config.name in ("claude", "anthropic")
            and config.provider_type == ProviderType.ANTHROPIC
        )

    async def get_usage(self) -> UsageStats:
        """Get current usage statistics."""
        return self._usage_stats

    async def check_quota(self) -> QuotaStatusInfo:
        """Check current (local) quota status."""
        try:
            remaining = (
                self.quota_manager.get_remaining_quota()
                if self.quota_manager.check_quota()
                else 0
            )
            return QuotaStatusInfo(
                provider_name=self.name,
                quota_type="requests",
                current_usage=self.quota_manager.get_current_usage(),
                quota_limit=self.quota_manager.get_quota_limit(),
                quota_remaining=remaining,
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
        """Check if the claude provider is available."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._cli_binary(),
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._child_env(),
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
        except Exception:
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
                    "effort": self.effort,
                    "max_tokens": self.max_tokens,
                    "max_retries": self.max_retries,
                    "retry_delay": self.retry_delay,
                },
            }
        )
        return base_status
