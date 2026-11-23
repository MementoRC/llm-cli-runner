"""Lean MCP Interface for LLM CLI Runner - Dynamic Tool Discovery.

This module implements the meta-tool pattern to reduce context consumption
while exposing LLM CLI operations through multiple providers (agy, LLaMA, claude (explicit-only)).

Meta-Tool Pattern:
- Expose only 3 standard meta-tools with minimal definitions
- Tools are discovered dynamically on-demand
- Full schemas retrieved only when needed
- Zero functionality loss with massive context savings
"""

import asyncio
import logging
import warnings
from typing import Any

from fastmcp import FastMCP

from mcp_server_llm_cli_runner.core.errors import ProviderError
from mcp_server_llm_cli_runner.core.models import (
    map_legacy_model,
    resolve_provider_name,
)
from mcp_server_llm_cli_runner.providers.manager import ProviderManager

logger = logging.getLogger(__name__)

_DEPRECATION_MESSAGE = (
    "Provider 'gemini' is deprecated and is served by 'agy' (Antigravity CLI); "
    "use provider='agy' instead"
)
_deprecation_warned = False


def _warn_gemini_deprecated() -> None:
    """Emit the gemini->agy deprecation warning once per process."""
    global _deprecation_warned
    if _deprecation_warned:
        return
    _deprecation_warned = True
    logger.warning(_DEPRECATION_MESSAGE)
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)


class LeanMCPInterface:
    """Lean MCP Interface implementing the meta-tool pattern for dynamic tool discovery.

    Exposes only 3 compact meta-tools (~500 tokens) with on-demand discovery
    instead of verbose tool definitions.
    """

    def __init__(self, config_manager: Any):
        """Initialize lean MCP interface.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.app = FastMCP("llm-cli-runner")

        # Initialize provider manager
        self.manager = ProviderManager(config_manager)

        # Tool registry: maps tool names to their implementations and metadata
        self.tool_registry = self._build_tool_registry()

        # Setup the 3 meta-tools
        self._setup_meta_tools()

    async def initialize(self) -> None:
        """Initialize the interface and its components."""
        await self.manager.initialize()
        self._init_providers()

    def _configured_model(self, name: str) -> str:
        """Return the configured model for a provider ("" when default/unset)."""
        cfg = self.config_manager.get_provider_config(name) or {}
        model = cfg.get("model_name") or ""
        return "" if model == "default" else str(model)

    def _init_providers(self) -> None:
        """Initialize and register available provider instances."""
        enabled = self.config_manager.get_enabled_providers()

        if "gemini" in enabled:
            logger.info("gemini is deprecated; served by agy")

        if "agy" in enabled or "gemini" in enabled:
            try:
                from mcp_server_llm_cli_runner.providers.agy import AgyProvider

                agy_model = self._configured_model("agy")
                if not agy_model and "gemini" in enabled:
                    gemini_model = self._configured_model("gemini")
                    if gemini_model:
                        agy_model, model_warning = map_legacy_model(gemini_model)
                        agy_model = agy_model or ""
                        logger.info(
                            f"Mapped gemini model '{gemini_model}' to agy model "
                            f"'{agy_model or 'default'}'"
                        )
                        if model_warning:
                            logger.warning(model_warning)

                self.manager.registry.register_provider(AgyProvider)
                self.manager.register_provider("agy", AgyProvider(model=agy_model))
                logger.info("Initialized and registered agy provider")
            except Exception as e:
                logger.warning(f"Failed to initialize agy provider: {e}")

        if "claude" in enabled or "anthropic" in enabled:
            try:
                from mcp_server_llm_cli_runner.providers.claude import ClaudeProvider

                claude_name = "claude" if "claude" in enabled else "anthropic"
                if "claude" in enabled and "anthropic" in enabled:
                    logger.warning(
                        "Both 'claude' and 'anthropic' are enabled; they are the "
                        "same provider. Using the 'claude' entry, ignoring 'anthropic'"
                    )
                claude_cfg = self.config_manager.get_provider_config(claude_name) or {}
                claude_kwargs: dict[str, Any] = {
                    "model": self._configured_model(claude_name) or None,
                    "effort": claude_cfg.get("effort") or None,
                    "cli_path": claude_cfg.get("cli_path") or None,
                }
                if claude_cfg.get("default_temperature") is not None:
                    claude_kwargs["temperature"] = float(
                        claude_cfg["default_temperature"]
                    )
                if claude_cfg.get("timeout_seconds"):
                    claude_kwargs["timeout"] = int(claude_cfg["timeout_seconds"])
                self.manager.registry.register_provider(ClaudeProvider)
                self.manager.register_provider(
                    "claude", ClaudeProvider(**claude_kwargs)
                )
                logger.info("Initialized and registered claude provider")
            except Exception as e:
                logger.warning(f"Failed to initialize claude provider: {e}")

        if "llama" in enabled:
            try:
                from mcp_server_llm_cli_runner.providers.llama import LLaMAProvider

                # Register the class
                self.manager.registry.register_provider(LLaMAProvider)

                # Create and register the instance (uses built-in defaults)
                provider = LLaMAProvider()
                self.manager.register_provider("llama", provider)
                logger.info("Initialized and registered LLaMA provider")
            except Exception as e:
                logger.warning(f"Failed to initialize LLaMA provider: {e}")

    def _build_tool_registry(self) -> dict[str, dict[str, Any]]:
        """Build comprehensive tool registry with metadata for dynamic discovery."""
        registry = {}

        # LLM Completion Tool
        registry["llm_complete"] = {
            "implementation": self._llm_complete,
            "description": "Execute LLM completion with provider selection and streaming support",
            "domain": "llm",
            "complexity": "focused",
            "schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to the LLM",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Provider to use (agy (alias gemini), llama, claude (alias anthropic)). If omitted, prefers the configured default_provider and auto-routes if it is unavailable; a named provider fails with PROVIDER_UNAVAILABLE if it is unavailable (no fallback). 'gemini' is a deprecated alias for agy. 'claude' (Claude Code CLI, subscription login, billed to your Claude plan) is explicit-only: it is never auto-selected or used as a fallback",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model name (provider-specific)",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (0.0-2.0)",
                        "default": 0.7,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                    },
                    "stream": {
                        "type": "boolean",
                        "description": "Enable streaming response",
                        "default": False,
                    },
                },
                "required": ["prompt"],
            },
            "examples": [
                {"prompt": "What is the capital of France?", "provider": "agy"},
                {
                    "prompt": "Explain quantum computing",
                    "provider": "llama",
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
            ],
        }

        # List Providers Tool
        registry["list_providers"] = {
            "implementation": self._list_providers,
            "description": "List available LLM providers and their status (claude is explicit-only: never auto-routed)",
            "domain": "provider",
            "complexity": "focused",
            "schema": {"type": "object", "properties": {}, "required": []},
            "examples": [{}],
        }

        # Provider Info Tool
        registry["provider_info"] = {
            "implementation": self._provider_info,
            "description": "Get detailed information about a specific provider",
            "domain": "provider",
            "complexity": "focused",
            "schema": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name (agy (alias gemini), llama, claude (alias anthropic; explicit-only)). 'gemini' is a deprecated alias for agy",
                    }
                },
                "required": ["provider"],
            },
            "examples": [
                {"provider": "agy"},
                {"provider": "llama"},
                {"provider": "claude"},
            ],
        }

        return registry

    async def _send_notification(
        self, notification_type: str, data: dict[str, Any]
    ) -> None:
        """Send MCP notification if server supports it."""
        try:
            send_notification = getattr(self.app, "send_notification", None)
            if send_notification is not None and callable(send_notification):
                result = send_notification(notification_type, data)
                if asyncio.iscoroutine(result):
                    await result
            else:
                logger.debug(f"Notification [{notification_type}]: {data}")
        except Exception as e:
            logger.warning(f"Failed to send notification {notification_type}: {e}")

    async def _ensure_agy_available(self) -> None:
        """Raise a clear error when a redirected gemini call cannot be served by agy."""
        instance = self.manager.get_provider("agy")
        if instance is None:
            msg = (
                "Provider 'gemini' is deprecated and served by 'agy', but agy is not "
                "installed or not available; install the Antigravity CLI or set "
                "AGY_CLI_PATH"
            )
            raise ProviderError(
                msg,
                provider="agy",
                error_code="PROVIDER_UNAVAILABLE",
                context={"requested_provider": "gemini"},
            )
        check = getattr(instance, "is_available", None)
        if check is not None and not await check():
            msg = "agy provider unavailable (CLI error or daily quota exhausted)"
            raise ProviderError(
                msg,
                provider="agy",
                error_code="PROVIDER_UNAVAILABLE",
                context={"requested_provider": "gemini"},
            )

    def _default_provider(self) -> str:
        """Return the configured default provider (falls back to llama)."""
        try:
            name = self.config_manager.get_default_provider()
        except Exception:
            return "llama"
        return name if isinstance(name, str) and name else "llama"

    # Tool implementations
    async def _llm_complete(
        self,
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Execute LLM completion using ProviderManager."""
        import time

        operation_start_time = time.time()
        extras: dict[str, str] = {}
        requested_provider = provider
        default_provider = None if requested_provider else self._default_provider()
        provider, aliased = resolve_provider_name(
            requested_provider or default_provider or "llama"
        )
        aliased = aliased and requested_provider is not None
        # Label for notifications until a response names the actual provider
        label = provider if requested_provider else (default_provider or "auto")
        if aliased:
            _warn_gemini_deprecated()
            extras["deprecated_alias"] = "gemini->agy"
            original_model = model
            model, model_warning = map_legacy_model(model)
            if model_warning:
                logger.warning(model_warning)
            if (original_model or None) != model and original_model not in (
                "",
                "default",
            ):
                extras["model_mapped"] = f"{original_model}->{model or 'default'}"
        operation_id = f"{label}_{int(operation_start_time * 1000)}"

        try:
            if aliased:
                await self._ensure_agy_available()
            # Send operation started notification
            await self._send_notification(
                "llm/started",
                {
                    "operation_id": operation_id,
                    "provider": label,
                    "model": model,
                    "timestamp": time.time(),
                },
            )

            # Lazy import to avoid circular dependency
            from mcp_server_llm_cli_runner.core.models import LLMRequest

            # Create LLM request
            request = LLMRequest(
                prompt=prompt,
                provider=provider if requested_provider else None,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens or 1000,
                system_prompt=None,
                metadata={"preferred_provider": default_provider}
                if default_provider
                else {},
            )

            # Create periodic progress notifications
            async def send_progress_heartbeat():
                heartbeat_interval = 2.0
                while True:
                    await asyncio.sleep(heartbeat_interval)
                    elapsed = time.time() - operation_start_time
                    await self._send_notification(
                        "llm/progress",
                        {
                            "operation_id": operation_id,
                            "provider": label,
                            "elapsed_seconds": elapsed,
                            "timestamp": time.time(),
                        },
                    )

            heartbeat_task = asyncio.create_task(send_progress_heartbeat())

            try:
                # Use ProviderManager to route and execute request
                response = await self.manager.route_request(request)

                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

                used_provider = getattr(response, "provider", None) or label

                # Send completion notification
                await self._send_notification(
                    "llm/completed",
                    {
                        "operation_id": operation_id,
                        "provider": used_provider,
                        "model": response.model,
                        "tokens_used": response.tokens_used,
                        "response_time_ms": response.response_time_ms,
                        "timestamp": time.time(),
                    },
                )

                result = {
                    "provider": used_provider,
                    "model": response.model,
                    "completion": response.content,
                    "success": response.success,
                    "tokens_used": response.tokens_used,
                    "cost": response.cost,
                    "response_time_ms": response.response_time_ms,
                    "metadata": response.metadata,
                }

                if "retry_attempts" in response.metadata:
                    result["retry_attempts"] = response.metadata["retry_attempts"]

                result.update(extras)
                return result

            except Exception as provider_error:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

                retry_info = getattr(provider_error, "retry_attempts", None)
                if retry_info is not None:
                    for attempt in retry_info:
                        await self._send_notification(
                            "llm/retry",
                            {
                                "operation_id": operation_id,
                                "provider": label,
                                "attempt": attempt["attempt"],
                                "error": attempt["error"],
                                "duration_ms": attempt["duration_ms"],
                                "timestamp": attempt["timestamp"],
                            },
                        )

                await self._send_notification(
                    "llm/failed",
                    {
                        "operation_id": operation_id,
                        "provider": label,
                        "error": str(provider_error),
                        "retry_attempts": retry_info,
                        "timestamp": time.time(),
                    },
                )
                raise

        except Exception as e:
            logger.error(f"Error in llm_complete: {e}")
            error_result = {"error": str(e)}
            error_code = getattr(e, "error_code", None)
            if error_code:
                error_result["error_code"] = error_code
            retry_attempts = getattr(e, "retry_attempts", None)
            if retry_attempts is not None:
                error_result["attempts"] = retry_attempts
            return error_result

    async def _probe_health(self, name: str) -> dict[str, Any]:
        """Probe a provider and return its health dict (includes ``available``)."""
        return (await self.manager.health_monitor.probe(name)).to_dict()

    async def _list_providers(self) -> dict[str, Any]:
        """List available providers using ProviderManager."""
        try:
            initialized = self.manager.registry.list_providers()
            configured = self.config_manager.get_enabled_providers()

            visible = [p for p in initialized if p != "gemini"]
            # Probe concurrently; each probe is bounded by the monitor's
            # PROBE_TIMEOUT and reports its own failure, so one slow or broken
            # provider cannot delay or fail the rest.
            results = await asyncio.gather(*(self._probe_health(n) for n in visible))
            status = dict(zip(visible, results, strict=True))
            return {
                "providers": visible,
                "count": len(visible),
                "configured": [p for p in configured if p != "gemini"],
                "health_status": status,
                "deprecated": {"gemini": "agy"},
            }
        except Exception as e:
            logger.error(f"Error in list_providers: {e}")
            return {"error": str(e)}

    async def _provider_info(self, provider: str) -> dict[str, Any]:
        """Get provider information using ProviderManager."""
        try:
            requested = provider
            provider, aliased = resolve_provider_name(provider)
            provider_instance = self.manager.get_provider(provider)

            if not provider_instance:
                return {"error": f"Provider '{provider}' not initialized"}

            # Same probe as list_providers, so availability and status agree
            health = await self._probe_health(provider)

            info = {
                "provider": provider,
                "initialized": True,
                "available": health["available"],
                "health": health,
                "stats": self.manager.get_provider_stats(provider),
                "metadata": getattr(provider_instance, "metadata", {}),
            }
            if aliased:
                info.update(
                    {
                        "deprecated": True,
                        "redirected_to": provider,
                        "requested_provider": requested,
                    }
                )
            return info
        except Exception as e:
            logger.error(f"Error in provider_info: {e}")
            return {"error": str(e)}

    def _setup_meta_tools(self):
        """Setup the 3 meta-tools for dynamic discovery."""

        @self.app.tool(
            description="Discover llm-cli-runner tools (3 total) for LLM completions and provider management. USE WHEN: running LLM queries, checking providers, multi-provider access"
        )
        def discover_tools(pattern: str = "") -> dict[str, Any]:
            tools = []
            for name, info in self.tool_registry.items():
                if pattern and pattern.strip() and pattern.lower() not in name.lower():
                    continue
                tools.append(
                    {
                        "name": name,
                        "description": info["description"],
                        "domain": info.get("domain", "llm"),
                        "complexity": info.get("complexity", "focused"),
                    }
                )
            return {
                "available_tools": tools,
                "total_tools": len(self.tool_registry),
                "filtered_count": len(tools),
            }

        @self.app.tool(
            description="Get full specification for specific llm-cli-runner tool including schema and examples."
        )
        def get_tool_spec(tool_name: str) -> dict[str, Any]:
            if tool_name not in self.tool_registry:
                return {"error": f"Tool '{tool_name}' not found"}
            tool_info = self.tool_registry[tool_name]
            return {
                "name": tool_name,
                "schema": tool_info["schema"],
                "examples": tool_info["examples"],
            }

        @self.app.tool(description="Execute llm-cli-runner tool with parameters.")
        async def execute_tool(
            tool_name: str, parameters: dict[str, Any]
        ) -> dict[str, Any]:
            if tool_name not in self.tool_registry:
                return {"error": f"Tool '{tool_name}' not found"}
            tool_info = self.tool_registry[tool_name]
            tool_func = tool_info["implementation"]
            try:
                result = tool_func(**parameters)
                if asyncio.iscoroutine(result):
                    result = await result
                return {"tool": tool_name, "status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                return {"tool": tool_name, "status": "error", "error": str(e)}

    def get_app(self) -> FastMCP:
        """Get the FastMCP app instance."""
        return self.app


async def create_lean_interface(config_manager: Any) -> FastMCP:
    """Create and initialize a lean MCP interface."""
    lean_interface = LeanMCPInterface(config_manager)
    await lean_interface.initialize()
    return lean_interface.get_app()
