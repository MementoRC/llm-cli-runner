"""Lean MCP Interface for LLM CLI Runner - Dynamic Tool Discovery.

This module implements the meta-tool pattern to reduce context consumption
while exposing LLM CLI operations through multiple providers (Gemini, OpenAI, LLaMA).

Meta-Tool Pattern:
- Expose only 3 standard meta-tools with minimal definitions
- Tools are discovered dynamically on-demand
- Full schemas retrieved only when needed
- Zero functionality loss with massive context savings
"""

import asyncio
import logging
from typing import Any

from fastmcp import FastMCP

from mcp_server_llm_cli_runner.providers.manager import ProviderManager

logger = logging.getLogger(__name__)


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

    def _init_providers(self) -> None:
        """Initialize and register available provider instances."""
        enabled = self.config_manager.get_enabled_providers()

        if "gemini" in enabled:
            try:
                from mcp_server_llm_cli_runner.providers.gemini import GeminiProvider

                # Register the class with the manager's registry
                self.manager.registry.register_provider(GeminiProvider)

                # Create and register the instance (uses built-in defaults)
                provider = GeminiProvider()
                self.manager.register_provider("gemini", provider)
                logger.info("Initialized and registered Gemini provider")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini provider: {e}")

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
                        "description": "Provider to use (gemini, openai, llama)",
                        "default": "llama",
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
                {"prompt": "What is the capital of France?", "provider": "gemini"},
                {
                    "prompt": "Explain quantum computing",
                    "provider": "openai",
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
            ],
        }

        # List Providers Tool
        registry["list_providers"] = {
            "implementation": self._list_providers,
            "description": "List available LLM providers and their status",
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
                        "description": "Provider name (gemini, openai, llama)",
                    }
                },
                "required": ["provider"],
            },
            "examples": [{"provider": "gemini"}, {"provider": "llama"}],
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

    # Tool implementations
    async def _llm_complete(
        self,
        prompt: str,
        provider: str = "llama",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Execute LLM completion using ProviderManager."""
        import time

        operation_start_time = time.time()
        operation_id = f"{provider}_{int(operation_start_time * 1000)}"

        try:
            # Send operation started notification
            await self._send_notification(
                "llm/started",
                {
                    "operation_id": operation_id,
                    "provider": provider,
                    "model": model,
                    "timestamp": time.time(),
                },
            )

            # Lazy import to avoid circular dependency
            from mcp_server_llm_cli_runner.core.models import LLMRequest

            # Create LLM request
            request = LLMRequest(
                prompt=prompt,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens or 1000,
                system_prompt=None,
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
                            "provider": provider,
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

                # Send completion notification
                await self._send_notification(
                    "llm/completed",
                    {
                        "operation_id": operation_id,
                        "provider": provider,
                        "model": response.model,
                        "tokens_used": response.tokens_used,
                        "response_time_ms": response.response_time_ms,
                        "timestamp": time.time(),
                    },
                )

                result = {
                    "provider": response.provider
                    if hasattr(response, "provider")
                    else provider,
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
                                "provider": provider,
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
                        "provider": provider,
                        "error": str(provider_error),
                        "retry_attempts": retry_info,
                        "timestamp": time.time(),
                    },
                )
                raise

        except Exception as e:
            logger.error(f"Error in llm_complete: {e}")
            error_result = {"error": str(e)}
            retry_attempts = getattr(e, "retry_attempts", None)
            if retry_attempts is not None:
                error_result["attempts"] = retry_attempts
            return error_result

    def _list_providers(self) -> dict[str, Any]:
        """List available providers using ProviderManager."""
        try:
            status = self.manager.health_monitor.get_all_health_statuses()
            initialized = self.manager.registry.list_providers()
            configured = self.config_manager.get_enabled_providers()

            return {
                "providers": initialized,
                "count": len(initialized),
                "configured": configured,
                "health_status": status,
            }
        except Exception as e:
            logger.error(f"Error in list_providers: {e}")
            return {"error": str(e)}

    async def _provider_info(self, provider: str) -> dict[str, Any]:
        """Get provider information using ProviderManager."""
        try:
            health = self.manager.health_monitor.get_provider_health(provider)
            provider_instance = self.manager.get_provider(provider)

            if not provider_instance:
                return {"error": f"Provider '{provider}' not initialized"}

            # Try to get detailed health if available
            is_available = False
            if hasattr(provider_instance, "is_available"):
                is_available = await provider_instance.is_available()

            return {
                "provider": provider,
                "initialized": True,
                "available": is_available,
                "health": health.to_dict() if health else None,
                "stats": self.manager.get_provider_stats(provider),
                "metadata": getattr(provider_instance, "metadata", {}),
            }
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
