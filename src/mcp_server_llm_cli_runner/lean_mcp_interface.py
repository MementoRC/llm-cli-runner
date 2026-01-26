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

        # Initialize provider instances
        self._providers: dict[str, Any] = {}
        self._init_providers()

        # Tool registry: maps tool names to their implementations and metadata
        self.tool_registry = self._build_tool_registry()

        # Setup the 3 meta-tools
        self._setup_meta_tools()

    def _init_providers(self) -> None:
        """Initialize available provider instances."""
        enabled = self.config_manager.get_enabled_providers()

        if "gemini" in enabled:
            try:
                # Lazy import to avoid circular dependency
                from mcp_server_llm_cli_runner.providers.gemini import GeminiProvider

                self._providers["gemini"] = GeminiProvider()
                logger.info("Initialized Gemini provider")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini provider: {e}")

        # TODO: Add other providers (llama, openai) as they are implemented

    def _build_tool_registry(self) -> dict[str, dict[str, Any]]:
        """Build comprehensive tool registry with metadata for dynamic discovery.

        Each tool entry contains:
        - implementation: The actual function
        - schema: Full parameter schema
        - domain: Tool domain (llm, provider, config)
        - complexity: Tool complexity (focused, comprehensive)
        - description: Brief description
        - examples: Usage examples
        """
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
        """Execute LLM completion."""
        try:
            # Check if provider is available
            providers = self.config_manager.get_enabled_providers()
            if provider not in providers:
                return {
                    "error": f"Provider '{provider}' not available",
                    "available_providers": providers,
                }

            # Check if provider is initialized
            if provider not in self._providers:
                return {
                    "error": f"Provider '{provider}' not initialized",
                    "available_providers": list(self._providers.keys()),
                }

            # Lazy import to avoid circular dependency
            from mcp_server_llm_cli_runner.core.models import LLMRequest

            # Create LLM request
            # Note: stream is handled separately via stream_generate()
            request = LLMRequest(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens or 1000,
                system_prompt=None,
            )

            # Get provider instance and execute
            provider_instance = self._providers[provider]

            # Await the async generate method
            response = await provider_instance.generate(request)

            return {
                "provider": provider,
                "model": response.model,
                "completion": response.content,
                "success": response.success,
                "tokens_used": response.tokens_used,
                "cost": response.cost,
                "response_time_ms": response.response_time_ms,
                "metadata": response.metadata,
            }
        except Exception as e:
            logger.error(f"Error in llm_complete: {e}")
            return {"error": str(e)}

    def _list_providers(self) -> dict[str, Any]:
        """List available providers."""
        try:
            providers = self.config_manager.get_enabled_providers()
            return {"providers": providers, "count": len(providers)}
        except Exception as e:
            logger.error(f"Error in list_providers: {e}")
            return {"error": str(e)}

    def _provider_info(self, provider: str) -> dict[str, Any]:
        """Get provider information."""
        try:
            providers = self.config_manager.get_enabled_providers()
            if provider not in providers:
                return {
                    "error": f"Provider '{provider}' not found",
                    "available_providers": providers,
                }

            # TODO: Get actual provider configuration
            return {
                "provider": provider,
                "status": "available",
                "config": {"type": "cli" if provider in ["gemini", "llama"] else "api"},
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
            """Get available tools with minimal context consumption.

            Args:
                pattern: Filter by name pattern (substring match, empty string for all tools)

            Returns:
                Compact tool list with names and brief descriptions
            """
            tools = []

            for name, info in self.tool_registry.items():
                # Apply pattern filter if provided
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
                "domains": list(
                    {info.get("domain", "llm") for info in self.tool_registry.values()}
                ),
                "complexity_levels": list(
                    {
                        info.get("complexity", "focused")
                        for info in self.tool_registry.values()
                    }
                ),
            }

        @self.app.tool(
            description="Get full specification for specific llm-cli-runner tool including schema and examples. USE WHEN: need parameter details for LLM/provider tools before execution"
        )
        def get_tool_spec(tool_name: str) -> dict[str, Any]:
            """Get full specification for specific tool including schema and examples.

            Args:
                tool_name: Name of tool to get specification for

            Returns:
                Complete tool specification with schema, examples, and usage notes
            """
            if tool_name not in self.tool_registry:
                available_tools = list(self.tool_registry.keys())
                return {
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": available_tools,
                }

            tool_info = self.tool_registry[tool_name]
            return {
                "name": tool_name,
                "description": tool_info["description"],
                "domain": tool_info.get("domain", "llm"),
                "complexity": tool_info.get("complexity", "focused"),
                "schema": tool_info["schema"],
                "examples": tool_info["examples"],
            }

        @self.app.tool(
            description="Execute llm-cli-runner tool with parameters. Supports LLM completions, provider queries, configuration. USE WHEN: executing LLM queries, checking provider status"
        )
        async def execute_tool(
            tool_name: str, parameters: dict[str, Any]
        ) -> dict[str, Any]:
            """Execute tool with parameters using dynamic dispatch.

            Args:
                tool_name: Name of tool to execute
                parameters: Tool parameters as object

            Returns:
                Tool execution result with standard error handling
            """
            if tool_name not in self.tool_registry:
                available_tools = list(self.tool_registry.keys())
                return {
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": available_tools,
                }

            tool_info = self.tool_registry[tool_name]
            tool_func = tool_info["implementation"]

            try:
                # Execute tool with parameters (handle both sync and async)
                result = tool_func(**parameters)
                # If result is a coroutine, await it
                if asyncio.iscoroutine(result):
                    result = await result
                return {"tool": tool_name, "status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                return {"tool": tool_name, "status": "error", "error": str(e)}

    def get_app(self) -> FastMCP:
        """Get the FastMCP app instance."""
        return self.app


def create_lean_interface(config_manager: Any) -> FastMCP:
    """Create a lean MCP interface with minimal context consumption.

    Args:
        config_manager: Initialized configuration manager

    Returns:
        FastMCP app with 3 meta-tools exposing full functionality
    """
    lean_interface = LeanMCPInterface(config_manager)
    return lean_interface.get_app()
