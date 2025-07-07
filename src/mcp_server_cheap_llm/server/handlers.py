"""MCP Server handlers for cheap LLM providers."""

from typing import Dict, List, Any
from mcp.server import Server
from mcp.types import Tool, TextContent, CallToolRequest, CallToolResult

from mcp_server_cheap_llm.utils.config import ConfigManager
from mcp_server_cheap_llm.utils.logging import get_logger


class CheapLLMServer:
    """Main MCP server implementation for cheap LLM providers."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the CheapLLMServer.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = get_logger(__name__)
        self._server = Server("cheap-llm")
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup MCP server handlers."""
        self._server.list_tools = self._list_tools
        self._server.call_tool = self._call_tool
        
        self.logger.info("Server handlers initialized")
    
    async def _list_tools(self) -> List[Tool]:
        """List available tools based on enabled providers.
        
        Returns:
            List of available tools
        """
        tools = []
        enabled_providers = self.config_manager.get_enabled_providers()
        
        for provider in enabled_providers:
            if provider == "gemini":
                tools.append(Tool(
                    name="gemini_generate",
                    description="Generate text using Gemini CLI",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to generate text from"
                            },
                            "model": {
                                "type": "string",
                                "description": "Gemini model to use (optional)",
                                "default": "gemini-1.5-flash"
                            }
                        },
                        "required": ["prompt"]
                    }
                ))
            
            elif provider == "codex":
                tools.append(Tool(
                    name="codex_generate",
                    description="Generate code using OpenAI Codex",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The code prompt to generate from"
                            },
                            "language": {
                                "type": "string",
                                "description": "Programming language (optional)",
                                "default": "python"
                            }
                        },
                        "required": ["prompt"]
                    }
                ))
            
            elif provider == "llama":
                tools.append(Tool(
                    name="llama_generate",
                    description="Generate text using local LLaMA model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to generate text from"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate (optional)",
                                "default": 256
                            }
                        },
                        "required": ["prompt"]
                    }
                ))
        
        self.logger.info(f"Listed {len(tools)} tools for providers: {enabled_providers}")
        return tools
    
    async def _call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Call a tool based on the request.
        
        Args:
            request: Tool call request
            
        Returns:
            Tool call result
        """
        tool_name = request.params.name
        arguments = request.params.arguments or {}
        
        self.logger.info(f"Calling tool: {tool_name}")
        
        try:
            if tool_name == "gemini_generate":
                response = await self._call_gemini(arguments)
            elif tool_name == "codex_generate":
                response = await self._call_codex(arguments)
            elif tool_name == "llama_generate":
                response = await self._call_llama(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            return CallToolResult(
                content=[TextContent(type="text", text=response)]
            )
        
        except Exception as e:
            self.logger.error(f"Tool call failed: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )
    
    async def _call_gemini(self, arguments: Dict[str, Any]) -> str:
        """Call Gemini CLI provider.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Generated response
        """
        prompt = arguments["prompt"]
        model = arguments.get("model", "gemini-1.5-flash")
        
        # TODO: Implement actual Gemini CLI call
        # For now, return a placeholder
        return f"Gemini ({model}) response to: {prompt}"
    
    async def _call_codex(self, arguments: Dict[str, Any]) -> str:
        """Call OpenAI Codex provider.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Generated response
        """
        prompt = arguments["prompt"]
        language = arguments.get("language", "python")
        
        # TODO: Implement actual Codex API call
        # For now, return a placeholder
        return f"Codex ({language}) response to: {prompt}"
    
    async def _call_llama(self, arguments: Dict[str, Any]) -> str:
        """Call local LLaMA provider.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Generated response
        """
        prompt = arguments["prompt"]
        max_tokens = arguments.get("max_tokens", 256)
        
        # TODO: Implement actual LLaMA call
        # For now, return a placeholder
        return f"LLaMA (max_tokens={max_tokens}) response to: {prompt}"
    
    def get_mcp_server(self) -> Server:
        """Get the MCP server instance.
        
        Returns:
            MCP Server instance
        """
        return self._server