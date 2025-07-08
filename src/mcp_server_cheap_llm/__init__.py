"""MCP Server for Cheap LLM Providers.

This module provides a Model Context Protocol server that integrates with
cost-effective Large Language Model providers:
- Gemini CLI: Google Gemini models via command-line interface
- Codex: OpenAI Codex models for code generation
- LLaMA: Local LLaMA models via llama-cpp-python

Key components:
    CheapLLMServer: Main MCP server implementation
    ProviderManager: Manages multiple LLM providers
    ConfigManager: Handles provider configuration

Example usage:
    >>> from mcp_server_cheap_llm import main
    >>> main()  # Starts the MCP server
"""

from mcp_server_cheap_llm.server.handlers import CheapLLMServer
from mcp_server_cheap_llm.__main__ import main

__version__ = "0.1.0"
__all__ = ["CheapLLMServer", "main"]
