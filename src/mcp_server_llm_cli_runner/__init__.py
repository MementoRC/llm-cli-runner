"""MCP Server for Multi-Provider LLM Access.

This module provides a Model Context Protocol server that integrates with
multiple Large Language Model providers via CLI tools:
- Gemini CLI: Google Gemini models via command-line interface
- OpenAI: OpenAI models via API
- LLaMA: Local LLaMA models via llama-cpp-python

Key components:
    lean_mcp_interface.create_lean_interface: FastMCP-based lean MCP interface
    ProviderManager: Manages multiple LLM providers
    ConfigManager: Handles provider configuration

Example usage:
    >>> from mcp_server_llm_cli_runner import main
    >>> main()  # Starts the MCP server
"""

from mcp_server_llm_cli_runner.__main__ import main

__version__ = "0.1.0"
__all__ = ["main"]
