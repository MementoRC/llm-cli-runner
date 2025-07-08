"""Entry point for MCP Server Cheap LLM.

This module provides the command-line interface and server startup logic.
Follows atomic design principles with minimal complexity (100-150 lines).

Key functions:
    main: Primary entry point for CLI
    setup_logging: Configure structured logging
    create_server: Factory function for server creation

Example:
    >>> python -m mcp_server_cheap_llm
    >>> python -m mcp_server_cheap_llm --debug
"""

import asyncio
import sys
from typing import Optional

import click
import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server

from mcp_server_cheap_llm.server.handlers import CheapLLMServer
from mcp_server_cheap_llm.utils.config import ConfigManager
from mcp_server_cheap_llm.utils.logging import setup_logging


def create_server(config_path: Optional[str] = None, debug: bool = False) -> Server:
    """Create and configure the MCP server instance.

    Args:
        config_path: Optional path to configuration file
        debug: Enable debug logging mode

    Returns:
        Configured MCP Server instance

    Raises:
        ConfigurationError: If configuration is invalid

    Example:
        >>> server = create_server(debug=True)
        >>> # Server ready for stdio_server()
    """
    setup_logging(debug=debug)
    logger = structlog.get_logger(__name__)

    try:
        config_manager = ConfigManager(config_path)
        cheap_llm_server = CheapLLMServer(config_manager)

        logger.info(
            "Server created successfully",
            providers=config_manager.get_enabled_providers(),
            debug_mode=debug,
        )

        return cheap_llm_server.get_mcp_server()

    except Exception as e:
        logger.error("Failed to create server", error=str(e), exc_info=debug)
        raise


@click.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to configuration file"
)
@click.option("--debug", "-d", is_flag=True, help="Enable debug logging")
def main(config: Optional[str] = None, debug: bool = False) -> None:
    """Start the MCP Server for Cheap LLM Providers.

    This server provides cost-effective access to various LLM providers
    through the Model Context Protocol.

    Args:
        config: Optional configuration file path
        debug: Enable verbose debug logging

    Example:
        $ mcp-server-cheap-llm --debug
        $ mcp-server-cheap-llm --config /path/to/config.toml
    """
    try:
        server = create_server(config_path=config, debug=debug)

        async def run_server():
            async with stdio_server() as (read_stream, write_stream):
                init_options = server.create_initialization_options()
                await server.run(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    initialization_options=init_options,
                )

        asyncio.run(run_server())

    except KeyboardInterrupt:
        click.echo("Server shutdown requested by user", err=True)
        sys.exit(0)

    except Exception as e:
        click.echo(f"Server startup failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
