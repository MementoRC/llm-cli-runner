"""Entry point for MCP Server LLM CLI Runner.

This module provides the command-line interface and server startup logic.
Follows atomic design principles with minimal complexity (100-150 lines).

Key functions:
    main: Primary entry point for CLI
    setup_logging: Configure structured logging
    create_server: Factory function for server creation

Example:
    >>> python -m mcp_server_llm_cli_runner
    >>> python -m mcp_server_llm_cli_runner --debug

"""

import asyncio
import sys
from typing import Any

import click  # type: ignore[import-not-found]
import structlog  # type: ignore[import-not-found]
from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

from mcp_server_llm_cli_runner.server.handlers import LLMCliRunnerServer
from mcp_server_llm_cli_runner.utils.config import ConfigManager
from mcp_server_llm_cli_runner.utils.logging import setup_logging


def create_server(config_path: str | None = None, debug: bool = False) -> Any:
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
    # Configure logging with debug flag
    setup_logging(debug=debug)
    logger = structlog.get_logger(__name__)

    try:
        config_manager = ConfigManager(config_path)
        llm_cli_runner_server = LLMCliRunnerServer(config_manager)

        logger.info(
            "Server created successfully",
            providers=config_manager.get_enabled_providers(),
            debug_mode=debug,
        )

        return llm_cli_runner_server.get_mcp_server()

    except Exception as e:
        logger.exception("Failed to create server", error=str(e), exc_info=debug)
        raise


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option("--debug", "-d", is_flag=True, help="Enable debug logging")
def main(config: str | None = None, debug: bool = False) -> None:
    """Start the MCP Server for LLM CLI Runner.

    This server provides cost-effective access to various LLM providers
    through the Model Context Protocol.

    Args:
        config: Optional configuration file path
        debug: Enable verbose debug logging

    Example:
        $ mcp-server-llm-cli-runner --debug
        $ mcp-server-llm-cli-runner --config /path/to/config.toml

    """
    try:
        server = create_server(config_path=config, debug=debug)

        async def run_server() -> None:
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
