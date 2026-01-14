"""Entry point for MCP Server LLM CLI Runner.

This module provides the command-line interface and server startup logic
using the FastMCP framework with lean meta-tool pattern.

Key functions:
    main: Primary entry point for CLI
    create_server: Factory function for lean MCP interface creation

Example:
    >>> python -m mcp_server_llm_cli_runner
    >>> python -m mcp_server_llm_cli_runner --debug

"""

import logging
import sys

import click  # type: ignore[import-not-found]

from mcp_server_llm_cli_runner.lean_mcp_interface import create_lean_interface
from mcp_server_llm_cli_runner.utils.config import ConfigManager
from mcp_server_llm_cli_runner.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def create_server(config_path: str | None = None, debug: bool = False):
    """Create and configure the lean MCP server instance.

    Args:
        config_path: Optional path to configuration file
        debug: Enable debug logging mode

    Returns:
        FastMCP app instance ready to run

    Raises:
        ConfigurationError: If configuration is invalid

    Example:
        >>> app = create_server(debug=True)
        >>> app.run()

    """
    # Configure logging with debug flag
    setup_logging(debug=debug)
    logger = logging.getLogger(__name__)

    try:
        # Initialize configuration manager
        config_manager = ConfigManager(config_path)

        logger.info(
            "Server created successfully",
            extra={
                "providers": config_manager.get_enabled_providers(),
                "debug_mode": debug,
            },
        )

        # Create lean MCP interface with 3 meta-tools
        app = create_lean_interface(config_manager)

        logger.info("Lean MCP interface initialized with meta-tool pattern")
        logger.info("Context consumption: ~500 tokens (vs 10-15K for traditional MCP)")

        return app

    except Exception as e:
        logger.exception(f"Failed to create server: {e}")
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
    through the Model Context Protocol using the lean meta-tool pattern.

    Args:
        config: Optional configuration file path
        debug: Enable verbose debug logging

    Example:
        $ mcp-server-llm-cli-runner --debug
        $ mcp-server-llm-cli-runner --config /path/to/config.toml

    """
    try:
        # Create the FastMCP app
        app = create_server(config_path=config, debug=debug)

        # Run the server (FastMCP handles stdio automatically)
        logger.info("Starting LLM CLI Runner MCP server...")
        app.run()

    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
