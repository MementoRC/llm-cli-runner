"""
Main MCP Git Server Application.

This module provides the ServerApplication class that serves as the primary entry point
and orchestrator for the entire MCP Git server system. It integrates all decomposed
components into a cohesive, production-ready application.

The ServerApplication replaces the monolithic server.py approach with a clean,
component-based architecture that provides the same functionality with improved
structure, maintainability, and LLM compatibility.
"""

import asyncio
import logging
import signal
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from ..frameworks.mcp_server_framework import MCPServerFramework
from ..frameworks.server_configuration import ServerConfigurationManager
from ..frameworks.server_core import MCPGitServerCore
from ..frameworks.server_github import GitHubService
from ..frameworks.server_middleware import MiddlewareChainManager
from ..frameworks.server_security import SecurityFramework
from ..operations.server_notifications import NotificationOperations
from ..protocols.debugging_protocol import DebuggableComponent
from ..services.git_service import GitService
from ..services.github_service import GitHubServiceConfig
from ..services.server_metrics import MetricsService
from ..services.server_session import SessionManager

logger = logging.getLogger(__name__)


class ServerApplicationConfig:
    """Configuration for the server application."""

    def __init__(
        self,
        repository_path: Path | None = None,
        enable_metrics: bool = True,
        enable_security: bool = True,
        enable_notifications: bool = True,
        test_mode: bool = False,
        debug_mode: bool = False,
    ):
        """
        Initialize server application configuration.

        Args:
            repository_path: Optional path to the git repository to serve
            enable_metrics: Whether to enable metrics collection
            enable_security: Whether to enable security framework
            enable_notifications: Whether to enable notification operations
            test_mode: Whether to run in test mode
            debug_mode: Whether to enable debug mode
        """
        self.repository_path = repository_path
        self.enable_metrics = enable_metrics
        self.enable_security = enable_security
        self.enable_notifications = enable_notifications
        self.test_mode = test_mode
        self.debug_mode = debug_mode


class ServerApplication(DebuggableComponent):
    """
    Main MCP Git Server Application.

    This class orchestrates all components of the MCP Git server into a cohesive
    application. It handles initialization, startup, runtime management, and
    graceful shutdown of all subsystems.

    The application follows a component-based architecture where each major
    subsystem is represented by a dedicated component that can be independently
    configured, started, stopped, and debugged.

    Example usage:
        >>> config = ServerApplicationConfig(
        ...     repository_path=Path("/path/to/repo"),
        ...     enable_metrics=True
        ... )
        >>> app = ServerApplication(config)
        >>> await app.start()
        >>> # Server is now running
        >>> await app.stop()
    """

    def __init__(self, config: ServerApplicationConfig | None = None):
        """
        Initialize the server application.

        Args:
            config: Application configuration
        """
        self.config = config or ServerApplicationConfig()
        self._initialized = False
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._components: dict[str, Any] = {}

        # Core framework components
        self._framework: MCPServerFramework | None = None
        self._server_core: MCPGitServerCore | None = None
        self._configuration_manager: ServerConfigurationManager | None = None

        # Service components
        self._git_service: GitService | None = None
        self._github_service: GitHubService | None = None
        self._metrics_service: MetricsService | None = None
        self._session_manager: SessionManager | None = None

        # Operations components
        self._notification_operations: NotificationOperations | None = None

        # Infrastructure components
        self._middleware_manager: MiddlewareChainManager | None = None
        self._security_framework: SecurityFramework | None = None

        logger.info("ServerApplication initialized")

    async def initialize(self) -> None:
        """
        Initialize all application components.

        This method sets up all components in the correct order, ensuring
        that dependencies are satisfied and all systems are ready to start.
        """
        if self._initialized:
            logger.warning("ServerApplication already initialized")
            return

        logger.info("Initializing ServerApplication components...")

        try:
            # Phase 1: Initialize core framework
            await self._initialize_core_framework()

            # Phase 2: Initialize configuration management
            await self._initialize_configuration()

            # Phase 3: Initialize infrastructure components
            await self._initialize_infrastructure()

            # Phase 4: Initialize services
            await self._initialize_services()

            # Phase 5: Initialize operations
            await self._initialize_operations()

            # Phase 6: Register all components with framework
            await self._register_components()

            # Phase 7: Initialize the framework
            await self._framework.initialize()

            self._initialized = True
            logger.info("ServerApplication initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize ServerApplication: {e}")
            await self._cleanup_partial_initialization()
            raise

    async def _initialize_core_framework(self) -> None:
        """Initialize the core MCP framework."""
        logger.debug("Initializing core framework...")

        # Create the main framework
        self._framework = MCPServerFramework()

        # Create the server core
        self._server_core = MCPGitServerCore("mcp-git-server")

        # Initialize the MCP server within the core
        self._server_core.initialize_server(self.config.repository_path)

        logger.debug("Core framework initialized")

    async def _initialize_configuration(self) -> None:
        """Initialize configuration management."""
        logger.debug("Initializing configuration management...")

        self._configuration_manager = ServerConfigurationManager()

        # Initialize with default configuration
        await self._configuration_manager.initialize()

        # Get the validated configuration
        self._configuration_manager.get_current_config()

        logger.debug("Configuration management initialized")

    async def _initialize_infrastructure(self) -> None:
        """Initialize infrastructure components."""
        logger.debug("Initializing infrastructure components...")

        # Initialize security framework
        if self.config.enable_security:
            self._security_framework = SecurityFramework()
            logger.debug("Security framework initialized")

        # Initialize middleware management
        self._middleware_manager = MiddlewareChainManager()
        logger.debug("Middleware management initialized")

        logger.debug("Infrastructure components initialized")

    async def _initialize_services(self) -> None:
        """Initialize service components."""
        logger.debug("Initializing service components...")

        # Get configuration for services
        server_config = self._configuration_manager.get_current_config()

        # Initialize Git service
        self._git_service = GitService(
            repository_path=self.config.repository_path,
            config=server_config.git_config
        )

        # Initialize GitHub service
        github_config = GitHubServiceConfig(
            token=server_config.github_config.github_token,
            api_version=server_config.github_config.api_version
        )
        self._github_service = GitHubService(github_config)

        # Initialize metrics service
        if self.config.enable_metrics:
            self._metrics_service = MetricsService(
                enable_system_metrics=True,
                max_metric_history=1000
            )
            logger.debug("Metrics service initialized")

        # Initialize session management
        self._session_manager = SessionManager()
        logger.debug("Session manager initialized")

        logger.debug("Service components initialized")

    async def _initialize_operations(self) -> None:
        """Initialize operations components."""
        logger.debug("Initializing operations components...")

        # Initialize notification operations
        if self.config.enable_notifications:
            server_config = self._configuration_manager.get_current_config()
            self._notification_operations = NotificationOperations(server_config)
            logger.debug("Notification operations initialized")

        logger.debug("Operations components initialized")

    async def _register_components(self) -> None:
        """Register all components with the framework."""
        logger.debug("Registering components with framework...")

        # Register core components
        if self._server_core:
            self._framework.register_component(
                name="server_core",
                component=self._server_core,
                dependencies=[],
                priority=1
            )

        if self._configuration_manager:
            self._framework.register_component(
                name="configuration",
                component=self._configuration_manager,
                dependencies=[],
                priority=2
            )

        # Register infrastructure components
        if self._security_framework:
            self._framework.register_component(
                name="security",
                component=self._security_framework,
                dependencies=["configuration"],
                priority=3
            )

        if self._middleware_manager:
            self._framework.register_component(
                name="middleware",
                component=self._middleware_manager,
                dependencies=["configuration"],
                priority=4
            )

        # Register services
        if self._git_service:
            self._framework.register_component(
                name="git_service",
                component=self._git_service,
                dependencies=["configuration", "security"],
                priority=5
            )

        if self._github_service:
            self._framework.register_component(
                name="github_service",
                component=self._github_service,
                dependencies=["configuration", "security"],
                priority=6
            )

        if self._metrics_service:
            self._framework.register_component(
                name="metrics",
                component=self._metrics_service,
                dependencies=["configuration"],
                priority=7
            )

        if self._session_manager:
            self._framework.register_component(
                name="sessions",
                component=self._session_manager,
                dependencies=["configuration"],
                priority=8
            )

        # Register operations
        if self._notification_operations:
            self._framework.register_component(
                name="notifications",
                component=self._notification_operations,
                dependencies=["configuration"],
                priority=9
            )

        logger.debug("Component registration complete")

    async def start(self) -> None:
        """
        Start the server application.

        This method starts all components in the correct order and begins
        serving MCP requests. The method will block until the server is
        shut down.
        """
        if not self._initialized:
            await self.initialize()

        if self._running:
            logger.warning("ServerApplication already running")
            return

        logger.info("Starting ServerApplication...")

        try:
            # Start the framework (which starts all components)
            await self._framework.start()

            # Start the MCP server core
            if self._server_core:
                await self._server_core.start_server(test_mode=self.config.test_mode)

            self._running = True
            logger.info("ServerApplication started successfully")

            # Install signal handlers for graceful shutdown
            self._install_signal_handlers()

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except Exception as e:
            logger.error(f"Failed to start ServerApplication: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """
        Stop the server application gracefully.

        This method stops all components in reverse order and cleans up
        all resources.
        """
        if not self._running:
            logger.warning("ServerApplication not running")
            return

        logger.info("Stopping ServerApplication...")

        try:
            # Signal shutdown
            self._shutdown_event.set()

            # Stop the framework (which stops all components)
            if self._framework:
                await self._framework.stop()

            self._running = False
            logger.info("ServerApplication stopped successfully")

        except Exception as e:
            logger.error(f"Error during ServerApplication shutdown: {e}")
            raise

    async def restart(self) -> None:
        """
        Restart the server application.

        This method performs a graceful stop followed by a start.
        """
        logger.info("Restarting ServerApplication...")
        await self.stop()
        await self.start()

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        def signal_handler(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())

        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

    async def _cleanup_partial_initialization(self) -> None:
        """Clean up after partial initialization failure."""
        logger.debug("Cleaning up partial initialization...")

        # Stop any components that were started
        if self._framework:
            try:
                await self._framework.stop()
            except Exception as e:
                logger.error(f"Error stopping framework during cleanup: {e}")

        # Reset initialization state
        self._initialized = False

    @asynccontextmanager
    async def _component_context(self, name: str, component: Any) -> AsyncIterator[Any]:
        """Context manager for component lifecycle."""
        logger.debug(f"Starting component: {name}")
        try:
            if hasattr(component, "start"):
                await component.start()
            yield component
        except Exception as e:
            logger.error(f"Error in component {name}: {e}")
            raise
        finally:
            logger.debug(f"Stopping component: {name}")
            if hasattr(component, "stop"):
                try:
                    await component.stop()
                except Exception as e:
                    logger.error(f"Error stopping component {name}: {e}")

    # DebuggableComponent Protocol Implementation

    def get_component_state(self) -> dict[str, Any]:
        """Get current state of the server application."""
        component_states = {}

        # Collect state from all registered components
        if self._framework:
            for name, registration in self._framework._components.items():
                component = registration.component
                if hasattr(component, "get_component_state"):
                    try:
                        component_states[name] = component.get_component_state()
                    except Exception as e:
                        component_states[name] = {"error": str(e)}
                else:
                    component_states[name] = {"status": "no_state_available"}

        return {
            "application_id": "mcp_git_server_application",
            "status": "running" if self._running else "stopped",
            "initialized": self._initialized,
            "config": {
                "repository_path": str(self.config.repository_path) if self.config.repository_path else None,
                "enable_metrics": self.config.enable_metrics,
                "enable_security": self.config.enable_security,
                "enable_notifications": self.config.enable_notifications,
                "test_mode": self.config.test_mode,
                "debug_mode": self.config.debug_mode,
            },
            "components": component_states,
            "component_count": len(component_states),
        }

    def validate_component(self) -> dict[str, Any]:
        """Validate server application configuration and state."""
        issues = []
        recommendations = []

        # Check initialization state
        if not self._initialized:
            issues.append("Application not initialized")

        # Check component availability
        if not self._framework:
            issues.append("Core framework not available")

        if not self._server_core:
            issues.append("Server core not available")

        # Check optional components
        if self.config.enable_metrics and not self._metrics_service:
            issues.append("Metrics enabled but service not available")

        if self.config.enable_security and not self._security_framework:
            issues.append("Security enabled but framework not available")

        if self.config.enable_notifications and not self._notification_operations:
            issues.append("Notifications enabled but operations not available")

        # Generate recommendations
        if not self.config.enable_metrics:
            recommendations.append("Consider enabling metrics for production monitoring")

        if not self.config.enable_security:
            recommendations.append("Consider enabling security framework for production use")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def get_debug_info(self, detailed: bool = False) -> dict[str, Any]:
        """Get debug information about the server application."""
        debug_info = {
            "application_type": "MCPGitServerApplication",
            "state": self.get_component_state(),
            "validation": self.validate_component(),
        }

        if detailed:
            # Add detailed component debug information
            detailed_components = {}
            if self._framework:
                for name, registration in self._framework._components.items():
                    component = registration.component
                    if hasattr(component, "get_debug_info"):
                        try:
                            detailed_components[name] = component.get_debug_info(detailed=True)
                        except Exception as e:
                            detailed_components[name] = {"error": str(e)}

            debug_info["detailed_components"] = detailed_components

            # Add framework debug information
            if self._framework:
                debug_info["framework_debug"] = self._framework.get_debug_info(detailed=True)

        return debug_info


async def main(
    repository_path: Path | None = None,
    test_mode: bool = False,
    debug_mode: bool = False,
) -> None:
    """
    Main entry point for the MCP Git Server Application.

    This function provides a simple interface for starting the server application
    with common configuration options.

    Args:
        repository_path: Optional path to the git repository to serve
        test_mode: Whether to run in test mode
        debug_mode: Whether to enable debug mode

    Example:
        >>> await main(repository_path=Path("/path/to/repo"))
    """
    # Configure logging
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create application configuration
    config = ServerApplicationConfig(
        repository_path=repository_path,
        test_mode=test_mode,
        debug_mode=debug_mode,
    )

    # Create and start the application
    app = ServerApplication(config)

    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        await app.stop()


# Backward compatibility alias
serve = main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP Git Server Application")
    parser.add_argument(
        "--repository",
        type=Path,
        help="Path to the git repository to serve"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Run the server application
    asyncio.run(main(
        repository_path=args.repository,
        test_mode=args.test_mode,
        debug_mode=args.debug,
    ))
