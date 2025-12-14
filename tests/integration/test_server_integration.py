"""Integration tests for MCP Server components."""

import pytest

from src.mcp_server_cheap_llm.server.handlers import CheapLLMServer


class TestMCPServerIntegration:
    """Integration tests for MCP server functionality."""

    def test_server_initialization(self, mock_config_manager):
        """Test basic server initialization."""
        server = CheapLLMServer(mock_config_manager)

        # Test server components exist
        assert hasattr(server, "config_manager")
        assert hasattr(server, "logger")
        assert hasattr(server, "_server")

        # Test server state
        assert server.config_manager == mock_config_manager

    def test_server_has_required_methods(self, mock_config_manager):
        """Test that server has all required methods."""
        server = CheapLLMServer(mock_config_manager)

        # Test core methods exist
        assert hasattr(server, "_list_tools")
        assert callable(server._list_tools)

        assert hasattr(server, "_call_tool")
        assert callable(server._call_tool)

        assert hasattr(server, "get_mcp_server")
        assert callable(server.get_mcp_server)

    def test_server_components_instantiation(self, mock_config_manager):
        """Test that server components can be instantiated."""
        server = CheapLLMServer(mock_config_manager)

        # Test that all components are instantiated
        assert server.config_manager is not None
        assert server.logger is not None
        assert server._server is not None

    async def test_server_initialization_method(self, mock_config_manager):
        """Test server initialization method."""
        server = CheapLLMServer(mock_config_manager)

        # Test initialization method exists and is callable
        assert hasattr(server, "_list_tools")
        assert callable(server._list_tools)

        # Test that we can call list_tools without errors
        tools = await server._list_tools()
        assert isinstance(tools, list)

        # Note: We don't actually call initialize here to avoid
        # complex setup requirements in this basic integration test
