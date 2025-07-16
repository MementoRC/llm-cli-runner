"""Unit tests for server handlers - TDD approach."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from mcp.types import (  # type: ignore[import-not-found]
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    TextContent,
    Tool,
)

from mcp_server_cheap_llm.server.handlers import CheapLLMServer


class TestCheapLLMServerTDD:
    """Test CheapLLMServer using TDD approach."""

    def test_server_import(self):
        """Test that we can import CheapLLMServer."""
        from mcp_server_cheap_llm.server.handlers import CheapLLMServer

        assert CheapLLMServer is not None

    def test_server_instantiation(self):
        """Test CheapLLMServer can be instantiated."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["gemini"]

        server = CheapLLMServer(mock_config_manager)
        assert server is not None
        assert server.config_manager == mock_config_manager

    def test_server_has_mcp_server(self):
        """Test CheapLLMServer has get_mcp_server method."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["gemini"]

        server = CheapLLMServer(mock_config_manager)
        assert hasattr(server, "get_mcp_server")

        mcp_server = server.get_mcp_server()
        assert mcp_server is not None

    @pytest.mark.asyncio
    async def test_list_tools_returns_list(self):
        """Test _list_tools returns a list of tools."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["gemini"]

        server = CheapLLMServer(mock_config_manager)
        tools = await server._list_tools()

        assert isinstance(tools, list)
        assert len(tools) == 1  # One tool for gemini
        assert isinstance(tools[0], Tool)
        assert tools[0].name == "gemini_generate"

    @pytest.mark.asyncio
    async def test_list_tools_multiple_providers(self):
        """Test _list_tools with multiple enabled providers."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = [
            "gemini",
            "codex",
            "llama",
        ]

        server = CheapLLMServer(mock_config_manager)
        tools = await server._list_tools()

        assert isinstance(tools, list)
        assert len(tools) == 3  # Three tools for three providers

        tool_names = [tool.name for tool in tools]
        assert "gemini_generate" in tool_names
        assert "codex_generate" in tool_names
        assert "llama_generate" in tool_names

    @pytest.mark.asyncio
    async def test_call_tool_gemini(self):
        """Test _call_tool with gemini provider."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["gemini"]

        server = CheapLLMServer(mock_config_manager)

        # Mock the gemini call
        with patch.object(
            server, "_call_gemini", return_value="Test response"
        ) as mock_gemini:
            request = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="gemini_generate", arguments={"prompt": "Hello"}
                ),
            )

            result = await server._call_tool(request)

            # Verify the call and result
            mock_gemini.assert_called_once_with({"prompt": "Hello"})
            assert isinstance(result, CallToolResult)
            assert len(result.content) == 1
            assert isinstance(result.content[0], TextContent)
            assert result.content[0].text == "Test response"

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool(self):
        """Test _call_tool with unknown tool returns error."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["gemini"]

        server = CheapLLMServer(mock_config_manager)

        request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(
                name="unknown_tool", arguments={"prompt": "Hello"}
            ),
        )

        result = await server._call_tool(request)

        # Should return error result
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        assert len(result.content) == 1
        assert "Unknown tool" in result.content[0].text

    @pytest.mark.asyncio
    async def test_call_gemini_placeholder(self):
        """Test _call_gemini returns placeholder response."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["gemini"]

        server = CheapLLMServer(mock_config_manager)

        arguments = {"prompt": "Hello", "model": "gemini-pro"}
        response = await server._call_gemini(arguments)

        assert isinstance(response, str)
        assert "Gemini" in response
        assert "gemini-pro" in response
        assert "Hello" in response

    @pytest.mark.asyncio
    async def test_call_codex_placeholder(self):
        """Test _call_codex returns placeholder response."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["codex"]

        server = CheapLLMServer(mock_config_manager)

        arguments = {"prompt": "def hello():", "language": "python"}
        response = await server._call_codex(arguments)

        assert isinstance(response, str)
        assert "Codex" in response
        assert "python" in response
        assert "def hello():" in response

    @pytest.mark.asyncio
    async def test_call_llama_placeholder(self):
        """Test _call_llama returns placeholder response."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["llama"]

        server = CheapLLMServer(mock_config_manager)

        arguments = {"prompt": "Tell me a story", "max_tokens": 500}
        response = await server._call_llama(arguments)

        assert isinstance(response, str)
        assert "LLaMA" in response
        assert "max_tokens=500" in response
        assert "Tell me a story" in response

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test tool call error handling."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["gemini"]

        server = CheapLLMServer(mock_config_manager)

        # Mock gemini to raise an exception
        with patch.object(
            server, "_call_gemini", side_effect=RuntimeError("Test error")
        ) as mock_gemini:
            request = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="gemini_generate", arguments={"prompt": "Hello"}
                ),
            )

            result = await server._call_tool(request)

            # Should return error result
            assert isinstance(result, CallToolResult)
            assert result.isError is True
            assert len(result.content) == 1
            assert "Error: Test error" in result.content[0].text


class TestToolSchemas:
    """Test tool schema definitions."""

    @pytest.mark.asyncio
    async def test_gemini_tool_schema(self):
        """Test gemini tool has correct schema."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["gemini"]

        server = CheapLLMServer(mock_config_manager)
        tools = await server._list_tools()

        gemini_tool = tools[0]
        assert gemini_tool.name == "gemini_generate"
        assert gemini_tool.description == "Generate text using Gemini CLI"

        # Check input schema
        schema = gemini_tool.inputSchema
        assert schema["type"] == "object"
        assert "prompt" in schema["properties"]
        assert "model" in schema["properties"]
        assert schema["required"] == ["prompt"]
        assert schema["properties"]["model"]["default"] == "gemini-1.5-flash"

    @pytest.mark.asyncio
    async def test_codex_tool_schema(self):
        """Test codex tool has correct schema."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["codex"]

        server = CheapLLMServer(mock_config_manager)
        tools = await server._list_tools()

        codex_tool = tools[0]
        assert codex_tool.name == "codex_generate"
        assert codex_tool.description == "Generate code using OpenAI Codex"

        # Check input schema
        schema = codex_tool.inputSchema
        assert schema["type"] == "object"
        assert "prompt" in schema["properties"]
        assert "language" in schema["properties"]
        assert schema["required"] == ["prompt"]
        assert schema["properties"]["language"]["default"] == "python"

    @pytest.mark.asyncio
    async def test_llama_tool_schema(self):
        """Test llama tool has correct schema."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = ["llama"]

        server = CheapLLMServer(mock_config_manager)
        tools = await server._list_tools()

        llama_tool = tools[0]
        assert llama_tool.name == "llama_generate"
        assert llama_tool.description == "Generate text using local LLaMA model"

        # Check input schema
        schema = llama_tool.inputSchema
        assert schema["type"] == "object"
        assert "prompt" in schema["properties"]
        assert "max_tokens" in schema["properties"]
        assert schema["required"] == ["prompt"]
        assert schema["properties"]["max_tokens"]["default"] == 256
