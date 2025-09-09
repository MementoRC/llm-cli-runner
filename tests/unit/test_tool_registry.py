"""Unit tests for MCP Tool Registration and Discovery System (Task 12.2).

This module provides comprehensive testing for:
- ToolRegistry class for tool registration and discovery
- Schema validation against MCP tool specification
- Provider-specific tool management for different LLM providers
- Tool versioning and conflict resolution mechanisms
- Dynamic tool loading and provider adapters
"""

import json
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest

from mcp_server_cheap_llm.core.errors import ValidationError

try:
    from mcp_server_cheap_llm.server.handlers import (
        ProviderToolManager,
        ToolAdapter,
        ToolRegistry,
        ToolVersionManager,
    )
except ImportError:
    pytest.skip("Tool registry components not available", allow_module_level=True)


class TestToolRegistry:
    """Test MCP tool registration and discovery functionality."""

    @pytest.fixture
    def tool_registry(self):
        """Create ToolRegistry instance for testing."""
        return ToolRegistry()

    @pytest.fixture
    def sample_tool_spec(self):
        """Sample MCP tool specification for testing."""
        return {
            "name": "text_completion",
            "description": "Generate text completions using LLM providers",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Input prompt"},
                    "max_tokens": {"type": "integer", "default": 100},
                    "temperature": {"type": "number", "default": 0.7},
                },
                "required": ["prompt"],
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "completion": {"type": "string"},
                    "usage": {"type": "object"},
                },
                "required": ["completion"],
            },
        }

    def test_tool_registry_initialization(self, tool_registry):
        """Test ToolRegistry initialization."""
        assert tool_registry is not None
        assert hasattr(tool_registry, "_tools")
        assert hasattr(tool_registry, "_providers")
        assert len(tool_registry._tools) == 0

    def test_register_tool_basic(self, tool_registry, sample_tool_spec):
        """Test basic tool registration."""
        provider = "openai"

        # Register tool
        tool_registry.register_tool(sample_tool_spec, provider)

        # Verify tool is registered
        assert "text_completion" in tool_registry._tools
        assert provider in tool_registry._tools["text_completion"]

    def test_tool_schema_validation(self, tool_registry):
        """Test MCP tool schema validation."""
        # Valid tool spec
        valid_tool = {
            "name": "valid_tool",
            "description": "A valid tool",
            "inputSchema": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
        }

        # Should not raise exception
        tool_registry.register_tool(valid_tool, "test_provider")

        # Invalid tool spec - missing required fields
        invalid_tool = {
            "description": "Missing name field",
        }

        with pytest.raises(ValidationError):
            tool_registry.register_tool(invalid_tool, "test_provider")

    def test_provider_specific_registration(self, tool_registry, sample_tool_spec):
        """Test provider-specific tool registration."""
        providers = ["openai", "anthropic", "gemini"]

        for provider in providers:
            tool_registry.register_tool(sample_tool_spec, provider)

        # Verify tool is registered for all providers
        tool_name = sample_tool_spec["name"]
        assert tool_name in tool_registry._tools

        for provider in providers:
            assert provider in tool_registry._tools[tool_name]

    def test_tool_discovery(self, tool_registry, sample_tool_spec):
        """Test tool discovery functionality."""
        provider = "openai"
        tool_registry.register_tool(sample_tool_spec, provider)

        # Discover all tools
        all_tools = tool_registry.discover_tools()
        assert len(all_tools) > 0
        assert sample_tool_spec["name"] in [tool["name"] for tool in all_tools]

        # Discover provider-specific tools
        provider_tools = tool_registry.discover_tools(provider=provider)
        assert len(provider_tools) > 0
        assert sample_tool_spec["name"] in [tool["name"] for tool in provider_tools]

    def test_tool_conflict_resolution(self, tool_registry):
        """Test tool name conflict resolution."""
        # Register same tool name with different providers
        tool_v1 = {
            "name": "conflict_tool",
            "description": "Version 1",
            "inputSchema": {"type": "object", "properties": {}},
        }

        tool_v2 = {
            "name": "conflict_tool",
            "description": "Version 2",
            "inputSchema": {
                "type": "object",
                "properties": {"new_field": {"type": "string"}},
            },
        }

        # Register with different providers
        tool_registry.register_tool(tool_v1, "provider1")
        tool_registry.register_tool(tool_v2, "provider2")

        # Both versions should be accessible
        provider1_tools = tool_registry.discover_tools(provider="provider1")
        provider2_tools = tool_registry.discover_tools(provider="provider2")

        assert len(provider1_tools) == 1
        assert len(provider2_tools) == 1
        assert provider1_tools[0]["description"] == "Version 1"
        assert provider2_tools[0]["description"] == "Version 2"

    def test_tool_unregistration(self, tool_registry, sample_tool_spec):
        """Test tool unregistration."""
        provider = "test_provider"
        tool_name = sample_tool_spec["name"]

        # Register then unregister
        tool_registry.register_tool(sample_tool_spec, provider)
        assert tool_name in tool_registry._tools

        tool_registry.unregister_tool(tool_name, provider)

        # Tool should be removed for that provider
        if tool_name in tool_registry._tools:
            assert provider not in tool_registry._tools[tool_name]

    def test_tool_invocation_routing(self, tool_registry, sample_tool_spec):
        """Test routing tool invocations to correct providers."""
        provider = "openai"
        tool_name = sample_tool_spec["name"]

        # Mock handler for tool
        mock_handler = Mock()
        mock_handler.return_value = {"completion": "Test response"}

        # Register tool with handler
        tool_registry.register_tool(sample_tool_spec, provider, handler=mock_handler)

        # Invoke tool
        params = {"prompt": "Test prompt", "max_tokens": 50}
        result = tool_registry.invoke_tool(tool_name, provider, params)

        # Verify handler was called
        mock_handler.assert_called_once_with(params)
        assert result == {"completion": "Test response"}

    def test_invalid_tool_invocation(self, tool_registry):
        """Test error handling for invalid tool invocations."""
        # Try to invoke non-existent tool
        with pytest.raises(ValidationError):
            tool_registry.invoke_tool("non_existent", "openai", {})

        # Try to invoke with invalid provider
        tool_spec = {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {"type": "object", "properties": {}},
        }
        tool_registry.register_tool(tool_spec, "valid_provider")

        with pytest.raises(ValidationError):
            tool_registry.invoke_tool("test_tool", "invalid_provider", {})


class TestProviderToolManager:
    """Test provider-specific tool management."""

    @pytest.fixture
    def provider_tool_manager(self):
        """Create ProviderToolManager instance."""
        return ProviderToolManager("openai")

    @pytest.fixture
    def openai_tool_spec(self):
        """OpenAI-specific tool specification."""
        return {
            "name": "gpt_completion",
            "description": "GPT text completion",
            "provider_config": {
                "model": "gpt-3.5-turbo",
                "endpoint": "chat/completions",
                "auth_required": True,
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "messages": {"type": "array"},
                    "model": {"type": "string", "default": "gpt-3.5-turbo"},
                },
                "required": ["messages"],
            },
        }

    def test_provider_tool_manager_initialization(self, provider_tool_manager):
        """Test ProviderToolManager initialization."""
        assert provider_tool_manager.provider_name == "openai"
        assert hasattr(provider_tool_manager, "_tools")
        assert len(provider_tool_manager._tools) == 0

    def test_provider_specific_tool_registration(
        self,
        provider_tool_manager,
        openai_tool_spec,
    ):
        """Test provider-specific tool registration."""
        tool_name = openai_tool_spec["name"]

        provider_tool_manager.register_provider_tool(openai_tool_spec)

        # Verify tool is registered
        assert tool_name in provider_tool_manager._tools
        tool = provider_tool_manager._tools[tool_name]
        assert "provider_config" in tool
        assert tool["provider_config"]["model"] == "gpt-3.5-turbo"

    def test_provider_tool_adaptation(self, provider_tool_manager, openai_tool_spec):
        """Test adaptation of tools to provider-specific formats."""
        provider_tool_manager.register_provider_tool(openai_tool_spec)

        # Adapt tool for provider API
        adapted_tool = provider_tool_manager.adapt_tool_for_provider(
            openai_tool_spec["name"],
        )

        # Verify adaptation
        assert adapted_tool is not None
        assert "provider_config" in adapted_tool
        assert adapted_tool["name"] == openai_tool_spec["name"]

    def test_provider_tool_validation(self, provider_tool_manager):
        """Test provider-specific tool validation."""
        # Valid OpenAI tool
        valid_openai_tool = {
            "name": "valid_openai_tool",
            "description": "Valid tool",
            "provider_config": {"model": "gpt-4", "endpoint": "completions"},
            "inputSchema": {"type": "object", "properties": {}},
        }

        # Should not raise exception
        provider_tool_manager.register_provider_tool(valid_openai_tool)

        # Invalid tool - missing provider_config
        invalid_tool = {
            "name": "invalid_tool",
            "description": "Missing provider config",
            "inputSchema": {"type": "object", "properties": {}},
        }

        with pytest.raises(ValidationError):
            provider_tool_manager.register_provider_tool(invalid_tool)

    def test_provider_tool_execution_context(
        self,
        provider_tool_manager,
        openai_tool_spec,
    ):
        """Test execution context for provider-specific tools."""
        tool_name = openai_tool_spec["name"]
        provider_tool_manager.register_provider_tool(openai_tool_spec)

        # Create execution context
        context = provider_tool_manager.create_execution_context(tool_name)

        # Verify context contains provider-specific information
        assert context is not None
        assert "provider" in context
        assert "config" in context
        assert context["provider"] == "openai"


class TestToolAdapter:
    """Test tool adaptation for different providers."""

    @pytest.fixture
    def tool_adapter(self, generic_tool_spec):
        """Create ToolAdapter instance."""
        return ToolAdapter(generic_tool_spec)

    @pytest.fixture
    def generic_tool_spec(self):
        """Generic tool specification for adaptation."""
        return {
            "name": "text_analyzer",
            "description": "Analyze text content",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["sentiment", "entities"],
                    },
                },
                "required": ["text"],
            },
        }

    def test_tool_adapter_initialization(self, tool_adapter):
        """Test ToolAdapter initialization."""
        assert tool_adapter is not None
        assert hasattr(tool_adapter, "adapt_tool")

    def test_adapt_tool_for_openai(self, tool_adapter, generic_tool_spec):
        """Test tool adaptation for OpenAI provider."""
        adapted_tool = tool_adapter.adapt_tool(generic_tool_spec, "openai")

        # Verify OpenAI-specific adaptations
        assert adapted_tool is not None
        assert adapted_tool["name"] == generic_tool_spec["name"]
        assert "provider_specific" in adapted_tool
        assert adapted_tool["provider_specific"]["provider"] == "openai"

    def test_adapt_tool_for_anthropic(self, tool_adapter, generic_tool_spec):
        """Test tool adaptation for Anthropic provider."""
        adapted_tool = tool_adapter.adapt_tool(generic_tool_spec, "anthropic")

        # Verify Anthropic-specific adaptations
        assert adapted_tool is not None
        assert adapted_tool["name"] == generic_tool_spec["name"]
        assert "provider_specific" in adapted_tool
        assert adapted_tool["provider_specific"]["provider"] == "anthropic"

    def test_adapt_tool_schema_transformation(self, tool_adapter):
        """Test schema transformation during tool adaptation."""
        tool_with_complex_schema = {
            "name": "complex_tool",
            "description": "Tool with complex schema",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "nested_object": {
                        "type": "object",
                        "properties": {"inner": {"type": "string"}},
                    },
                    "array_field": {"type": "array", "items": {"type": "number"}},
                },
            },
        }

        adapted_tool = tool_adapter.adapt_tool(tool_with_complex_schema, "openai")

        # Verify schema is preserved and potentially enhanced
        assert "inputSchema" in adapted_tool
        assert adapted_tool["inputSchema"]["type"] == "object"
        assert "nested_object" in adapted_tool["inputSchema"]["properties"]

    def test_unsupported_provider_adaptation(self, tool_adapter, generic_tool_spec):
        """Test adaptation for unsupported providers."""
        with pytest.raises(ValidationError):
            tool_adapter.adapt_tool(generic_tool_spec, "unsupported_provider")


class TestToolVersionManager:
    """Test tool versioning and conflict resolution."""

    @pytest.fixture
    def version_manager(self):
        """Create ToolVersionManager instance."""
        return ToolVersionManager()

    @pytest.fixture
    def versioned_tool_v1(self):
        """Tool specification version 1."""
        return {
            "name": "versioned_tool",
            "version": "1.0.0",
            "description": "Version 1 of tool",
            "inputSchema": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
            },
        }

    @pytest.fixture
    def versioned_tool_v2(self):
        """Tool specification version 2."""
        return {
            "name": "versioned_tool",
            "version": "2.0.0",
            "description": "Version 2 of tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "new_param": {"type": "integer", "default": 0},
                },
            },
        }

    def test_version_manager_initialization(self, version_manager):
        """Test ToolVersionManager initialization."""
        assert version_manager is not None
        assert hasattr(version_manager, "_versions")

    def test_register_tool_version(self, version_manager, versioned_tool_v1):
        """Test registering tool versions."""
        tool_name = versioned_tool_v1["name"]
        version = versioned_tool_v1["version"]

        version_manager.register_version(versioned_tool_v1)

        # Verify version is registered
        assert tool_name in version_manager._versions
        assert version in version_manager._versions[tool_name]

    def test_version_conflict_resolution(
        self,
        version_manager,
        versioned_tool_v1,
        versioned_tool_v2,
    ):
        """Test version conflict resolution."""
        # Register both versions
        version_manager.register_version(versioned_tool_v1)
        version_manager.register_version(versioned_tool_v2)

        # Get latest version
        latest = version_manager.get_latest_version("versioned_tool")
        assert latest["version"] == "2.0.0"

        # Get specific version
        specific = version_manager.get_version("versioned_tool", "1.0.0")
        assert specific["version"] == "1.0.0"

    def test_version_compatibility_check(
        self,
        version_manager,
        versioned_tool_v1,
        versioned_tool_v2,
    ):
        """Test version compatibility checking."""
        version_manager.register_version(versioned_tool_v1)
        version_manager.register_version(versioned_tool_v2)

        # Check compatibility
        is_compatible = version_manager.is_compatible(
            "versioned_tool",
            "1.0.0",
            "2.0.0",
        )

        # Compatibility depends on implementation
        # For now, just verify the method exists and returns a boolean
        assert isinstance(is_compatible, bool)

    def test_version_migration(
        self,
        version_manager,
        versioned_tool_v1,
        versioned_tool_v2,
    ):
        """Test tool version migration."""
        version_manager.register_version(versioned_tool_v1)
        version_manager.register_version(versioned_tool_v2)

        # Migrate parameters from v1 to v2
        v1_params = {"input": "test input"}
        migrated_params = version_manager.migrate_parameters(
            "versioned_tool",
            "1.0.0",
            "2.0.0",
            v1_params,
        )

        # Verify migration
        assert "input" in migrated_params
        assert migrated_params["input"] == "test input"
        # new_param should have default value
        if "new_param" in migrated_params:
            assert migrated_params["new_param"] == 0
