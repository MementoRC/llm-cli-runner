"""
Tests for MCP Server Framework module.

This module tests the core MCP server framework functionality including:
- Component registration and lifecycle management
- Plugin architecture and management
- Event system and subscription handling
- Dependency injection and resolution
- Framework state management and debugging
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, call
from typing import Any, Dict, List

from src.mcp_server_git.frameworks.mcp_server_framework import (
    MCPServerFramework,
    MCPPlugin,
    ComponentRegistration,
    EventSubscription,
)
from src.mcp_server_git.protocols.debugging_protocol import (
    ComponentState,
    ValidationResult,
    DebugInfo,
)


class MockComponent:
    """Mock component for testing."""
    
    def __init__(self, name: str, fail_on: str = None):
        self.name = name
        self.fail_on = fail_on
        self.initialize_called = False
        self.start_called = False
        self.stop_called = False
    
    async def initialize(self):
        if self.fail_on == "initialize":
            raise RuntimeError(f"Mock failure in {self.name}.initialize")
        self.initialize_called = True
    
    async def start(self):
        if self.fail_on == "start":
            raise RuntimeError(f"Mock failure in {self.name}.start")
        self.start_called = True
    
    async def stop(self):
        if self.fail_on == "stop":
            raise RuntimeError(f"Mock failure in {self.name}.stop")
        self.stop_called = True


class MockPlugin(MCPPlugin):
    """Mock plugin for testing."""
    
    def __init__(self, name: str, fail_on: str = None):
        super().__init__(name)
        self.fail_on = fail_on
        self.initialize_called = False
        self.start_called = False
        self.stop_called = False
    
    async def initialize(self):
        if self.fail_on == "initialize":
            raise RuntimeError(f"Mock failure in {self.name}.initialize")
        self.initialize_called = True
        self._initialized = True
    
    async def start(self):
        if self.fail_on == "start":
            raise RuntimeError(f"Mock failure in {self.name}.start")
        self.start_called = True
        self._running = True
    
    async def stop(self):
        if self.fail_on == "stop":
            raise RuntimeError(f"Mock failure in {self.name}.stop")
        self.stop_called = True
        self._running = False


class TestMCPServerFramework:
    """Test cases for MCPServerFramework class."""
    
    def test_framework_initialization(self):
        """Test framework initialization with configuration."""
        config = {"test_key": "test_value"}
        framework = MCPServerFramework(config)
        
        assert framework._config == config
        assert framework._components == {}
        assert framework._plugins == {}
        assert framework._started is False
        assert framework._initialization_order == []
    
    def test_framework_initialization_without_config(self):
        """Test framework initialization without configuration."""
        framework = MCPServerFramework()
        
        assert framework._config == {}
        assert framework._components == {}
        assert framework._plugins == {}
        assert framework._started is False
    
    def test_register_component_success(self):
        """Test successful component registration."""
        framework = MCPServerFramework()
        component = MockComponent("test_component")
        
        framework.register_component(
            "test_component",
            component,
            dependencies=["dep1", "dep2"],
            priority=50,
            auto_start=False
        )
        
        assert "test_component" in framework._components
        registration = framework._components["test_component"]
        assert registration.name == "test_component"
        assert registration.component == component
        assert registration.dependencies == ["dep1", "dep2"]
        assert registration.priority == 50
        assert registration.auto_start is False
    
    def test_register_component_duplicate_name(self):
        """Test component registration with duplicate name."""
        framework = MCPServerFramework()
        component1 = MockComponent("component1")
        component2 = MockComponent("component2")
        
        framework.register_component("test", component1)
        
        with pytest.raises(ValueError, match="Component 'test' already registered"):
            framework.register_component("test", component2)
    
    def test_register_component_defaults(self):
        """Test component registration with default values."""
        framework = MCPServerFramework()
        component = MockComponent("test_component")
        
        framework.register_component("test_component", component)
        
        registration = framework._components["test_component"]
        assert registration.dependencies == []
        assert registration.priority == 100
        assert registration.auto_start is True
    
    def test_register_plugin_success(self):
        """Test successful plugin registration."""
        framework = MCPServerFramework()
        plugin = MockPlugin("test_plugin")
        
        framework.register_plugin(plugin)
        
        assert "test_plugin" in framework._plugins
        assert framework._plugins["test_plugin"] == plugin
    
    def test_register_plugin_duplicate_name(self):
        """Test plugin registration with duplicate name."""
        framework = MCPServerFramework()
        plugin1 = MockPlugin("test_plugin")
        plugin2 = MockPlugin("test_plugin")
        
        framework.register_plugin(plugin1)
        
        with pytest.raises(ValueError, match="Plugin 'test_plugin' already registered"):
            framework.register_plugin(plugin2)
    
    def test_subscribe_to_event(self):
        """Test event subscription."""
        framework = MCPServerFramework()
        callback = MagicMock()
        
        framework.subscribe_to_event("test_event", callback, "test_component", priority=50)
        
        assert "test_event" in framework._event_subscriptions
        subscriptions = framework._event_subscriptions["test_event"]
        assert len(subscriptions) == 1
        
        subscription = subscriptions[0]
        assert subscription.event_type == "test_event"
        assert subscription.callback == callback
        assert subscription.component_name == "test_component"
        assert subscription.priority == 50
    
    def test_subscribe_to_event_priority_ordering(self):
        """Test event subscription priority ordering."""
        framework = MCPServerFramework()
        callback1 = MagicMock()
        callback2 = MagicMock()
        callback3 = MagicMock()
        
        framework.subscribe_to_event("test_event", callback2, "component2", priority=200)
        framework.subscribe_to_event("test_event", callback1, "component1", priority=100)
        framework.subscribe_to_event("test_event", callback3, "component3", priority=150)
        
        subscriptions = framework._event_subscriptions["test_event"]
        assert len(subscriptions) == 3
        assert subscriptions[0].priority == 100
        assert subscriptions[1].priority == 150
        assert subscriptions[2].priority == 200
    
    @pytest.mark.asyncio
    async def test_emit_event_sync_callbacks(self):
        """Test event emission with synchronous callbacks."""
        framework = MCPServerFramework()
        callback1 = MagicMock()
        callback2 = MagicMock()
        
        framework.subscribe_to_event("test_event", callback1, "component1")
        framework.subscribe_to_event("test_event", callback2, "component2")
        
        await framework.emit_event("test_event", "test_data")
        
        callback1.assert_called_once_with("test_data")
        callback2.assert_called_once_with("test_data")
    
    @pytest.mark.asyncio
    async def test_emit_event_async_callbacks(self):
        """Test event emission with asynchronous callbacks."""
        framework = MCPServerFramework()
        callback1 = AsyncMock()
        callback2 = AsyncMock()
        
        framework.subscribe_to_event("test_event", callback1, "component1")
        framework.subscribe_to_event("test_event", callback2, "component2")
        
        await framework.emit_event("test_event", "test_data")
        
        callback1.assert_called_once_with("test_data")
        callback2.assert_called_once_with("test_data")
    
    @pytest.mark.asyncio
    async def test_emit_event_no_subscribers(self):
        """Test event emission with no subscribers."""
        framework = MCPServerFramework()
        
        # Should not raise an exception
        await framework.emit_event("nonexistent_event", "test_data")
    
    @pytest.mark.asyncio
    async def test_emit_event_callback_error(self):
        """Test event emission with callback error."""
        framework = MCPServerFramework()
        
        def failing_callback(data):
            raise RuntimeError("Callback error")
        
        working_callback = MagicMock()
        
        framework.subscribe_to_event("test_event", failing_callback, "failing_component")
        framework.subscribe_to_event("test_event", working_callback, "working_component")
        
        # Should not raise exception, should continue with other callbacks
        await framework.emit_event("test_event", "test_data")
        
        working_callback.assert_called_once_with("test_data")
    
    def test_get_component_exists(self):
        """Test getting an existing component."""
        framework = MCPServerFramework()
        component = MockComponent("test_component")
        
        framework.register_component("test_component", component)
        
        retrieved = framework.get_component("test_component")
        assert retrieved == component
    
    def test_get_component_not_exists(self):
        """Test getting a non-existent component."""
        framework = MCPServerFramework()
        
        retrieved = framework.get_component("nonexistent")
        assert retrieved is None
    
    def test_resolve_initialization_order_no_dependencies(self):
        """Test dependency resolution with no dependencies."""
        framework = MCPServerFramework()
        
        framework.register_component("component1", MockComponent("component1"), priority=200)
        framework.register_component("component2", MockComponent("component2"), priority=100)
        framework.register_component("component3", MockComponent("component3"), priority=150)
        
        order = framework._resolve_initialization_order()
        
        # Should be sorted by priority
        assert order == ["component2", "component3", "component1"]
    
    def test_resolve_initialization_order_with_dependencies(self):
        """Test dependency resolution with dependencies."""
        framework = MCPServerFramework()
        
        framework.register_component("component1", MockComponent("component1"), dependencies=["component2"])
        framework.register_component("component2", MockComponent("component2"), dependencies=["component3"])
        framework.register_component("component3", MockComponent("component3"))
        
        order = framework._resolve_initialization_order()
        
        # Dependencies should come first
        assert order.index("component3") < order.index("component2")
        assert order.index("component2") < order.index("component1")
    
    def test_resolve_initialization_order_circular_dependency(self):
        """Test dependency resolution with circular dependencies."""
        framework = MCPServerFramework()
        
        framework.register_component("component1", MockComponent("component1"), dependencies=["component2"])
        framework.register_component("component2", MockComponent("component2"), dependencies=["component1"])
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            framework._resolve_initialization_order()
    
    def test_resolve_initialization_order_missing_dependency(self):
        """Test dependency resolution with missing dependencies."""
        framework = MCPServerFramework()
        
        framework.register_component("component1", MockComponent("component1"), dependencies=["nonexistent"])
        
        with pytest.raises(ValueError, match="Dependency 'nonexistent' not found"):
            framework._resolve_initialization_order()
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful framework initialization."""
        framework = MCPServerFramework()
        
        component1 = MockComponent("component1")
        component2 = MockComponent("component2")
        plugin1 = MockPlugin("plugin1")
        
        framework.register_component("component1", component1)
        framework.register_component("component2", component2)
        framework.register_plugin(plugin1)
        
        await framework.initialize()
        
        assert component1.initialize_called
        assert component2.initialize_called
        assert plugin1.initialize_called
        assert plugin1.initialized
        assert len(framework._initialization_order) == 2
        assert len(framework._component_states) == 2
    
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        framework = MCPServerFramework()
        framework._initialized = True  # Simulate already initialized
        
        with pytest.raises(RuntimeError, match="Framework already initialized"):
            await framework.initialize()
    
    @pytest.mark.asyncio
    async def test_initialize_plugin_failure(self):
        """Test initialization with plugin failure."""
        framework = MCPServerFramework()
        
        plugin1 = MockPlugin("plugin1", fail_on="initialize")
        framework.register_plugin(plugin1)
        
        with pytest.raises(RuntimeError, match="Mock failure in plugin1.initialize"):
            await framework.initialize()
    
    @pytest.mark.asyncio
    async def test_initialize_component_failure(self):
        """Test initialization with component failure."""
        framework = MCPServerFramework()
        
        component1 = MockComponent("component1", fail_on="initialize")
        framework.register_component("component1", component1)
        
        with pytest.raises(RuntimeError, match="Mock failure in component1.initialize"):
            await framework.initialize()
    
    @pytest.mark.asyncio
    async def test_start_success(self):
        """Test successful framework start."""
        framework = MCPServerFramework()
        
        component1 = MockComponent("component1")
        component2 = MockComponent("component2")
        plugin1 = MockPlugin("plugin1")
        
        framework.register_component("component1", component1)
        framework.register_component("component2", component2, auto_start=False)
        framework.register_plugin(plugin1)
        
        await framework.initialize()
        await framework.start()
        
        assert component1.start_called
        assert not component2.start_called  # auto_start=False
        assert plugin1.start_called
        assert plugin1.running
        assert framework._started
        assert framework._framework_started_at is not None
    
    @pytest.mark.asyncio
    async def test_start_not_initialized(self):
        """Test start without initialization."""
        framework = MCPServerFramework()
        
        with pytest.raises(RuntimeError, match="Framework not initialized"):
            await framework.start()
    
    @pytest.mark.asyncio
    async def test_start_already_started(self):
        """Test start when already started."""
        framework = MCPServerFramework()
        component1 = MockComponent("component1")
        framework.register_component("component1", component1)
        
        await framework.initialize()
        await framework.start()
        
        with pytest.raises(RuntimeError, match="Framework already started"):
            await framework.start()
    
    @pytest.mark.asyncio
    async def test_start_plugin_failure(self):
        """Test start with plugin failure."""
        framework = MCPServerFramework()
        
        plugin1 = MockPlugin("plugin1", fail_on="start")
        framework.register_plugin(plugin1)
        
        await framework.initialize()
        
        with pytest.raises(RuntimeError, match="Mock failure in plugin1.start"):
            await framework.start()
    
    @pytest.mark.asyncio
    async def test_start_component_failure(self):
        """Test start with component failure."""
        framework = MCPServerFramework()
        
        component1 = MockComponent("component1", fail_on="start")
        framework.register_component("component1", component1)
        
        await framework.initialize()
        
        with pytest.raises(RuntimeError, match="Mock failure in component1.start"):
            await framework.start()
    
    @pytest.mark.asyncio
    async def test_stop_success(self):
        """Test successful framework stop."""
        framework = MCPServerFramework()
        
        component1 = MockComponent("component1")
        component2 = MockComponent("component2")
        plugin1 = MockPlugin("plugin1")
        
        framework.register_component("component1", component1)
        framework.register_component("component2", component2)
        framework.register_plugin(plugin1)
        
        await framework.initialize()
        await framework.start()
        await framework.stop()
        
        assert component1.stop_called
        assert component2.stop_called
        assert plugin1.stop_called
        assert not plugin1.running
        assert not framework._started
    
    @pytest.mark.asyncio
    async def test_stop_not_started(self):
        """Test stop when not started."""
        framework = MCPServerFramework()
        
        # Should not raise exception
        await framework.stop()
    
    @pytest.mark.asyncio
    async def test_stop_with_shutdown_handlers(self):
        """Test stop with shutdown handlers."""
        framework = MCPServerFramework()
        
        sync_handler = MagicMock()
        async_handler = AsyncMock()
        
        framework.add_shutdown_handler(sync_handler)
        framework.add_shutdown_handler(async_handler)
        
        component1 = MockComponent("component1")
        framework.register_component("component1", component1)
        
        await framework.initialize()
        await framework.start()
        await framework.stop()
        
        sync_handler.assert_called_once()
        async_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_with_error_in_shutdown_handler(self):
        """Test stop with error in shutdown handler."""
        framework = MCPServerFramework()
        
        def failing_handler():
            raise RuntimeError("Handler error")
        
        working_handler = MagicMock()
        
        framework.add_shutdown_handler(failing_handler)
        framework.add_shutdown_handler(working_handler)
        
        component1 = MockComponent("component1")
        framework.register_component("component1", component1)
        
        await framework.initialize()
        await framework.start()
        
        # Should not raise exception, should continue with other handlers
        await framework.stop()
        
        working_handler.assert_called_once()
    
    def test_add_shutdown_handler(self):
        """Test adding shutdown handlers."""
        framework = MCPServerFramework()
        
        handler1 = MagicMock()
        handler2 = MagicMock()
        
        framework.add_shutdown_handler(handler1)
        framework.add_shutdown_handler(handler2)
        
        assert len(framework._shutdown_handlers) == 2
        assert handler1 in framework._shutdown_handlers
        assert handler2 in framework._shutdown_handlers


class TestMCPServerFrameworkDebuggable:
    """Test cases for MCPServerFramework DebuggableComponent implementation."""
    
    @pytest.mark.asyncio
    async def test_get_component_state(self):
        """Test getting component state."""
        framework = MCPServerFramework()
        component1 = MockComponent("component1")
        plugin1 = MockPlugin("plugin1")
        
        framework.register_component("component1", component1)
        framework.register_plugin(plugin1)
        
        await framework.initialize()
        await framework.start()
        
        state = framework.get_component_state()
        
        assert state.component_id == "mcp_server_framework"
        assert state.component_type == "MCPServerFramework"
        assert state.state_data["started"] is True
        assert state.state_data["components_count"] == 1
        assert state.state_data["plugins_count"] == 1
        assert state.state_data["framework_started_at"] is not None
        assert "component1" in state.state_data["component_states"]
    
    def test_validate_component_success(self):
        """Test component validation success."""
        framework = MCPServerFramework()
        
        framework.register_component("component1", MockComponent("component1"), dependencies=["component2"])
        framework.register_component("component2", MockComponent("component2"))
        
        result = framework.validate_component()
        
        assert result.is_valid is True
        assert result.issues == []
        assert result.component_id == "mcp_server_framework"
    
    def test_validate_component_missing_dependency(self):
        """Test component validation with missing dependency."""
        framework = MCPServerFramework()
        
        framework.register_component("component1", MockComponent("component1"), dependencies=["nonexistent"])
        
        result = framework.validate_component()
        
        assert result.is_valid is False
        assert len(result.issues) == 1
        assert "Dependency 'nonexistent' not found" in result.issues[0]
    
    def test_validate_component_circular_dependency(self):
        """Test component validation with circular dependency."""
        framework = MCPServerFramework()
        
        framework.register_component("component1", MockComponent("component1"), dependencies=["component2"])
        framework.register_component("component2", MockComponent("component2"), dependencies=["component1"])
        
        result = framework.validate_component()
        
        assert result.is_valid is False
        assert len(result.issues) == 1
        assert "Dependency resolution error" in result.issues[0]
    
    def test_get_debug_info_basic(self):
        """Test getting basic debug info."""
        framework = MCPServerFramework()
        
        framework.register_component("component1", MockComponent("component1"))
        framework.register_plugin(MockPlugin("plugin1"))
        framework.subscribe_to_event("test_event", MagicMock(), "component1")
        
        debug_info = framework.get_debug_info("basic")
        
        assert debug_info.component_id == "mcp_server_framework"
        assert debug_info.debug_data["framework_started"] is False
        assert debug_info.debug_data["components_registered"] == 1
        assert debug_info.debug_data["plugins_registered"] == 1
        assert debug_info.debug_data["event_subscriptions"] == 1
        assert isinstance(debug_info.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_get_debug_info_detailed(self):
        """Test getting detailed debug info."""
        framework = MCPServerFramework()
        
        framework.register_component("component1", MockComponent("component1"))
        framework.register_plugin(MockPlugin("plugin1"))
        framework.subscribe_to_event("test_event", MagicMock(), "component1")
        
        await framework.initialize()
        
        debug_info = framework.get_debug_info("detailed")
        
        assert debug_info.component_id == "mcp_server_framework"
        assert "component_names" in debug_info.debug_data
        assert "plugin_names" in debug_info.debug_data
        assert "initialization_order" in debug_info.debug_data
        assert "event_types" in debug_info.debug_data
        assert debug_info.debug_data["component_names"] == ["component1"]
        assert debug_info.debug_data["plugin_names"] == ["plugin1"]
        assert debug_info.debug_data["event_types"] == ["test_event"]


class TestMCPPlugin:
    """Test cases for MCPPlugin abstract base class."""
    
    def test_plugin_initialization(self):
        """Test plugin initialization."""
        plugin = MockPlugin("test_plugin")
        
        assert plugin.name == "test_plugin"
        assert plugin.initialized is False
        assert plugin.running is False
    
    @pytest.mark.asyncio
    async def test_plugin_lifecycle(self):
        """Test plugin lifecycle management."""
        plugin = MockPlugin("test_plugin")
        
        assert not plugin.initialized
        assert not plugin.running
        
        await plugin.initialize()
        assert plugin.initialized
        assert not plugin.running
        
        await plugin.start()
        assert plugin.initialized
        assert plugin.running
        
        await plugin.stop()
        assert plugin.initialized
        assert not plugin.running


class TestComponentRegistration:
    """Test cases for ComponentRegistration dataclass."""
    
    def test_component_registration_creation(self):
        """Test creating component registration."""
        component = MockComponent("test_component")
        
        registration = ComponentRegistration(
            name="test_component",
            component=component,
            dependencies=["dep1", "dep2"],
            priority=50,
            auto_start=False
        )
        
        assert registration.name == "test_component"
        assert registration.component == component
        assert registration.dependencies == ["dep1", "dep2"]
        assert registration.priority == 50
        assert registration.auto_start is False
        assert isinstance(registration.registered_at, datetime)
    
    def test_component_registration_defaults(self):
        """Test component registration with default values."""
        component = MockComponent("test_component")
        
        registration = ComponentRegistration(
            name="test_component",
            component=component
        )
        
        assert registration.dependencies == []
        assert registration.priority == 100
        assert registration.auto_start is True


class TestEventSubscription:
    """Test cases for EventSubscription dataclass."""
    
    def test_event_subscription_creation(self):
        """Test creating event subscription."""
        callback = MagicMock()
        
        subscription = EventSubscription(
            event_type="test_event",
            callback=callback,
            component_name="test_component",
            priority=50
        )
        
        assert subscription.event_type == "test_event"
        assert subscription.callback == callback
        assert subscription.component_name == "test_component"
        assert subscription.priority == 50
    
    def test_event_subscription_default_priority(self):
        """Test event subscription with default priority."""
        callback = MagicMock()
        
        subscription = EventSubscription(
            event_type="test_event",
            callback=callback,
            component_name="test_component"
        )
        
        assert subscription.priority == 100