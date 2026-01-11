"""Integration tests for provider routing and selection."""

import pytest

from src.mcp_server_llm_cli_runner.core.models import ProviderType
from src.mcp_server_llm_cli_runner.providers.manager import ProviderManager
from src.mcp_server_llm_cli_runner.providers.registry import ProviderRegistry


class TestRoutingIntegration:
    """Integration tests for provider routing and selection."""

    def test_provider_registry_instantiation(self):
        """Test that provider registry can be instantiated."""
        registry = ProviderRegistry()
        assert registry is not None
        assert hasattr(registry, "_providers")
        assert hasattr(registry, "_instances")

    def test_provider_manager_instantiation(self):
        """Test that provider manager can be instantiated."""
        manager = ProviderManager()
        assert manager is not None
        assert manager.registry is not None
        assert manager.initialized is False

    def test_provider_type_enum_exists(self):
        """Test that provider type enumeration exists and has expected values."""
        assert hasattr(ProviderType, "GEMINI")
        assert hasattr(ProviderType, "OPENAI")
        assert hasattr(ProviderType, "CODEX")
        assert hasattr(ProviderType, "LLAMA")

    def test_manager_has_expected_components(self):
        """Test that manager has all expected components."""
        manager = ProviderManager()

        # Test core components exist
        assert hasattr(manager, "registry")
        assert hasattr(manager, "router")
        assert hasattr(manager, "health_monitor")
        assert hasattr(manager, "cache_service")
        assert hasattr(manager, "usage_metrics")
        assert hasattr(manager, "quota_trackers")

        # Test initialization state
        assert hasattr(manager, "initialized")
        assert manager.initialized is False

    async def test_manager_initialization_method_exists(self):
        """Test that manager has initialize method."""
        manager = ProviderManager()
        assert hasattr(manager, "initialize")
        assert callable(manager.initialize)

    def test_registry_has_expected_methods(self):
        """Test that registry has expected methods."""
        registry = ProviderRegistry()

        # Test required methods exist
        assert hasattr(registry, "register_provider")
        assert callable(registry.register_provider)

    def test_basic_workflow_components_importable(self):
        """Test that all basic workflow components can be imported."""
        # Test imports work
        from src.mcp_server_llm_cli_runner.core.models import ProviderType
        from src.mcp_server_llm_cli_runner.providers.manager import ProviderManager
        from src.mcp_server_llm_cli_runner.providers.registry import ProviderRegistry

        # Test instantiation works
        registry = ProviderRegistry()
        manager = ProviderManager()

        assert registry is not None
        assert manager is not None
        assert ProviderType.GEMINI is not None
