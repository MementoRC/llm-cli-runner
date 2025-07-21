"""Unit tests for abstract provider interface and registry."""

import asyncio
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from mcp_server_cheap_llm.core.errors import ProviderError, ValidationError
from mcp_server_cheap_llm.core.models import LLMRequest, LLMResponse, ProviderStatus
from mcp_server_cheap_llm.providers.base import (
    CostEstimate,
    LLMProvider,
    ProviderCapabilities,
    ProviderMetadata,
    QuotaStatus,
    UsageStats,
)
from mcp_server_cheap_llm.providers.registry import (
    ProviderRegistry,
    get_provider,
    register_provider,
)


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._status = ProviderStatus.AVAILABLE
        self._usage = UsageStats()
        self._quota = QuotaStatus(
            remaining_requests=1000, remaining_tokens=100000, reset_time=1234567890
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True, batch=False, embeddings=True)

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name="mock",
            cost_per_token=Decimal("0.001"),
            rate_limits={"requests_per_minute": 60},
            model_variants=["mock-small", "mock-large"],
            max_tokens=4096,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        from mcp_server_cheap_llm.core.models import ProviderType

        return LLMResponse(
            content="Mock response",
            provider=ProviderType.GEMINI,  # Use valid enum value
            tokens_used=10,
            metadata={"model": "mock-small"},
        )

    def validate_config(self) -> bool:
        return "api_key" in self.config

    def get_usage(self) -> UsageStats:
        return self._usage

    def check_quota(self) -> QuotaStatus:
        return self._quota

    def estimate_cost(self, request: LLMRequest) -> CostEstimate:
        estimated_tokens = len(request.prompt.split()) * 2  # Simple estimation
        cost = Decimal(str(estimated_tokens)) * self.metadata.cost_per_token

        return CostEstimate(
            estimated_tokens=estimated_tokens,
            estimated_cost=cost,
            breakdown={"input": cost},
        )

    def get_status(self) -> ProviderStatus:
        return self._status

    def _get_required_config_keys(self) -> list[str]:
        return ["api_key"]


class TestProviderCapabilities:
    """Test provider capabilities dataclass."""

    def test_default_capabilities(self):
        """Test default capability values."""
        caps = ProviderCapabilities()

        assert caps.streaming is False
        assert caps.batch is False
        assert caps.embeddings is False
        assert caps.function_calling is False
        assert caps.vision is False
        assert caps.audio is False

    def test_custom_capabilities(self):
        """Test custom capability values."""
        caps = ProviderCapabilities(
            streaming=True, embeddings=True, function_calling=True
        )

        assert caps.streaming is True
        assert caps.batch is False  # Default
        assert caps.embeddings is True
        assert caps.function_calling is True
        assert caps.vision is False  # Default
        assert caps.audio is False  # Default


class TestProviderMetadata:
    """Test provider metadata dataclass."""

    def test_provider_metadata_creation(self):
        """Test provider metadata creation."""
        metadata = ProviderMetadata(
            name="test",
            cost_per_token=Decimal("0.002"),
            rate_limits={"rpm": 100},
            model_variants=["test-1", "test-2"],
            max_tokens=8192,
        )

        assert metadata.name == "test"
        assert metadata.cost_per_token == Decimal("0.002")
        assert metadata.rate_limits == {"rpm": 100}
        assert metadata.model_variants == ["test-1", "test-2"]
        assert metadata.max_tokens == 8192
        assert metadata.supported_languages == {"en"}  # Default

    def test_custom_supported_languages(self):
        """Test custom supported languages."""
        metadata = ProviderMetadata(
            name="test",
            cost_per_token=Decimal("0.001"),
            rate_limits={},
            model_variants=[],
            max_tokens=1000,
            supported_languages={"en", "es", "fr"},
        )

        assert metadata.supported_languages == {"en", "es", "fr"}


class TestUsageStats:
    """Test usage statistics dataclass."""

    def test_default_usage_stats(self):
        """Test default usage stats."""
        stats = UsageStats()

        assert stats.requests_made == 0
        assert stats.tokens_used == 0
        assert stats.cost_incurred == Decimal("0.00")
        assert stats.error_rate == 0.0
        assert stats.avg_response_time == 0.0

    def test_custom_usage_stats(self):
        """Test custom usage stats."""
        stats = UsageStats(
            requests_made=100,
            tokens_used=5000,
            cost_incurred=Decimal("5.50"),
            error_rate=2.5,
            avg_response_time=1.2,
        )

        assert stats.requests_made == 100
        assert stats.tokens_used == 5000
        assert stats.cost_incurred == Decimal("5.50")
        assert stats.error_rate == 2.5
        assert stats.avg_response_time == 1.2


class TestQuotaStatus:
    """Test quota status dataclass."""

    def test_quota_status_creation(self):
        """Test quota status creation."""
        quota = QuotaStatus(
            remaining_requests=500, remaining_tokens=25000, reset_time=1234567890
        )

        assert quota.remaining_requests == 500
        assert quota.remaining_tokens == 25000
        assert quota.reset_time == 1234567890
        assert quota.is_exhausted is False  # Default

    def test_exhausted_quota(self):
        """Test exhausted quota status."""
        quota = QuotaStatus(
            remaining_requests=0,
            remaining_tokens=0,
            reset_time=1234567890,
            is_exhausted=True,
        )

        assert quota.is_exhausted is True


class TestCostEstimate:
    """Test cost estimate dataclass."""

    def test_cost_estimate_creation(self):
        """Test cost estimate creation."""
        estimate = CostEstimate(
            estimated_tokens=100,
            estimated_cost=Decimal("0.10"),
            breakdown={"input": Decimal("0.05"), "output": Decimal("0.05")},
        )

        assert estimate.estimated_tokens == 100
        assert estimate.estimated_cost == Decimal("0.10")
        assert estimate.breakdown["input"] == Decimal("0.05")
        assert estimate.breakdown["output"] == Decimal("0.05")


class TestLLMProvider:
    """Test abstract LLM provider."""

    def test_provider_initialization_valid(self):
        """Test valid provider initialization."""
        config = {"api_key": "test_key"}
        provider = MockProvider(config)

        assert provider.config == config
        assert provider.validate_config() is True

    def test_provider_initialization_invalid_config_type(self):
        """Test provider initialization with invalid config type."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            MockProvider("invalid_config")

    def test_provider_initialization_missing_required_keys(self):
        """Test provider initialization with missing required keys."""
        config = {"wrong_key": "value"}

        with pytest.raises(
            ValidationError, match="Missing required configuration keys"
        ):
            MockProvider(config)

    def test_provider_capabilities(self):
        """Test provider capabilities."""
        config = {"api_key": "test_key"}
        provider = MockProvider(config)

        caps = provider.capabilities
        assert caps.streaming is True
        assert caps.batch is False
        assert caps.embeddings is True

    def test_provider_metadata(self):
        """Test provider metadata."""
        config = {"api_key": "test_key"}
        provider = MockProvider(config)

        metadata = provider.metadata
        assert metadata.name == "mock"
        assert metadata.cost_per_token == Decimal("0.001")
        assert metadata.model_variants == ["mock-small", "mock-large"]

    @pytest.mark.asyncio
    async def test_provider_generate(self):
        """Test provider generate method."""
        config = {"api_key": "test_key"}
        provider = MockProvider(config)

        request = LLMRequest(prompt="Test prompt")
        response = await provider.generate(request)

        from mcp_server_cheap_llm.core.models import ProviderType

        assert response.content == "Mock response"
        assert response.provider == ProviderType.GEMINI
        assert response.metadata["model"] == "mock-small"

    def test_provider_usage_stats(self):
        """Test provider usage statistics."""
        config = {"api_key": "test_key"}
        provider = MockProvider(config)

        stats = provider.get_usage()
        assert isinstance(stats, UsageStats)

    def test_provider_quota_check(self):
        """Test provider quota check."""
        config = {"api_key": "test_key"}
        provider = MockProvider(config)

        quota = provider.check_quota()
        assert quota.remaining_requests == 1000
        assert quota.remaining_tokens == 100000

    def test_provider_cost_estimation(self):
        """Test provider cost estimation."""
        config = {"api_key": "test_key"}
        provider = MockProvider(config)

        request = LLMRequest(prompt="Hello world")
        estimate = provider.estimate_cost(request)

        assert estimate.estimated_tokens == 4  # 2 words * 2
        assert estimate.estimated_cost == Decimal("0.004")  # 4 * 0.001

    def test_provider_status(self):
        """Test provider status."""
        config = {"api_key": "test_key"}
        provider = MockProvider(config)

        status = provider.get_status()
        assert status == ProviderStatus.AVAILABLE

    def test_provider_string_representation(self):
        """Test provider string representations."""
        config = {"api_key": "test_key"}
        provider = MockProvider(config)

        assert str(provider) == "MockProvider(name=mock)"
        assert "MockProvider" in repr(provider)
        assert "name=mock" in repr(provider)
        assert "status=ProviderStatus.AVAILABLE" in repr(provider)


class TestProviderRegistry:
    """Test provider registry."""

    def setup_method(self):
        """Setup for each test method."""
        ProviderRegistry.clear()

    def teardown_method(self):
        """Cleanup after each test method."""
        ProviderRegistry.clear()

    def test_register_provider(self):
        """Test provider registration."""
        ProviderRegistry.register("mock", MockProvider)

        providers = ProviderRegistry.list_providers()
        assert "mock" in providers
        assert providers["mock"] == MockProvider

    def test_register_invalid_provider(self):
        """Test registering invalid provider."""

        class InvalidProvider:
            pass

        with pytest.raises(ValidationError, match="must inherit from LLMProvider"):
            ProviderRegistry.register("invalid", InvalidProvider)

    def test_get_provider_class(self):
        """Test getting provider class."""
        ProviderRegistry.register("mock", MockProvider)

        provider_class = ProviderRegistry.get_provider_class("mock")
        assert provider_class == MockProvider

    def test_get_provider_class_not_found(self):
        """Test getting non-existent provider class."""
        with pytest.raises(
            ProviderError, match="Provider 'nonexistent' not registered"
        ):
            ProviderRegistry.get_provider_class("nonexistent")

    def test_create_provider(self):
        """Test creating provider instance."""
        ProviderRegistry.register("mock", MockProvider)

        config = {"api_key": "test_key"}
        provider = ProviderRegistry.create_provider("mock", config)

        assert isinstance(provider, MockProvider)
        assert provider.config == config

    def test_create_provider_invalid_config(self):
        """Test creating provider with invalid config."""
        ProviderRegistry.register("mock", MockProvider)

        config = {"wrong_key": "value"}

        with pytest.raises(ProviderError, match="Failed to create provider"):
            ProviderRegistry.create_provider("mock", config)

    def test_get_instance(self):
        """Test getting provider instance."""
        ProviderRegistry.register("mock", MockProvider)

        config = {"api_key": "test_key"}
        created_provider = ProviderRegistry.create_provider("mock", config)
        retrieved_provider = ProviderRegistry.get_instance("mock")

        assert created_provider is retrieved_provider

    def test_get_instance_not_found(self):
        """Test getting non-existent instance."""
        result = ProviderRegistry.get_instance("nonexistent")
        assert result is None

    def test_list_instances(self):
        """Test listing provider instances."""
        ProviderRegistry.register("mock", MockProvider)

        config = {"api_key": "test_key"}
        provider = ProviderRegistry.create_provider("mock", config)

        instances = ProviderRegistry.list_instances()
        assert "mock" in instances
        assert instances["mock"] is provider

    def test_clear_registry(self):
        """Test clearing registry."""
        ProviderRegistry.register("mock", MockProvider)
        config = {"api_key": "test_key"}
        ProviderRegistry.create_provider("mock", config)

        ProviderRegistry.clear()

        assert len(ProviderRegistry.list_providers()) == 0
        assert len(ProviderRegistry.list_instances()) == 0


class TestGlobalRegistryFunctions:
    """Test global registry convenience functions."""

    def setup_method(self):
        """Setup for each test method."""
        ProviderRegistry.clear()

    def teardown_method(self):
        """Cleanup after each test method."""
        ProviderRegistry.clear()

    def test_register_provider_function(self):
        """Test global register_provider function."""
        register_provider("mock", MockProvider)

        providers = ProviderRegistry.list_providers()
        assert "mock" in providers
        assert providers["mock"] == MockProvider

    def test_get_provider_function_existing_instance(self):
        """Test getting existing provider instance."""
        register_provider("mock", MockProvider)
        config = {"api_key": "test_key"}

        # Create instance
        provider1 = get_provider("mock", config)

        # Get same instance
        provider2 = get_provider("mock")

        assert provider1 is provider2

    def test_get_provider_function_new_instance(self):
        """Test creating new provider instance."""
        register_provider("mock", MockProvider)
        config = {"api_key": "test_key"}

        provider = get_provider("mock", config)

        assert isinstance(provider, MockProvider)
        assert provider.config == config

    def test_get_provider_function_no_config(self):
        """Test getting provider without config when instance doesn't exist."""
        register_provider("mock", MockProvider)

        with pytest.raises(ValidationError, match="Configuration required"):
            get_provider("mock")

    def test_get_provider_function_not_registered(self):
        """Test getting unregistered provider."""
        config = {"api_key": "test_key"}

        with pytest.raises(ProviderError, match="not registered"):
            get_provider("nonexistent", config)
