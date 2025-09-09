"""Integration tests for OpenAI provider."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from src.mcp_server_cheap_llm.core.models import (
    LLMRequest,
    ProviderConfig,
    ProviderType,
)
from src.mcp_server_cheap_llm.providers.openai import OpenAIProvider
from tests.test_helpers import MockOpenAIClient, MockResponseBuilder


class TestOpenAIIntegration:
    """Integration tests for OpenAI provider."""

    @pytest.fixture
    def provider_config(self):
        """Create provider configuration for integration tests."""
        return ProviderConfig(
            name="openai_integration",
            provider_type=ProviderType.OPENAI,
            models=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"],
            api_key=os.getenv("OPENAI_API_KEY", "test-key-for-mocking"),
        )

    @pytest.fixture
    def provider(self, provider_config):
        """Create OpenAI provider for integration tests."""
        return OpenAIProvider(config=provider_config)

    @pytest.fixture
    def test_request(self):
        """Create test request."""
        return LLMRequest(
            prompt="Explain what Python is in one sentence.",
            provider=ProviderType.OPENAI,
            max_tokens=100,
            temperature=0.7,
        )

    async def test_provider_initialization_and_validation(self, provider):
        """Test that the provider initializes correctly with valid configuration."""
        # Verify provider is created with correct configuration
        assert provider.name == "openai_integration"
        assert provider.provider_type == ProviderType.OPENAI
        assert provider.config.models == ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"]
        assert provider.config.api_key == os.getenv(
            "OPENAI_API_KEY", "test-key-for-mocking"
        )

        # Verify configuration validation passes
        assert provider.validate_config(provider.config) is True

    @patch("src.mcp_server_cheap_llm.providers.openai.AsyncOpenAI", autospec=True)
    async def test_provider_initialization_with_mocked_client(
        self, mock_openai_class, provider
    ):
        """Test provider initialization with mocked OpenAI client."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Initialize provider
        await provider.initialize()

        # Verify initialization
        assert provider._initialized is True
        assert provider.client is mock_client
        mock_openai_class.assert_called_once()

    async def test_integration_with_test_client(self, provider, test_request):
        """Test full integration with test client (reduced mocking)."""
        # Use test double instead of complex mocking
        test_client = MockOpenAIClient()
        test_client.set_response(
            "Python is a high-level programming language known for its simplicity.",
            tokens_used=25,
        )

        # Inject test client directly
        provider.client = test_client
        provider._initialized = True

        # Test generation
        result = await provider.generate(
            prompt=test_request.prompt,
            model="gpt-4o-mini",
            max_tokens=test_request.max_tokens,
            temperature=test_request.temperature,
        )

        # Verify results
        assert result.success is True
        assert (
            result.content
            == "Python is a high-level programming language known for its simplicity."
        )
        assert result.provider == "openai_integration"
        assert result.model == "gpt-4o-mini"
        assert result.tokens_used == 25

    @patch("src.mcp_server_cheap_llm.providers.openai.AsyncOpenAI", autospec=True)
    async def test_integration_usage_tracking(self, mock_openai_class, provider):
        """Test that usage statistics are tracked during integration."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Mock successful response
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 20
        mock_response.usage.completion_tokens = 12
        mock_response.usage.prompt_tokens = 8

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Initial usage stats
        initial_stats = await provider.get_usage()
        assert initial_stats.total_requests == 0
        assert initial_stats.successful_requests == 0

        # Make a request
        await provider.generate("Test prompt", model="gpt-4o-mini")

        # Check updated stats
        updated_stats = await provider.get_usage()
        assert updated_stats.total_requests == 1
        assert updated_stats.successful_requests == 1
        assert updated_stats.total_tokens == 20

    @patch("src.mcp_server_cheap_llm.providers.openai.AsyncOpenAI", autospec=True)
    async def test_integration_error_handling(self, mock_openai_class, provider):
        """Test error handling in integration scenarios."""
        # Mock the OpenAI client to raise an exception
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        # Test error handling
        result = await provider.generate("Test prompt", model="gpt-4o-mini")

        # Verify error response
        assert result.success is False
        assert "API Error" in result.error_message
        assert result.content == ""
        assert result.tokens_used == 0

        # Check that failure stats are updated
        stats = await provider.get_usage()
        assert stats.total_requests == 1
        assert stats.failed_requests == 1

    @patch("src.mcp_server_cheap_llm.providers.openai.AsyncOpenAI", autospec=True)
    async def test_integration_quota_checking(self, mock_openai_class, provider):
        """Test quota status checking in integration."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Test initial quota status
        quota_status = await provider.check_quota()
        assert quota_status.value in ["healthy", "warning", "exceeded"]

        # Make some successful requests and check quota again
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 10
        mock_response.usage.completion_tokens = 6
        mock_response.usage.prompt_tokens = 4

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Make several requests
        for _ in range(3):
            await provider.generate("Test prompt", model="gpt-4o-mini")

        # Check quota again
        quota_status = await provider.check_quota()
        assert quota_status.value in ["healthy", "warning", "exceeded"]

    @patch("src.mcp_server_cheap_llm.providers.openai.AsyncOpenAI", autospec=True)
    async def test_integration_health_check(self, mock_openai_class, provider):
        """Test health check integration."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.models.list = AsyncMock()

        # Mock generation for health test
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.prompt_tokens = 2

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Perform health check
        health_status = await provider.health_check()

        # Verify health check results
        assert health_status["provider"] == "openai_integration"
        assert "available" in health_status
        assert "test_generation" in health_status
        assert "timestamp" in health_status
        assert "models_configured" in health_status
        assert (
            health_status["models_configured"] == 3
        )  # gpt-4o-mini, gpt-3.5-turbo, gpt-4

    async def test_integration_model_info(self, provider):
        """Test model information retrieval."""
        # Test getting info for available model
        model_info = await provider.get_model_info("gpt-4")
        assert model_info["available"] is True
        assert model_info["model"] == "gpt-4"
        assert model_info["provider"] == "openai_integration"
        assert "description" in model_info

        # Test getting info for unavailable model
        unavailable_info = await provider.get_model_info("unknown-model")
        assert unavailable_info["available"] is False
        assert "error" in unavailable_info

    async def test_integration_cost_estimation(self, provider):
        """Test cost estimation integration."""
        # Test cost estimation
        cost_info = await provider.estimate_cost(
            "Hello world", model="gpt-3.5-turbo", max_tokens=100
        )

        # Verify cost information
        assert hasattr(cost_info, "estimated_cost_usd")
        assert cost_info.cost_breakdown["model"] == "gpt-3.5-turbo"
        assert isinstance(cost_info.estimated_cost_usd, float)
        assert cost_info.estimated_cost_usd >= 0

    async def test_integration_streaming_with_test_client(self, provider):
        """Test streaming generation with test client (reduced mocking)."""
        # Use test double for streaming
        test_client = MockOpenAIClient()
        test_client.set_streaming_response(["Hello", " world", "!"])

        # Inject test client
        provider.client = test_client
        provider._initialized = True

        # Test streaming
        results = []
        async for chunk in provider.generate_stream("Hello", model="gpt-4o-mini"):
            results.append(chunk)

        # Verify streaming results
        assert len(results) == 4  # 3 content chunks + 1 final chunk
        assert results[0]["content"] == "Hello"
        assert results[1]["content"] == " world"
        assert results[2]["content"] == "!"
        assert results[3]["is_final"] is True

    async def test_integration_cleanup(self, provider):
        """Test provider cleanup integration."""
        # Set up a mock client
        from unittest.mock import AsyncMock

        provider.client = AsyncMock()
        provider._initialized = True

        # Test cleanup
        await provider.cleanup()

        # Verify cleanup
        assert provider.client is None
        assert provider._initialized is False
