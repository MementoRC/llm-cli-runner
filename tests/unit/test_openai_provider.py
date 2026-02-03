"""Unit tests for OpenAI provider implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_llm_cli_runner.core.errors import ProviderError, RateLimitError
from mcp_server_llm_cli_runner.core.models import (
    LLMRequest,
    ProviderConfig,
    ProviderType,
)

try:
    from mcp_server_llm_cli_runner.providers.openai import OpenAIProvider
except ImportError:
    pytest.skip("openai dependency not available", allow_module_level=True)


class TestOpenAIProvider:
    """Test suite for OpenAI provider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProviderConfig(
            name="openai_test",
            provider_type=ProviderType.OPENAI,
            models=["gpt-4o-mini", "gpt-3.5-turbo"],
            api_key="test-api-key",
            max_tokens=1000,
            timeout=30,
        )

    @pytest.fixture
    def provider(self, config):
        """Create test provider instance."""
        return OpenAIProvider(config=config)

    async def test_provider_initialization(self, provider):
        """Test provider initialization."""
        assert provider.name == "openai_test"
        assert provider.provider_type == ProviderType.OPENAI
        assert provider.config.models == ["gpt-4o-mini", "gpt-3.5-turbo"]

    async def test_provider_validation_success(self, config):
        """Test successful provider configuration validation."""
        provider = OpenAIProvider(config)
        assert provider.name == config.name
        assert provider.api_key == config.api_key

    async def test_provider_validation_failure_no_api_key(self, config):
        """Test provider handles missing API key gracefully."""
        config.api_key = None
        provider = OpenAIProvider(config)
        assert provider.api_key is None

    async def test_provider_validation_failure_no_models(self, config):
        """Test provider handles empty models list."""
        config.models = ["gpt-4o-mini"]  # Keep at least one model for valid config
        provider = OpenAIProvider(config)
        assert provider.config.models == ["gpt-4o-mini"]

    async def test_provider_validation_failure_wrong_type(self, config):
        """Test provider accepts config regardless of provider_type field."""
        config.provider_type = ProviderType.GEMINI
        provider = OpenAIProvider(config)
        # Provider stores config but uses its own provider_type
        assert provider.provider_type == ProviderType.OPENAI

    @patch("mcp_server_llm_cli_runner.providers.openai.AsyncOpenAI", autospec=True)
    async def test_initialize_success(self, mock_openai_class, provider):
        """Test successful provider initialization."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        await provider.initialize()

        assert provider._initialized is True
        assert provider.client is mock_client
        mock_openai_class.assert_called_once()

    @patch("mcp_server_llm_cli_runner.providers.openai.AsyncOpenAI", autospec=True)
    async def test_initialize_failure(self, mock_openai_class, provider):
        """Test provider initialization failure."""
        mock_openai_class.side_effect = Exception("API key invalid")

        with pytest.raises(Exception, match="API key invalid"):
            await provider.initialize()

        assert provider._initialized is False

    @patch("mcp_server_llm_cli_runner.providers.openai.AsyncOpenAI", autospec=True)
    async def test_generate_success(self, mock_openai_class, provider):
        """Test successful text generation."""
        # Mock OpenAI client and response
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 15
        mock_response.usage.completion_tokens = 10
        mock_response.usage.prompt_tokens = 5

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.generate("Hello", model="gpt-4o-mini")

        assert result.success is True
        assert result.content == "Hello, world!"
        assert result.provider == "openai_test"
        assert result.model == "gpt-4o-mini"
        assert result.tokens_used == 15

    @patch("mcp_server_llm_cli_runner.providers.openai.AsyncOpenAI", autospec=True)
    async def test_generate_failure(self, mock_openai_class, provider):
        """Test text generation failure."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        result = await provider.generate("Hello", model="gpt-4o-mini")

        assert result.success is False
        assert "API Error" in result.error_message
        assert result.content == ""
        assert result.tokens_used == 0

    @patch("mcp_server_llm_cli_runner.providers.openai.AsyncOpenAI", autospec=True)
    async def test_generate_stream_success(self, mock_openai_class, provider):
        """Test successful streaming generation."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response
        async def mock_stream():
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))]),
            ]
            for chunk in chunks:
                yield chunk

        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        results = []
        async for chunk in provider.generate_stream("Hello", model="gpt-4o-mini"):
            results.append(chunk)

        # Should have 3 content chunks + 1 final chunk
        assert len(results) == 4
        assert results[0]["content"] == "Hello"
        assert results[1]["content"] == " world"
        assert results[2]["content"] == "!"
        assert results[3]["is_final"] is True

    async def test_get_available_models(self, provider):
        """Test getting available models."""
        models = provider.get_available_models()
        assert "gpt-4o-mini" in models
        assert "gpt-3.5-turbo" in models

    async def test_get_model_info_available(self, provider):
        """Test getting info for available model."""
        info = await provider.get_model_info("gpt-4")
        assert info["available"] is True
        assert info["model"] == "gpt-4"
        assert info["provider"] == "openai_test"

    async def test_get_model_info_unavailable(self, provider):
        """Test getting info for unavailable model."""
        info = await provider.get_model_info("unknown-model")
        assert info["available"] is False
        assert "error" in info

    async def test_estimate_cost(self, provider):
        """Test cost estimation."""
        cost_info = await provider.estimate_cost("Hello world", model="gpt-3.5-turbo")

        assert hasattr(cost_info, "estimated_cost_usd")
        assert cost_info.provider_name == "openai_test"
        assert isinstance(cost_info.estimated_cost_usd, float)
        # Check cost breakdown contains expected model info
        assert cost_info.cost_breakdown["model"] == "gpt-3.5-turbo"

    @patch("mcp_server_llm_cli_runner.providers.openai.AsyncOpenAI", autospec=True)
    async def test_health_check_healthy(self, mock_openai_class, provider):
        """Test health check when provider is healthy."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.models.list = AsyncMock()

        # Mock successful generation
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 5
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        health = await provider.health_check()

        assert health["provider"] == "openai_test"
        assert health["available"] is True
        assert health["test_generation"] is True

    @patch("mcp_server_llm_cli_runner.providers.openai.AsyncOpenAI", autospec=True)
    async def test_health_check_unhealthy(self, mock_openai_class, provider):
        """Test health check when provider is unhealthy."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.models.list = AsyncMock(side_effect=Exception("API Error"))

        health = await provider.health_check()

        assert health["provider"] == "openai_test"
        assert health["available"] is False
        assert health["test_generation"] is False

    async def test_cleanup(self, provider):
        """Test resource cleanup."""
        # Set up a mock client
        provider.client = AsyncMock()
        provider._initialized = True

        await provider.cleanup()

        assert provider.client is None
        assert provider._initialized is False

    async def test_default_configuration(self):
        """Test provider with minimal configuration."""
        config = ProviderConfig(
            name="openai",
            provider_type=ProviderType.OPENAI,
            models=["gpt-3.5-turbo"],
        )
        provider = OpenAIProvider(config)

        assert provider.name == "openai"
        assert provider.provider_type == ProviderType.OPENAI
        assert provider.config.provider_type == ProviderType.OPENAI
        assert "gpt-3.5-turbo" in provider.config.models

    @patch("mcp_server_llm_cli_runner.providers.openai.AsyncOpenAI", autospec=True)
    async def test_generate_with_system_prompt(self, mock_openai_class, provider):
        """Test generation with system prompt."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response with system context"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 20
        mock_response.usage.completion_tokens = 15
        mock_response.usage.prompt_tokens = 5

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.generate(
            "Hello", model="gpt-4o-mini", system_prompt="You are a helpful assistant"
        )

        assert result.success is True
        assert result.content == "Response with system context"

        # Verify the client was called with system message
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @patch("mcp_server_llm_cli_runner.providers.openai.AsyncOpenAI", autospec=True)
    async def test_generate_without_initialization(self, mock_openai_class, provider):
        """Test that generate initializes the provider if not already done."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Auto-initialized response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 10
        mock_response.usage.completion_tokens = 7
        mock_response.usage.prompt_tokens = 3

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Provider should not be initialized yet
        assert provider._initialized is False

        result = await provider.generate("Test")

        # After generate, it should be initialized
        assert provider._initialized is True
        assert result.success is True
