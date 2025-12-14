"""Test helpers and mock utilities for MCP Server Cheap LLM tests."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock


class MockResponseBuilder:
    """Builder pattern for creating mock responses."""

    def __init__(self):
        self.response_data = {}

    def with_content(self, content: str) -> "MockResponseBuilder":
        """Set response content."""
        self.response_data["content"] = content
        return self

    def with_tokens(
        self, total: int, completion: int = None, prompt: int = None
    ) -> "MockResponseBuilder":
        """Set token usage information."""
        self.response_data["tokens"] = {
            "total": total,
            "completion": completion or total // 2,
            "prompt": prompt or total // 2,
        }
        return self

    def with_model(self, model: str) -> "MockResponseBuilder":
        """Set model information."""
        self.response_data["model"] = model
        return self

    def build(self) -> MagicMock:
        """Build the mock response object."""
        mock_response = MagicMock()

        # Set up choices
        mock_choice = MagicMock()
        mock_choice.message.content = self.response_data.get("content", "Mock response")
        mock_response.choices = [mock_choice]

        # Set up usage
        if "tokens" in self.response_data:
            mock_usage = MagicMock()
            tokens = self.response_data["tokens"]
            mock_usage.total_tokens = tokens["total"]
            mock_usage.completion_tokens = tokens["completion"]
            mock_usage.prompt_tokens = tokens["prompt"]
            mock_response.usage = mock_usage
        else:
            mock_response.usage = None

        return mock_response


class MockOpenAIClient:
    """Mock OpenAI client for testing without API calls."""

    def __init__(self):
        self._response_content = "Default mock response"
        self._response_tokens = 10
        self._streaming_chunks = []
        self._should_raise = None

        # Set up the nested structure properly
        self.chat = self._create_chat_mock()
        self.models = self._create_models_mock()

    def _create_chat_mock(self):
        """Create the chat mock with proper structure."""
        chat_mock = MagicMock()
        chat_mock.completions = MagicMock()
        chat_mock.completions.create = self._create_completion_method()
        return chat_mock

    def _create_models_mock(self):
        """Create the models mock with proper structure."""
        models_mock = MagicMock()
        models_mock.list = self._create_models_list_method()
        return models_mock

    def _create_completion_method(self):
        """Create the async completion method."""

        async def create_completion(*args, **kwargs):
            if kwargs.get("stream", False):
                return self._create_streaming_response()
            else:
                return self._create_mock_response()

        return create_completion

    def _create_models_list_method(self):
        """Create the async models list method."""

        async def list_models():
            # Mock models list response
            mock_models = MagicMock()
            mock_models.data = []
            return mock_models

        return list_models

    def set_response(self, content: str, tokens_used: int = 10):
        """Set the response that will be returned by generate calls."""
        self._response_content = content
        self._response_tokens = tokens_used

    def set_streaming_response(self, chunks: list[str]):
        """Set the chunks that will be returned by streaming calls."""
        self._streaming_chunks = chunks

    def set_error(self, error: Exception):
        """Set an error to be raised on the next call."""
        self._should_raise = error

    async def close(self):
        """Mock client close method."""
        pass

    def _create_mock_response(self) -> MagicMock:
        """Create a mock response object."""
        if self._should_raise:
            error = self._should_raise
            self._should_raise = None  # Reset for next call
            raise error

        mock_response = MagicMock()

        # Set up choices
        mock_choice = MagicMock()
        mock_choice.message.content = self._response_content
        mock_response.choices = [mock_choice]

        # Set up usage
        mock_usage = MagicMock()
        mock_usage.total_tokens = self._response_tokens
        mock_usage.completion_tokens = self._response_tokens // 2
        mock_usage.prompt_tokens = self._response_tokens // 2
        mock_response.usage = mock_usage

        return mock_response

    async def _create_streaming_response(self) -> AsyncIterator[MagicMock]:
        """Create a mock streaming response."""
        if self._should_raise:
            error = self._should_raise
            self._should_raise = None
            raise error

        for _i, chunk_content in enumerate(self._streaming_chunks):
            mock_chunk = MagicMock()
            mock_delta = MagicMock()
            mock_delta.content = chunk_content
            mock_choice = MagicMock()
            mock_choice.delta = mock_delta
            mock_chunk.choices = [mock_choice]
            yield mock_chunk


class MockStreamChunk:
    """Mock streaming chunk for OpenAI responses."""

    def __init__(self, content: str, is_final: bool = False):
        self.content = content
        self.is_final = is_final

        # Mock the structure of an OpenAI streaming chunk
        self.choices = [MagicMock()]
        self.choices[0].delta = MagicMock()
        self.choices[0].delta.content = content if not is_final else None


class MockProviderFactory:
    """Factory for creating mock providers with different configurations."""

    @staticmethod
    def create_openai_provider(config_overrides: dict[str, Any] = None):
        """Create a mock OpenAI provider with test configuration."""
        from unittest.mock import Mock

        from src.mcp_server_cheap_llm.core.models import ProviderConfig, ProviderType

        config = ProviderConfig(
            name="test_openai",
            provider_type=ProviderType.OPENAI,
            models=["gpt-4o-mini", "gpt-3.5-turbo"],
            api_key="test-key",
            **(config_overrides or {}),
        )

        provider_mock = Mock()
        provider_mock.config = config
        provider_mock.name = config.name
        provider_mock.provider_type = config.provider_type
        provider_mock._initialized = True

        return provider_mock

    @staticmethod
    def create_gemini_provider(config_overrides: dict[str, Any] = None):
        """Create a mock Gemini provider with test configuration."""
        from unittest.mock import Mock

        from src.mcp_server_cheap_llm.core.models import ProviderConfig, ProviderType

        config = ProviderConfig(
            name="test_gemini",
            provider_type=ProviderType.GEMINI,
            models=["gemini-1.5-flash"],
            cli_path="/usr/local/bin/gemini",
            **(config_overrides or {}),
        )

        provider_mock = Mock()
        provider_mock.config = config
        provider_mock.name = config.name
        provider_mock.provider_type = config.provider_type
        provider_mock._initialized = True

        return provider_mock


def create_mock_llm_response(
    content: str = "Mock response",
    success: bool = True,
    provider: str = "test_provider",
    model: str = "test_model",
    tokens_used: int = 10,
    error_message: str = None,
):
    """Create a mock LLMResponse object for testing."""
    from src.mcp_server_cheap_llm.core.models import LLMResponse

    return LLMResponse(
        content=content,
        success=success,
        provider=provider,
        model=model,
        tokens_used=tokens_used,
        error_message=error_message,
        response_time_ms=100,
        metadata={},
    )


def create_mock_llm_request(
    prompt: str = "Test prompt",
    model: str = "test_model",
    max_tokens: int = 100,
    temperature: float = 0.7,
):
    """Create a mock LLMRequest object for testing."""
    from src.mcp_server_cheap_llm.core.models import LLMRequest

    return LLMRequest(
        prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature
    )
