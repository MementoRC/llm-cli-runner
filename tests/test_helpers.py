"""Test helpers and mock utilities for MCP Server LLM CLI Runner tests.

This module provides test doubles, builders, and utilities to replace
complex mocking patterns with more maintainable alternatives.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional, Union
from unittest.mock import MagicMock

from src.mcp_server_llm_cli_runner.core.models import ProviderConfig, ProviderType


class MockOpenAIResponse:
    """Test double for OpenAI API response objects."""

    def __init__(self, content: str, tokens_used: int = 25):
        self.content = content
        self.tokens_used = tokens_used
        self.choices = [self._create_choice(content)]
        self.usage = self._create_usage(tokens_used)

    def _create_choice(self, content: str):
        """Create a mock choice object."""
        choice = type("Choice", (), {})()
        choice.message = type("Message", (), {})()
        choice.message.content = content
        return choice

    def _create_usage(self, total_tokens: int):
        """Create a mock usage object."""
        usage = type("Usage", (), {})()
        usage.total_tokens = total_tokens
        usage.completion_tokens = int(total_tokens * 0.6)
        usage.prompt_tokens = total_tokens - usage.completion_tokens
        return usage


class MockStreamingChunk:
    """Test double for OpenAI streaming response chunks."""

    def __init__(self, content: str | None = None):
        self.choices = [self._create_delta_choice(content)]

    def _create_delta_choice(self, content: str | None):
        """Create a mock delta choice."""
        choice = type("Choice", (), {})()
        choice.delta = type("Delta", (), {})()
        choice.delta.content = content
        return choice


class ConfigBuilder:
    """Builder pattern for creating test configuration objects."""

    def __init__(self):
        self._name = "test"
        self._provider_type = ProviderType.OPENAI
        self._models = ["gpt-4o-mini"]
        self._api_key = "test-api-key"
        self._enabled = True
        self._max_tokens = 100
        self._timeout = 30

    def with_name(self, name: str) -> "ConfigBuilder":
        """Set the provider name."""
        self._name = name
        return self

    def with_provider_type(self, provider_type: ProviderType) -> "ConfigBuilder":
        """Set the provider type."""
        self._provider_type = provider_type
        return self

    def with_models(self, models: list[str]) -> "ConfigBuilder":
        """Set the available models."""
        self._models = models
        return self

    def with_api_key(self, api_key: str) -> "ConfigBuilder":
        """Set the API key."""
        self._api_key = api_key
        return self

    def disabled(self) -> "ConfigBuilder":
        """Mark provider as disabled."""
        self._enabled = False
        return self

    def with_max_tokens(self, max_tokens: int) -> "ConfigBuilder":
        """Set max tokens."""
        self._max_tokens = max_tokens
        return self

    def build(self) -> ProviderConfig:
        """Build the configuration object."""
        return ProviderConfig(
            name=self._name,
            provider_type=self._provider_type,
            models=self._models,
            api_key=self._api_key,
            enabled=self._enabled,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
        )


class MockOpenAIClient:
    """Test double for OpenAI AsyncOpenAI client.

    This replaces complex mock setups with a simple, predictable test double.
    """

    def __init__(self):
        self._responses: list[MockOpenAIResponse] = []
        self._stream_chunks: list[MockStreamingChunk] = []
        self._should_raise: Exception | None = None
        self._call_count = 0

    def set_response(self, content: str, tokens_used: int = 25) -> None:
        """Set the next response the client should return."""
        self._responses.append(MockOpenAIResponse(content, tokens_used))

    def set_streaming_response(self, chunks: list[str]) -> None:
        """Set streaming response chunks."""
        self._stream_chunks = [MockStreamingChunk(chunk) for chunk in chunks]
        self._stream_chunks.append(MockStreamingChunk(None))  # End marker

    def set_error(self, error: Exception) -> None:
        """Set an error to be raised on next call."""
        self._should_raise = error

    async def _get_next_response(self) -> MockOpenAIResponse:
        """Get the next configured response."""
        if self._should_raise:
            raise self._should_raise

        if not self._responses:
            # Default response if none configured
            return MockOpenAIResponse("Default test response")

        response = self._responses.pop(0)
        self._call_count += 1
        return response

    @property
    def chat(self):
        """Chat completions interface."""
        return self.Chat(self)

    @property
    def models(self):
        """Models interface."""
        return self.Models()

    class Chat:
        """Chat interface that contains completions."""

        def __init__(self, client):
            self._client = client
            self.completions = client.ChatCompletions(client)

    class ChatCompletions:
        """Chat completions endpoint mock."""

        def __init__(self, client):
            self._client = client

        async def create(self, **kwargs) -> MockOpenAIResponse:
            """Create a chat completion."""
            # Verify required parameters are present
            if "model" not in kwargs:
                raise ValueError("model parameter is required")
            if "messages" not in kwargs:
                raise ValueError("messages parameter is required")

            if kwargs.get("stream", False):
                return self._client._get_streaming_response()

            return await self._client._get_next_response()

    class Models:
        """Models endpoint mock."""

        async def list(self) -> dict[str, Any]:
            """List available models."""
            return {"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-3.5-turbo"}]}

    def _get_streaming_response(self):
        """Return streaming response generator."""

        async def stream_generator():
            for chunk in self._stream_chunks:
                yield chunk

        return stream_generator()


@dataclass
class LLMRequestBuilder:
    """Builder for creating LLM request objects."""

    prompt: str = "Test prompt"
    provider: ProviderType | None = None
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False

    def with_prompt(self, prompt: str) -> "LLMRequestBuilder":
        """Set the prompt."""
        self.prompt = prompt
        return self

    def with_provider(self, provider: ProviderType) -> "LLMRequestBuilder":
        """Set the provider."""
        self.provider = provider
        return self

    def with_model(self, model: str) -> "LLMRequestBuilder":
        """Set the model."""
        self.model = model
        return self

    def with_max_tokens(self, max_tokens: int) -> "LLMRequestBuilder":
        """Set max tokens."""
        self.max_tokens = max_tokens
        return self

    def with_temperature(self, temperature: float) -> "LLMRequestBuilder":
        """Set temperature."""
        self.temperature = temperature
        return self

    def streaming(self) -> "LLMRequestBuilder":
        """Enable streaming."""
        self.stream = True
        return self

    def complex_prompt(self) -> "LLMRequestBuilder":
        """Set a complex prompt for testing routing."""
        self.prompt = """
        Design a distributed consensus algorithm with Byzantine fault tolerance.
        Analyze the performance characteristics of Raft vs PBFT algorithms.
        Implement optimization strategies using advanced data structures.
        """
        return self

    def simple_prompt(self) -> "LLMRequestBuilder":
        """Set a simple prompt for testing routing."""
        self.prompt = "Hello, how are you?"
        return self

    def build(self):
        """Build the LLM request object."""
        # Import here to avoid circular imports
        from src.mcp_server_llm_cli_runner.core.models import LLMRequest

        kwargs = {"prompt": self.prompt}
        if self.provider:
            kwargs["provider"] = self.provider
        if self.model:
            kwargs["model"] = self.model
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.stream:
            kwargs["stream"] = self.stream

        return LLMRequest(**kwargs)


def create_test_providers() -> dict[str, Any]:
    """Create a set of test provider configurations."""
    return {
        "gemini": ConfigBuilder()
        .with_name("gemini")
        .with_provider_type(ProviderType.GEMINI)
        .with_models(["gemini-pro"])
        .build(),
        "openai": ConfigBuilder()
        .with_name("openai")
        .with_provider_type(ProviderType.OPENAI)
        .with_models(["gpt-4", "gpt-3.5-turbo"])
        .build(),
        "llama": ConfigBuilder()
        .with_name("llama")
        .with_provider_type(ProviderType.LLAMA)
        .with_models(["llama-7b"])
        .build(),
    }


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


class MockOpenAIClientSimple:
    """Mock OpenAI client for testing without API calls (simple version)."""

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

        from src.mcp_server_llm_cli_runner.core.models import (
            ProviderConfig,
            ProviderType,
        )

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

        from src.mcp_server_llm_cli_runner.core.models import (
            ProviderConfig,
            ProviderType,
        )

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
    from src.mcp_server_llm_cli_runner.core.models import LLMResponse

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
    from src.mcp_server_llm_cli_runner.core.models import LLMRequest

    return LLMRequest(
        prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature
    )
