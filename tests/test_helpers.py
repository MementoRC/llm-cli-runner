"""Test helpers and mock utilities for MCP Server LLM CLI Runner tests.

This module provides test doubles for testing without API calls.
"""

from typing import Any


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


class MockOpenAIClient:
    """Test double for OpenAI AsyncOpenAI client.

    This replaces complex mock setups with a simple, predictable test double.
    """

    def __init__(self):
        self._responses: list[MockOpenAIResponse] = []
        self._stream_chunks: list[MockStreamingChunk] = []
        self._should_raise: Exception | None = None

    def set_response(self, content: str, tokens_used: int = 25) -> None:
        """Set the next response the client should return."""
        self._responses.append(MockOpenAIResponse(content, tokens_used))

    def set_streaming_response(self, chunks: list[str]) -> None:
        """Set streaming response chunks."""
        self._stream_chunks = [MockStreamingChunk(chunk) for chunk in chunks]
        self._stream_chunks.append(MockStreamingChunk(None))  # End marker

    async def _get_next_response(self) -> MockOpenAIResponse:
        """Get the next configured response."""
        if self._should_raise:
            raise self._should_raise

        if not self._responses:
            return MockOpenAIResponse("Default test response")

        return self._responses.pop(0)

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
