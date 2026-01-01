# Provider Development Guide

## Overview

This guide explains how to extend MCP Server Cheap LLM with custom LLM providers.

## Provider Architecture

```
providers/
├── base.py          # Abstract base class
├── gemini.py        # Gemini implementation
├── openai.py        # OpenAI/Codex implementation
├── llama.py         # LLaMA implementation
└── your_provider.py # Your custom provider
```

## Creating a New Provider

### Step 1: Create Provider Class

```python
# src/mcp_server_cheap_llm/providers/your_provider.py
"""Custom provider implementation."""

from typing import Any

from mcp_server_cheap_llm.providers.base import BaseProvider
from mcp_server_cheap_llm.core.models import GenerateRequest, GenerateResponse
from mcp_server_cheap_llm.utils.logging import get_logger

logger = get_logger(__name__)


class YourProvider(BaseProvider):
    """Custom LLM provider implementation.

    This provider integrates with YourService API to provide
    text generation capabilities.

    Attributes:
        name: Provider identifier
        api_key: API key for authentication
        model_name: Default model to use

    Example:
        >>> provider = YourProvider(
        ...     name="your-provider",
        ...     api_key="sk-xxx",
        ...     model_name="your-model-v1"
        ... )
        >>> await provider.initialize()
        >>> result = await provider.generate("Hello world")
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        model_name: str = "default-model",
        **config: Any,
    ) -> None:
        """Initialize provider.

        Args:
            name: Provider identifier
            api_key: API key for authentication
            model_name: Model to use for generation
            **config: Additional configuration options
        """
        super().__init__(name)
        self.api_key = api_key
        self.model_name = model_name
        self.config = config
        self._client = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the provider and establish connections.

        This method should:
        1. Validate configuration
        2. Establish API connections
        3. Verify credentials
        4. Load any required resources

        Raises:
            ProviderError: If initialization fails
        """
        if self._initialized:
            return

        logger.info(f"Initializing {self.name} provider")

        # Validate configuration
        if not self.api_key:
            raise ValueError("API key is required")

        # Initialize client
        self._client = YourAPIClient(
            api_key=self.api_key,
            **self.config
        )

        # Verify connection
        await self._client.ping()

        self._initialized = True
        logger.info(f"{self.name} provider initialized successfully")

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            **kwargs: Additional generation parameters

        Returns:
            Generated text response

        Raises:
            ProviderError: If generation fails
            RateLimitError: If rate limit exceeded
        """
        if not self._initialized:
            await self.initialize()

        logger.debug(f"Generating with {self.name}: {prompt[:50]}...")

        try:
            response = await self._client.generate(
                prompt=prompt,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            logger.debug(f"Generated {len(response)} characters")
            return response

        except RateLimitException as e:
            logger.warning(f"Rate limit hit: {e}")
            raise RateLimitError(str(e)) from e

        except APIException as e:
            logger.error(f"Generation failed: {e}")
            raise ProviderError(str(e)) from e

    async def generate_stream(
        self,
        prompt: str,
        **kwargs: Any,
    ):
        """Generate text with streaming output.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Yields:
            Text chunks as they are generated
        """
        if not self._initialized:
            await self.initialize()

        async for chunk in self._client.generate_stream(
            prompt=prompt,
            model=self.model_name,
            **kwargs
        ):
            yield chunk

    async def get_provider_info(self) -> dict[str, Any]:
        """Get provider information and status.

        Returns:
            Dictionary containing:
            - name: Provider name
            - version: Provider version
            - models: Available models
            - status: Current status
        """
        return {
            "name": self.name,
            "type": "your-provider",
            "version": "1.0.0",
            "models": await self._get_available_models(),
            "status": "ready" if self._initialized else "not_initialized",
        }

    async def health_check(self) -> bool:
        """Check provider health.

        Returns:
            True if provider is healthy
        """
        try:
            await self._client.ping()
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        """Clean up provider resources."""
        if self._client:
            await self._client.close()
        self._initialized = False
        logger.info(f"{self.name} provider shut down")

    async def _get_available_models(self) -> list[str]:
        """Get list of available models."""
        if not self._client:
            return []
        return await self._client.list_models()
```

### Step 2: Implement Required Methods

The base class requires these methods:

| Method | Required | Description |
|--------|----------|-------------|
| `initialize()` | Yes | Set up connections |
| `generate()` | Yes | Generate text |
| `generate_stream()` | No | Streaming generation |
| `get_provider_info()` | Yes | Return provider metadata |
| `health_check()` | Yes | Check provider health |
| `shutdown()` | Yes | Clean up resources |

### Step 3: Register Provider

```python
# src/mcp_server_cheap_llm/providers/__init__.py

from .base import BaseProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .llama import LlamaProvider
from .your_provider import YourProvider  # Add this

PROVIDER_REGISTRY = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "llama": LlamaProvider,
    "your-provider": YourProvider,  # Add this
}

def get_provider(provider_type: str, **config) -> BaseProvider:
    """Factory function to create providers."""
    if provider_type not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider_type}")

    return PROVIDER_REGISTRY[provider_type](**config)
```

### Step 4: Add Configuration Schema

```python
# src/mcp_server_cheap_llm/core/config.py

YOUR_PROVIDER_SCHEMA = {
    "type": "object",
    "properties": {
        "api_key": {"type": "string"},
        "model_name": {"type": "string", "default": "default-model"},
        "timeout": {"type": "integer", "default": 30},
        "max_retries": {"type": "integer", "default": 3},
    },
    "required": ["api_key"],
}
```

### Step 5: Create Tool Definition

```python
# In your provider or tool registration

def get_tool_definition() -> dict:
    """Return MCP tool definition."""
    return {
        "name": "your_provider_generate",
        "description": "Generate text using YourProvider",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt to generate from",
                },
                "model": {
                    "type": "string",
                    "description": "Model to use",
                    "default": "default-model",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate",
                    "default": 1024,
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
            },
            "required": ["prompt"],
        },
    }
```

## Writing Tests

### Unit Tests

```python
# tests/unit/test_your_provider.py
"""Unit tests for YourProvider."""

import pytest
from unittest.mock import AsyncMock, patch

from mcp_server_cheap_llm.providers.your_provider import YourProvider


class TestYourProvider:
    """Test suite for YourProvider."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return YourProvider(
            name="test-provider",
            api_key="test-key",
            model_name="test-model",
        )

    @pytest.mark.asyncio
    async def test_initialize_success(self, provider):
        """Test successful initialization."""
        with patch.object(provider, '_client') as mock_client:
            mock_client.ping = AsyncMock()
            await provider.initialize()
            assert provider._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_no_api_key(self):
        """Test initialization without API key."""
        provider = YourProvider(name="test", api_key="")
        with pytest.raises(ValueError, match="API key"):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        provider._initialized = True
        provider._client = AsyncMock()
        provider._client.generate = AsyncMock(return_value="Generated text")

        result = await provider.generate("Test prompt")

        assert result == "Generated text"
        provider._client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_auto_initializes(self, provider):
        """Test that generate auto-initializes if needed."""
        with patch.object(provider, 'initialize') as mock_init:
            mock_init.return_value = None
            provider._initialized = False
            provider._client = AsyncMock()
            provider._client.generate = AsyncMock(return_value="text")

            await provider.generate("prompt")

            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, provider):
        """Test health check when healthy."""
        provider._client = AsyncMock()
        provider._client.ping = AsyncMock()

        result = await provider.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, provider):
        """Test health check when unhealthy."""
        provider._client = AsyncMock()
        provider._client.ping = AsyncMock(side_effect=Exception("Connection failed"))

        result = await provider.health_check()

        assert result is False
```

### Integration Tests

```python
# tests/integration/test_your_provider_integration.py
"""Integration tests for YourProvider."""

import os
import pytest

from mcp_server_cheap_llm.providers.your_provider import YourProvider


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("YOUR_API_KEY"),
    reason="YOUR_API_KEY not set"
)
class TestYourProviderIntegration:
    """Integration tests requiring real API access."""

    @pytest.fixture
    async def provider(self):
        """Create and initialize provider."""
        provider = YourProvider(
            name="integration-test",
            api_key=os.environ["YOUR_API_KEY"],
        )
        await provider.initialize()
        yield provider
        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_real_generation(self, provider):
        """Test generation with real API."""
        result = await provider.generate(
            "Say hello in exactly 3 words",
            max_tokens=10,
        )
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_streaming_generation(self, provider):
        """Test streaming generation."""
        chunks = []
        async for chunk in provider.generate_stream("Count to 5"):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0
```

## Best Practices

### Error Handling

```python
from mcp_server_cheap_llm.core.errors import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)

async def generate(self, prompt: str, **kwargs) -> str:
    try:
        return await self._client.generate(prompt, **kwargs)
    except AuthException:
        raise AuthenticationError("Invalid API key")
    except RateLimitException as e:
        raise RateLimitError(str(e), retry_after=e.retry_after)
    except ModelNotFoundException as e:
        raise ModelNotFoundError(f"Model not found: {e.model}")
    except Exception as e:
        raise ProviderError(f"Generation failed: {e}")
```

### Logging

```python
from mcp_server_cheap_llm.utils.logging import get_logger

logger = get_logger(__name__)

# Use appropriate log levels
logger.debug("Detailed debugging info")
logger.info("Important operations")
logger.warning("Potential issues")
logger.error("Errors that need attention")
```

### Configuration

```python
# Support both environment variables and config file
import os

class YourProvider:
    def __init__(self, api_key: str | None = None, **config):
        self.api_key = api_key or os.getenv("YOUR_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set YOUR_API_KEY env var "
                "or pass api_key parameter."
            )
```

### Rate Limiting

```python
from asyncio import Semaphore

class YourProvider:
    def __init__(self, max_concurrent: int = 10, **config):
        self._semaphore = Semaphore(max_concurrent)

    async def generate(self, prompt: str, **kwargs) -> str:
        async with self._semaphore:
            return await self._do_generate(prompt, **kwargs)
```

### Caching

```python
from functools import lru_cache
import hashlib

class YourProvider:
    def __init__(self, cache_enabled: bool = True, **config):
        self._cache_enabled = cache_enabled
        self._cache = {}

    def _cache_key(self, prompt: str, **kwargs) -> str:
        content = f"{prompt}:{sorted(kwargs.items())}"
        return hashlib.md5(content.encode()).hexdigest()

    async def generate(self, prompt: str, **kwargs) -> str:
        if self._cache_enabled:
            key = self._cache_key(prompt, **kwargs)
            if key in self._cache:
                return self._cache[key]

        result = await self._do_generate(prompt, **kwargs)

        if self._cache_enabled:
            self._cache[key] = result

        return result
```

## Checklist

Before submitting your provider:

- [ ] Implements all required base class methods
- [ ] Has comprehensive unit tests (>80% coverage)
- [ ] Has integration tests (can be skipped in CI)
- [ ] Includes proper error handling
- [ ] Uses structured logging
- [ ] Supports configuration via env vars and config file
- [ ] Includes docstrings with examples
- [ ] Has rate limiting/concurrency control
- [ ] Cleans up resources on shutdown
- [ ] Is registered in provider registry
- [ ] Has tool definition for MCP
