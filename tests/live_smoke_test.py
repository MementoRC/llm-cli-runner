#!/usr/bin/env python3
"""Live smoke test - actually calls the LLM provider."""

import pytest

from mcp_server_llm_cli_runner.core.models import LLMRequest
from mcp_server_llm_cli_runner.providers.agy import AgyProvider


@pytest.mark.asyncio
async def test_agy_live_call():
    """Actually call agy CLI and verify response."""
    provider = AgyProvider()

    # Check if CLI is available
    is_available = await provider.is_available()
    if not is_available:
        pytest.skip("agy CLI not available")

    # Make a real request
    request = LLMRequest(
        prompt="Say 'Hello from live test' and nothing else.",
        system_prompt=None,
        max_tokens=50,
        temperature=0.1,
    )

    response = await provider.generate(request)

    # Verify response structure
    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0
    assert response.success is True
    assert response.provider == "agy"
    print(f"✅ Live response: {response.content[:100]}...")
    print(f"✅ Tokens used: {response.tokens_used}")
    print(f"✅ Response time: {response.response_time_ms:.0f}ms")


@pytest.mark.asyncio
async def test_cache_service_live():
    """Test cache service with real operations."""
    from mcp_server_llm_cli_runner.cache.service import CacheService

    cache = CacheService()

    # Test set/get
    await cache.set("live_test_key", {"test": "data"}, ttl=60)
    result = await cache.get("live_test_key")
    assert result == {"test": "data"}
    print("✅ Cache set/get works")

    # Test delete
    await cache.delete("live_test_key")
    result = await cache.get("live_test_key")
    assert result is None
    print("✅ Cache delete works")

    print("✅ Cache operations verified")


@pytest.mark.asyncio
async def test_provider_initialization():
    """Test all providers can initialize."""
    from mcp_server_llm_cli_runner.providers.llama import LLaMAProvider

    # Agy
    agy = AgyProvider()
    assert agy.name == "agy"
    print(f"✅ AgyProvider: {agy.name}, model={agy.model!r}")

    # LLaMA
    llama = LLaMAProvider()
    assert llama.name == "llama"
    print(f"✅ LLaMAProvider: {llama.name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
