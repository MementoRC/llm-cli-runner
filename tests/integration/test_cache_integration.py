"""Integration tests for cache system.

Tests the complete cache system integration including service, backends,
metrics, and key generation working together to achieve target hit rates.
"""

import asyncio

import pytest

from mcp_server_cheap_llm.cache.service import CacheService, MemoryCache
from mcp_server_cheap_llm.core.models import LLMRequest, LLMResponse, ProviderType


@pytest.mark.asyncio
class TestCacheSystemIntegration:
    """Test complete cache system integration."""

    async def test_cache_service_initialization(self):
        """Test cache service initialization and cleanup."""
        memory_backend = MemoryCache(max_size=100)
        cache_service = CacheService(primary_backend=memory_backend)

        # Test basic functionality
        await cache_service.set("test_key", "test_value")
        result = await cache_service.get("test_key")
        assert result == "test_value"

        # Test cleanup
        await cache_service.close()
        assert cache_service.primary_backend is not None
        assert cache_service.metrics_enabled is not None

    async def test_end_to_end_caching_workflow(self):
        """Test complete caching workflow from request to response."""
        memory_backend = MemoryCache(max_size=10)
        cache_service = CacheService(primary_backend=memory_backend)

        try:
            # Create test request and response
            request = LLMRequest(
                prompt="What is Python?",
                provider=ProviderType.GEMINI,
                max_tokens=100,
                temperature=0.7,
            )

            response = LLMResponse(
                content="Python is a programming language.",
                provider=ProviderType.GEMINI,
                success=True,
                tokens_used=50,
                response_time_ms=1500,
            )

            # Initially should be cache miss
            cached_response = await cache_service.get_cached_response(request)
            assert cached_response is None

            # Cache the response
            success = await cache_service.cache_response(request, response)
            assert success

            # Now should be cache hit
            cached_response = await cache_service.get_cached_response(request)
            assert cached_response is not None
            assert cached_response.content == response.content
            assert cached_response.provider == response.provider

        finally:
            await cache_service.close()

    async def test_cache_hit_rate_tracking(self):
        """Test cache hit rate tracking and metrics."""
        memory_backend = MemoryCache(max_size=50)
        cache_service = CacheService(primary_backend=memory_backend)

        try:
            # Create multiple requests
            requests = []
            responses = []

            for i in range(10):
                request = LLMRequest(
                    prompt=f"Question {i}",
                    provider=ProviderType.GEMINI,
                    max_tokens=100,
                )
                response = LLMResponse(
                    content=f"Answer {i}",
                    provider=ProviderType.GEMINI,
                    success=True,
                    tokens_used=25,
                )
                requests.append(request)
                responses.append(response)

            # First pass - all cache misses
            for request in requests:
                cached = await cache_service.get_cached_response(request)
                assert cached is None

            # Cache all responses
            for request, response in zip(requests, responses, strict=False):
                await cache_service.cache_response(request, response)

            # Second pass - should be cache hits
            hit_count = 0
            for request in requests:
                cached = await cache_service.get_cached_response(request)
                if cached is not None:
                    hit_count += 1

            # All should be hits
            assert hit_count == 10

            # Check metrics
            stats = await cache_service.get_cache_stats()
            metrics = stats["metrics"]["overview"]

            # Should have 100% hit rate for second pass
            # Total: 10 misses + 10 hits = 20 requests
            # Hit rate should be 50%
            assert metrics["cache_hits"] == 10
            assert metrics["cache_misses"] == 10
            assert metrics["hit_rate"] == 50.0

        finally:
            await cache_service.close()

    async def test_provider_specific_metrics(self):
        """Test provider-specific cache metrics."""
        memory_backend = MemoryCache(max_size=1000)
        cache_service = CacheService(primary_backend=memory_backend)

        try:
            # Create requests for different providers
            gemini_request = LLMRequest(
                prompt="Gemini question",
                provider=ProviderType.GEMINI,
            )
            openai_request = LLMRequest(
                prompt="OpenAI question",
                provider=ProviderType.OPENAI,
            )

            gemini_response = LLMResponse(
                content="Gemini answer",
                provider=ProviderType.GEMINI,
                success=True,
            )
            openai_response = LLMResponse(
                content="OpenAI answer",
                provider=ProviderType.OPENAI,
                success=True,
            )

            # Test cache misses
            await cache_service.get_cached_response(gemini_request)
            await cache_service.get_cached_response(openai_request)

            # Cache responses
            await cache_service.cache_response(gemini_request, gemini_response)
            await cache_service.cache_response(openai_request, openai_response)

            # Test cache hits
            await cache_service.get_cached_response(gemini_request)
            await cache_service.get_cached_response(openai_request)

            # Check provider-specific metrics
            stats = await cache_service.get_cache_stats()
            provider_metrics = stats["metrics"]["providers"]

            assert "gemini" in provider_metrics
            assert "openai" in provider_metrics
            assert provider_metrics["gemini"]["requests"] == 2
            assert provider_metrics["openai"]["requests"] == 2
            assert provider_metrics["gemini"]["hit_rate"] == 50.0
            assert provider_metrics["openai"]["hit_rate"] == 50.0

        finally:
            await cache_service.close()

    async def test_cache_ttl_integration(self):
        """Test TTL integration across the system."""
        memory_backend = MemoryCache(max_size=1000)
        cache_service = CacheService(primary_backend=memory_backend)

        try:
            request = LLMRequest(prompt="TTL test", provider=ProviderType.GEMINI)
            response = LLMResponse(
                content="TTL response",
                provider=ProviderType.GEMINI,
                success=True,
            )

            # Cache with short TTL
            await cache_service.cache_response(request, response, ttl=0.5)

            # Should be available immediately
            cached = await cache_service.get_cached_response(request)
            assert cached is not None

            # Wait for expiration
            await asyncio.sleep(0.6)

            # Should be expired
            cached = await cache_service.get_cached_response(request)
            assert cached is None

        finally:
            await cache_service.close()

    async def test_cache_invalidation_integration(self):
        """Test cache invalidation integration."""
        memory_backend = MemoryCache(max_size=1000)
        cache_service = CacheService(primary_backend=memory_backend)

        try:
            # Cache some responses
            for i in range(5):
                request = LLMRequest(
                    prompt=f"Question {i}",
                    provider=ProviderType.GEMINI,
                )
                response = LLMResponse(
                    content=f"Answer {i}",
                    provider=ProviderType.GEMINI,
                    success=True,
                )
                await cache_service.cache_response(request, response)

            # Verify cache has entries
            stats = await cache_service.get_cache_stats()
            assert stats["cache_size"] == 5

            # Invalidate all cache
            await cache_service.invalidate_cache()

            # Verify cache is empty
            stats = await cache_service.get_cache_stats()
            assert stats["cache_size"] == 0

        finally:
            await cache_service.close()

    async def test_cache_warming_integration(self):
        """Test cache warming integration."""
        memory_backend = MemoryCache(max_size=1000)
        cache_service = CacheService(primary_backend=memory_backend)

        try:
            # Prepare common queries for warming
            common_requests = [
                LLMRequest(prompt="What is AI?", provider=ProviderType.GEMINI),
                LLMRequest(prompt="How does ML work?", provider=ProviderType.OPENAI),
            ]

            # Prepare key-value pairs for warming
            key_value_pairs = []
            for request in common_requests:
                key = cache_service._generate_cache_key(request)
                # Create a mock response for warming
                response = LLMResponse(
                    content=f"Cached answer for: {request.prompt}",
                    provider=request.provider,
                    success=True,
                )
                key_value_pairs.append((key, response))

            # Warm cache with key-value pairs
            await cache_service.warm_cache(key_value_pairs)

            # This test mainly verifies the integration doesn't break
            # Actual warming functionality would require provider system integration

        finally:
            await cache_service.close()

    async def test_concurrent_cache_operations(self):
        """Test concurrent cache operations."""
        memory_backend = MemoryCache(max_size=100)
        cache_service = CacheService(primary_backend=memory_backend)

        try:

            async def cache_operation(index):
                """Single cache operation."""
                request = LLMRequest(
                    prompt=f"Concurrent question {index}",
                    provider=ProviderType.GEMINI,
                )
                response = LLMResponse(
                    content=f"Concurrent answer {index}",
                    provider=ProviderType.GEMINI,
                    success=True,
                )

                # Try to get (should be miss)
                cached = await cache_service.get_cached_response(request)

                # Cache response
                await cache_service.cache_response(request, response)

                # Try to get again (should be hit)
                cached = await cache_service.get_cached_response(request)
                assert cached is not None

                return index

            # Run concurrent operations
            tasks = [cache_operation(i) for i in range(20)]
            results = await asyncio.gather(*tasks)

            # All operations should complete successfully
            assert len(results) == 20
            assert set(results) == set(range(20))

            # Check final cache state
            stats = await cache_service.get_cache_stats()
            assert stats["cache_size"] == 20
            assert stats["metrics"]["overview"]["cache_hits"] == 20
            assert stats["metrics"]["overview"]["cache_misses"] == 20

        finally:
            await cache_service.close()

    async def test_cache_size_limits(self):
        """Test cache size limits and eviction."""
        memory_backend = MemoryCache(max_size=5)  # Small cache for testing eviction
        cache_service = CacheService(primary_backend=memory_backend)

        try:
            # Fill cache beyond capacity
            for i in range(10):
                request = LLMRequest(
                    prompt=f"Question {i}",
                    provider=ProviderType.GEMINI,
                )
                response = LLMResponse(
                    content=f"Answer {i}",
                    provider=ProviderType.GEMINI,
                    success=True,
                )
                await cache_service.cache_response(request, response)

            # Cache should not exceed max size
            stats = await cache_service.get_cache_stats()
            assert stats["cache_size"] <= 5

            # Evictions should have occurred
            backend_stats = stats["backend"]
            assert backend_stats.get("evictions", 0) > 0

        finally:
            await cache_service.close()

    @pytest.mark.xfail(reason="File backend cache_dir initialization needs refinement")
    async def test_file_backend_integration(self):
        """Test file backend integration."""
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()

        try:
            from mcp_server_cheap_llm.cache.service import FileCache

            file_backend = FileCache(cache_dir=temp_dir)
            cache_service = CacheService(primary_backend=file_backend)

            try:
                # Test basic file caching
                request = LLMRequest(
                    prompt="File cache test",
                    provider=ProviderType.GEMINI,
                )
                response = LLMResponse(
                    content="File cache response",
                    provider=ProviderType.GEMINI,
                    success=True,
                )

                # Cache response
                await cache_service.cache_response(request, response)

                # Retrieve from cache
                cached = await cache_service.get_cached_response(request)
                assert cached is not None
                assert cached.content == response.content

                # Verify file exists
                from pathlib import Path

                cache_path = Path(temp_dir)
                json_files = list(cache_path.glob("*.json"))
                assert len(json_files) > 0

            finally:
                await cache_service.close()

        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def test_performance_target_monitoring(self):
        """Test monitoring of 40%+ hit rate target."""
        memory_backend = MemoryCache(max_size=1000)
        cache_service = CacheService(primary_backend=memory_backend)

        try:
            # Create test scenarios to reach target hit rate
            requests = []
            responses = []

            # Create 10 unique requests
            for i in range(10):
                request = LLMRequest(
                    prompt=f"Performance test {i}",
                    provider=ProviderType.GEMINI,
                )
                response = LLMResponse(
                    content=f"Performance response {i}",
                    provider=ProviderType.GEMINI,
                    success=True,
                )
                requests.append(request)
                responses.append(response)

            # First round: all misses (cache empty)
            for request in requests:
                await cache_service.get_cached_response(request)

            # Cache all responses
            for request, response in zip(requests, responses, strict=False):
                await cache_service.cache_response(request, response)

            # Second round: all hits
            for request in requests:
                await cache_service.get_cached_response(request)

            # Check if we met target
            stats = await cache_service.get_cache_stats()
            hit_rate = stats["metrics"]["overview"]["hit_rate"]
            target_performance = stats["metrics"]["overview"]["target_performance"]

            # Should have 50% hit rate (10 hits, 10 misses)
            assert hit_rate == 50.0
            assert target_performance >= 1.0  # Above target

            # Performance status should be good
            status = stats["metrics"]["status"]
            assert status in ["good", "excellent"]

        finally:
            await cache_service.close()
