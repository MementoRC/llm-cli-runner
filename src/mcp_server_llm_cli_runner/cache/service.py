"""Cache service and backend implementation.

This module provides caching functionality for the llm-cli-runner server,
including various backend implementations and management interfaces.
"""

import inspect
import json
import time
from abc import ABC, abstractmethod
from typing import Any

from mcp_server_llm_cli_runner.utils.logging import get_logger


class CacheBackend(ABC):
    """Abstract cache backend interface.

    This abstract base class defines the interface that all cache backends
    must implement. It provides the basic operations for storing, retrieving,
    and managing cached data.
    """

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Retrieve a value from the cache.

        Args:
            key: The cache key to retrieve

        Returns:
            The cached value or None if not found

        """

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store a value in the cache.

        Args:
            key: The cache key
            value: The value to store
            ttl: Time to live in seconds (optional)

        Returns:
            True if successful, False otherwise

        """

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.

        Args:
            key: The cache key to delete

        Returns:
            True if deleted, False if not found

        """

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""

    @abstractmethod
    async def close(self) -> None:
        """Close cache backend connection."""


class MemoryCache(CacheBackend):
    """In-memory cache backend implementation."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize memory cache.

        Args:
            max_size: Maximum number of items to store

        """
        self._cache: dict[str, tuple[Any, float | None]] = {}
        self._max_size = max_size
        self._evictions = 0  # Track evictions for stats
        self.logger = get_logger(__name__)

    async def get(self, key: str) -> Any | None:
        """Retrieve value from memory cache."""
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]

        # Check expiry
        if expiry is not None and time.time() > expiry:
            del self._cache[key]
            return None

        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store value in memory cache."""
        try:
            # Calculate expiry time
            expiry = time.time() + ttl if ttl else None

            # Evict oldest items if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Remove oldest item (first in dict)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1

            self._cache[key] = (value, expiry)
            return True
        except Exception as e:
            self.logger.exception(f"Failed to set cache key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        if key not in self._cache:
            return False

        # Check if expired
        _, expiry = self._cache[key]
        if expiry is not None and time.time() > expiry:
            del self._cache[key]
            return False

        return True

    async def close(self) -> None:
        """Close memory cache (cleanup)."""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "evictions": self._evictions,
            "keys": list(self._cache.keys()),
        }


class CacheService:
    """Main cache service that manages cache backends and provides high-level operations."""

    def __init__(
        self,
        backend: CacheBackend | None = None,
        primary_backend: CacheBackend | None = None,  # For backward compatibility
        default_ttl: int = 3600,
        metrics_enabled: bool = True,
    ) -> None:
        """Initialize cache service.

        Args:
            backend: Cache backend to use (defaults to MemoryCache)
            primary_backend: Alias for backend (for backward compatibility)
            default_ttl: Default time-to-live in seconds
            metrics_enabled: Whether to track metrics

        """
        # Support both parameter names for backward compatibility
        actual_backend = backend or primary_backend or MemoryCache()
        self._backend = actual_backend
        self._default_ttl = default_ttl
        self.metrics_enabled = metrics_enabled
        self.logger = get_logger(__name__)

        # Initialize metrics
        self._metrics: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

        # Provider-specific metrics
        self._provider_metrics: dict[str, dict[str, int]] = {}

    @property
    def primary_backend(self) -> CacheBackend:
        """Get the primary backend for backward compatibility.

        Returns:
            The primary cache backend

        """
        return self._backend

    async def get(self, key: str) -> Any | None:
        """Get value from cache with metrics tracking."""
        try:
            value = await self._backend.get(key)

            if self.metrics_enabled:
                if value is not None:
                    self._metrics["hits"] += 1
                else:
                    self._metrics["misses"] += 1

            return value

        except Exception as e:
            if self.metrics_enabled:
                self._metrics["errors"] += 1
            self.logger.exception(f"Cache get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with metrics tracking."""
        try:
            actual_ttl = ttl if ttl is not None else self._default_ttl
            success = await self._backend.set(key, value, actual_ttl)

            if self.metrics_enabled and success:
                self._metrics["sets"] += 1

            return success

        except Exception as e:
            if self.metrics_enabled:
                self._metrics["errors"] += 1
            self.logger.exception(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache with metrics tracking."""
        try:
            success = await self._backend.delete(key)

            if self.metrics_enabled and success:
                self._metrics["deletes"] += 1

            return success

        except Exception as e:
            if self.metrics_enabled:
                self._metrics["errors"] += 1
            self.logger.exception(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return await self._backend.exists(key)
        except Exception as e:
            self.logger.exception(f"Cache exists check error for key {key}: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries (if supported by backend)."""
        try:
            # For memory cache, we can implement clear
            if hasattr(self._backend, "_cache"):
                self._backend._cache.clear()  # type: ignore[attr-defined]
                return True
            self.logger.warning("Clear operation not supported by current backend")
            return False
        except Exception as e:
            self.logger.exception(f"Cache clear error: {e}")
            return False

    def generate_key(self, *components: str) -> str:
        """Generate a cache key from components.

        Args:
            *components: Key components to join

        Returns:
            Generated cache key

        """
        return ":".join(str(c) for c in components)

    def _generate_cache_key(self, request_data: Any) -> str:
        """Generate cache key for request data.

        This method is used by the cache warming functionality.

        Args:
            request_data: Request data (dictionary or Pydantic model)

        Returns:
            Generated cache key

        """
        return self._generate_request_key(request_data)

    def _get_provider_from_request(self, request_data: Any) -> str | None:
        """Extract provider information from request data.

        Args:
            request_data: Request data (dictionary or Pydantic model)

        Returns:
            Provider name as string, or None if not found

        """
        try:
            # Handle Pydantic models
            if hasattr(request_data, "provider"):
                provider = request_data.provider
                if hasattr(provider, "value"):  # Enum
                    return provider.value.lower()
                return str(provider).lower()

            # Handle dictionaries
            if isinstance(request_data, dict) and "provider" in request_data:
                provider = request_data["provider"]
                if hasattr(provider, "value"):  # Enum
                    return provider.value.lower()
                return str(provider).lower()

            return None
        except Exception:
            return None

    def _track_provider_metric(self, provider: str | None, metric_type: str) -> None:
        """Track provider-specific metrics.

        Args:
            provider: Provider name
            metric_type: Type of metric ('hit', 'miss', 'set', etc.)

        """
        if not self.metrics_enabled or not provider:
            return

        if provider not in self._provider_metrics:
            self._provider_metrics[provider] = {
                "hits": 0,
                "misses": 0,
                "requests": 0,
            }

        # Track the specific metric
        if metric_type == "hit":
            self._provider_metrics[provider]["hits"] += 1
            self._provider_metrics[provider]["requests"] += 1
        elif metric_type == "miss":
            self._provider_metrics[provider]["misses"] += 1
            self._provider_metrics[provider]["requests"] += 1

    def _generate_request_key(self, request_data: Any) -> str:
        """Generate cache key for request data.

        Args:
            request_data: Request data (dictionary or Pydantic model)

        Returns:
            Generated cache key

        """
        # Convert Pydantic models to dictionaries
        if hasattr(request_data, "model_dump"):
            # Pydantic v2
            cache_data = request_data.model_dump()
        elif hasattr(request_data, "dict"):
            # Pydantic v1
            cache_data = request_data.dict()
        elif isinstance(request_data, dict):
            # Already a dictionary
            cache_data = request_data.copy()
        else:
            # Convert to string representation for other types
            cache_data = {"data": str(request_data)}

        # Remove non-deterministic fields like timestamps
        if isinstance(cache_data, dict):
            cache_data = {
                k: v
                for k, v in cache_data.items()
                if k not in ("timestamp", "request_id")
            }

        # Sort keys for consistency
        key_str = json.dumps(cache_data, sort_keys=True, default=str)

        # Create shorter hash-based key
        import hashlib

        hash_obj = hashlib.md5(key_str.encode(), usedforsecurity=False)
        return f"req:{hash_obj.hexdigest()}"

    async def get_cached_response(self, request_data: Any) -> Any | None:
        """Get cached response for request.

        Args:
            request_data: Request data (dictionary or Pydantic model)

        Returns:
            Cached response or None

        """
        key = self._generate_request_key(request_data)
        provider = self._get_provider_from_request(request_data)

        result = await self.get(key)

        # Track provider-specific metrics
        if result is not None:
            self._track_provider_metric(provider, "hit")
        else:
            self._track_provider_metric(provider, "miss")

        return result

    async def cache_response(
        self,
        request_data: Any,
        response_data: Any,
        ttl: int | None = None,
    ) -> bool:
        """Cache response for request.

        Args:
            request_data: Request data (dictionary or Pydantic model)
            response_data: Response data to cache
            ttl: Time to live in seconds

        Returns:
            True if successful

        """
        key = self._generate_request_key(request_data)
        return await self.set(key, response_data, ttl)

    async def get_or_set(
        self,
        key: str,
        value_factory: Any,  # Callable[[], Awaitable[Any]] | Callable[[], Any]
        ttl: int | None = None,
    ) -> Any:
        """Get value from cache or set it using factory function.

        Args:
            key: Cache key
            value_factory: Function to generate value if not cached
            ttl: Time to live in seconds

        Returns:
            Cached or generated value

        """
        # Try to get from cache first
        value = await self.get(key)
        if value is not None:
            return value

        # Generate new value
        try:
            if callable(value_factory):
                new_value = value_factory()
                # Handle async factories
                if inspect.iscoroutine(new_value):
                    new_value = await new_value
            else:
                new_value = value_factory

            # Store in cache
            await self.set(key, new_value, ttl)
            return new_value

        except Exception as e:
            self.logger.exception(f"Error in get_or_set for key {key}: {e}")
            return None

    async def invalidate_cache(self) -> bool:
        """Invalidate (clear) all cache entries.

        Returns:
            True if successful

        """
        return await self.clear()

    async def warm_cache(self, key_value_pairs: list[tuple[str, Any]]) -> bool:
        """Warm cache with pre-computed key-value pairs.

        Args:
            key_value_pairs: List of (key, value) tuples to cache

        Returns:
            True if all entries were cached successfully

        """
        try:
            success_count = 0
            for key, value in key_value_pairs:
                if await self.set(key, value):
                    success_count += 1

            return success_count == len(key_value_pairs)
        except Exception as e:
            self.logger.exception(f"Error warming cache: {e}")
            return False

    async def close(self) -> None:
        """Close cache service and backend."""
        try:
            await self._backend.close()
        except Exception as e:
            self.logger.exception(f"Error closing cache backend: {e}")

    def get_metrics(self) -> dict[str, int | float]:
        """Get cache metrics.

        Returns:
            Dictionary of cache metrics

        """
        if not self.metrics_enabled:
            return {}

        # Create a new dictionary with mixed types
        result: dict[str, int | float] = {}
        for key, value in self._metrics.items():
            result[key] = value

        # Add calculated hit rate
        if result["hits"] + result["misses"] > 0:
            result["hit_rate"] = float(result["hits"]) / (
                result["hits"] + result["misses"]
            )
        else:
            result["hit_rate"] = 0.0

        return result

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics for testing.

        Returns:
            Dictionary with cache statistics in expected format

        """
        metrics = self.get_metrics()
        backend_stats = self.get_backend_stats()

        # Convert hit rate to percentage for test compatibility
        hit_rate_percentage = metrics.get("hit_rate", 0.0) * 100

        # Calculate provider-specific metrics with hit rates
        provider_stats = {}
        for provider, stats in self._provider_metrics.items():
            hits = stats.get("hits", 0)
            misses = stats.get("misses", 0)
            requests = stats.get("requests", 0)

            # Calculate hit rate as percentage
            hit_rate = (hits / requests * 100) if requests > 0 else 0.0

            provider_stats[provider] = {
                "hits": hits,
                "misses": misses,
                "requests": requests,
                "hit_rate": hit_rate,
            }

        # Calculate target performance (hit rate vs 40% target)
        target_performance = (
            hit_rate_percentage / 40.0 if hit_rate_percentage > 0 else 0.0
        )

        # Determine performance status
        if hit_rate_percentage >= 80:
            status = "excellent"
        elif hit_rate_percentage >= 40:
            status = "good"
        else:
            status = "poor"

        return {
            "cache_size": backend_stats.get("size", 0),
            "metrics": {
                "overview": {
                    "cache_hits": metrics.get("hits", 0),
                    "cache_misses": metrics.get("misses", 0),
                    "cache_sets": metrics.get("sets", 0),
                    "cache_deletes": metrics.get("deletes", 0),
                    "cache_errors": metrics.get("errors", 0),
                    "hit_rate": hit_rate_percentage,
                    "target_performance": target_performance,
                },
                "providers": provider_stats,
                "status": status,
            },
            "backend": backend_stats,
            "configuration": {
                "default_ttl": self._default_ttl,
                "metrics_enabled": self.metrics_enabled,
                "backend_type": type(self._backend).__name__,
            },
        }

    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        if self.metrics_enabled:
            for key in self._metrics:
                self._metrics[key] = 0
            self._provider_metrics.clear()

    def get_backend_stats(self) -> dict[str, Any]:
        """Get backend-specific statistics."""
        try:
            if hasattr(self._backend, "get_stats"):
                return self._backend.get_stats()  # type: ignore[attr-defined]
            return {}
        except Exception as e:
            self.logger.exception(f"Error getting backend stats: {e}")
            return {}


# Cache service instance for global use
_cache_service: CacheService | None = None


def get_cache_service() -> CacheService:
    """Get the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


async def cached_response(
    key: str,
    response_factory: Any,  # Callable[[], Awaitable[Any]] | Callable[[], Any]
    ttl: int | None = None,
) -> Any:
    """Decorator-style caching function.

    Args:
        key: Cache key
        response_factory: Function to generate response
        ttl: Time to live in seconds

    Returns:
        Cached or generated response

    """
    cache = get_cache_service()
    return await cache.get_or_set(key, response_factory, ttl)
