"""Cache backend implementations for different storage systems.

This module provides abstract base class for cache backends and concrete
implementations for memory, Redis, and file-based storage. All backends
support TTL management, LRU eviction, and consistent interfaces.

Supported backends:
    MemoryBackend: In-memory LRU cache with TTL
    RedisBackend: Redis-based distributed cache (planned)
    FileBackend: File-based persistent cache (planned)

Example:
    >>> backend = MemoryBackend(max_size=1000, default_ttl=3600)
    >>> await backend.set("key", "value", ttl=1800)
    >>> value = await backend.get("key")

"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any


class CacheEntry:
    """Cache entry with TTL and access tracking.

    Represents a single cached item with expiration time and
    access tracking for LRU eviction policies.

    Attributes:
        value: Cached data
        created_at: Entry creation timestamp
        expires_at: Entry expiration timestamp
        access_count: Number of times entry was accessed
        last_accessed: Last access timestamp

    """

    def __init__(self, value: Any, ttl: int | None = None) -> None:
        """Initialize cache entry.

        Args:
            value: Data to cache
            ttl: Time-to-live in seconds (None for no expiration)

        """
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl if ttl else None
        self.access_count = 1
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if cache entry has expired.

        Returns:
            True if entry is expired, False otherwise

        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def mark_accessed(self) -> None:
        """Mark entry as accessed for LRU tracking."""
        self.access_count += 1
        self.last_accessed = time.time()

    def time_to_live(self) -> float | None:
        """Get remaining time-to-live in seconds.

        Returns:
            Remaining TTL in seconds or None if no expiration

        """
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0, remaining)


class CacheBackend(ABC):
    """Abstract base class for cache backends.

    Defines the interface that all cache backends must implement.
    Supports async operations, TTL management, and LRU eviction.
    """

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Retrieve value from cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value or None if not found/expired

        """

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store value in cache.

        Args:
            key: Cache key to store under
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if stored successfully, False otherwise

        """

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove value from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if removed, False if key didn't exist

        """

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cached values.

        Returns:
            True if cleared successfully

        """

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and not expired

        """

    @abstractmethod
    async def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of cached items

        """

    @abstractmethod
    async def keys(self) -> list[str]:
        """Get all cache keys.

        Returns:
            List of all cache keys

        """

    @abstractmethod
    async def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics

        """

    @abstractmethod
    async def _cleanup_expired(self):
        """Clean up expired cache entries.

        Returns:
            None

        """

    @abstractmethod
    def close(self):
        """Close backend and cleanup resources.

        Returns:
            None

        """


class MemoryBackend(CacheBackend):
    """In-memory cache backend with LRU eviction and TTL.

    Thread-safe in-memory cache using OrderedDict for LRU tracking.
    Supports TTL expiration and automatic cleanup of expired entries.

    Attributes:
        max_size: Maximum number of cache entries
        default_ttl: Default TTL for new entries
        cleanup_interval: Interval for expired entry cleanup

    Example:
        >>> backend = MemoryBackend(max_size=1000, default_ttl=3600)
        >>> await backend.set("user:123", user_data, ttl=1800)
        >>> user = await backend.get("user:123")

    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int | None = 3600,
        cleanup_interval: int = 300,
    ) -> None:
        """Initialize memory cache backend.

        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds (None for no expiration)
            cleanup_interval: Cleanup interval in seconds

        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        # Thread-safe cache storage using OrderedDict for LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

        # Statistics tracking
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

        # Background cleanup task (created when first used)
        self._cleanup_task = None

    def _ensure_cleanup_task(self) -> None:
        """Ensure cleanup task is running."""
        if self._cleanup_task is None:
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_expired())
            except RuntimeError:
                # No event loop running, task will be created later
                pass

    async def get(self, key: str) -> Any | None:
        """Retrieve value from memory cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value or None if not found/expired

        """
        self._ensure_cleanup_task()
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                self._expirations += 1
                self._misses += 1
                return None

            # Mark as accessed and move to end (most recently used)
            entry.mark_accessed()
            self._cache.move_to_end(key)
            self._hits += 1

            return entry.value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store value in memory cache.

        Args:
            key: Cache key to store under
            value: Value to cache
            ttl: Time-to-live in seconds (uses default_ttl if None)

        Returns:
            True if stored successfully

        """
        async with self._lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl

            # Create cache entry
            entry = CacheEntry(value, ttl)

            # Check if we need to evict entries
            if key not in self._cache and len(self._cache) >= self.max_size:
                await self._evict_lru()

            # Store entry (will overwrite if key exists)
            self._cache[key] = entry
            self._cache.move_to_end(key)  # Mark as most recently used

            return True

    async def delete(self, key: str) -> bool:
        """Remove value from memory cache.

        Args:
            key: Cache key to remove

        Returns:
            True if removed, False if key didn't exist

        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> bool:
        """Clear all cached values.

        Returns:
            True if cleared successfully

        """
        async with self._lock:
            self._cache.clear()
            return True

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and not expired

        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False

            if entry.is_expired():
                del self._cache[key]
                self._expirations += 1
                return False

            return True

    async def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of cached items

        """
        async with self._lock:
            return len(self._cache)

    async def keys(self) -> list[str]:
        """Get all cache keys.

        Returns:
            List of all cache keys

        """
        async with self._lock:
            return list(self._cache.keys())

    async def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics

        """
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "backend_type": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": round(hit_rate, 2),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "total_requests": total_requests,
            }

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            # Remove oldest entry (least recently used)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._evictions += 1

    async def _cleanup_expired(self) -> None:
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                async with self._lock:
                    expired_keys = []
                    current_time = time.time()

                    for key, entry in self._cache.items():
                        if entry.expires_at and current_time > entry.expires_at:
                            expired_keys.append(key)

                    for key in expired_keys:
                        del self._cache[key]
                        self._expirations += 1

            except asyncio.CancelledError:
                break
            except Exception:
                # Continue cleanup on errors
                continue

    def close(self) -> None:
        """Close backend and cleanup resources."""
        if hasattr(self, "_cleanup_task") and self._cleanup_task:
            try:
                self._cleanup_task.cancel()
            except RuntimeError:
                # Event loop may be closed
                pass


class FileBackend(CacheBackend):
    """File-based cache backend for persistence.

    Stores cache entries as JSON files in a directory structure.
    Supports TTL and provides persistence across restarts.

    Note: This is a simplified implementation. For production use,
    consider more sophisticated file management and indexing.
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        default_ttl: int | None = 3600,
        max_files: int = 10000,
    ) -> None:
        """Initialize file cache backend.

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default TTL in seconds
            max_files: Maximum number of cache files

        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.max_files = max_files

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Any | None:
        """Retrieve value from file cache."""
        try:
            file_path = self._get_file_path(key)

            if not file_path.exists():
                self._misses += 1
                return None

            # Read cache entry
            with open(file_path) as f:
                data = json.load(f)

            # Check expiration
            if data.get("expires_at") and time.time() > data["expires_at"]:
                # Remove expired file
                file_path.unlink(missing_ok=True)
                self._misses += 1
                return None

            self._hits += 1
            return data["value"]

        except Exception:
            self._misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store value in file cache."""
        try:
            file_path = self._get_file_path(key)

            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl

            # Prepare cache entry
            entry_data = {
                "value": value,
                "created_at": time.time(),
                "expires_at": time.time() + ttl if ttl else None,
            }

            # Write to file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(entry_data, f)

            return True

        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Remove value from file cache."""
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False

    async def clear(self) -> bool:
        """Clear all cached files."""
        try:
            for file_path in self.cache_dir.rglob("*.json"):
                file_path.unlink()
            return True
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in file cache."""
        value = await self.get(key)
        return value is not None

    async def size(self) -> int:
        """Get current cache size."""
        try:
            return len(list(self.cache_dir.rglob("*.json")))
        except Exception:
            return 0

    async def keys(self) -> list[str]:
        """Get all cache keys."""
        try:
            keys = []
            for file_path in self.cache_dir.rglob("*.json"):
                # Extract key from file path
                relative_path = file_path.relative_to(self.cache_dir)
                key = str(relative_path.with_suffix(""))
                keys.append(key)
            return keys
        except Exception:
            return []

    async def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "backend_type": "file",
            "size": await self.size(),
            "max_files": self.max_files,
            "hit_rate": round(hit_rate, 2),
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "cache_dir": str(self.cache_dir),
        }

    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Create safe filename from key
        safe_key = key.replace(":", "_").replace("/", "_")
        return self.cache_dir / f"{safe_key}.json"

    async def _cleanup_expired(self) -> None:
        """Clean up expired file cache entries."""
        try:
            current_time = time.time()
            for file_path in self.cache_dir.rglob("*.json"):
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    if data.get("expires_at") and current_time > data["expires_at"]:
                        file_path.unlink(missing_ok=True)
                except Exception:
                    # Skip files that cant be read
                    continue
        except Exception:
            # Continue on errors
            pass

    def close(self) -> None:
        """Close file backend (no resources to cleanup)."""
