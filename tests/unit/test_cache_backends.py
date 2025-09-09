"""Unit tests for cache backends.

Tests memory and file cache backends for functionality, TTL management,
LRU eviction, and performance characteristics.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from mcp_server_cheap_llm.cache.backends import CacheEntry, FileBackend, MemoryBackend


class TestCacheEntry:
    """Test cache entry functionality."""

    def test_cache_entry_creation(self):
        """Test cache entry creation and properties."""
        value = "test_value"
        ttl = 3600

        entry = CacheEntry(value, ttl)

        assert entry.value == value
        assert entry.created_at > 0
        assert entry.expires_at > entry.created_at
        assert entry.access_count == 1
        assert entry.last_accessed == entry.created_at

    def test_cache_entry_no_ttl(self):
        """Test cache entry without TTL."""
        entry = CacheEntry("value")

        assert entry.expires_at is None
        assert not entry.is_expired()
        assert entry.time_to_live() is None

    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        # Entry that expires immediately
        entry = CacheEntry("value", ttl=0.001)  # 1ms TTL

        # Should not be expired immediately
        assert not entry.is_expired()

        # Wait and check expiration
        time.sleep(0.002)
        assert entry.is_expired()

    def test_cache_entry_access_tracking(self):
        """Test access count tracking."""
        entry = CacheEntry("value")

        initial_count = entry.access_count
        initial_time = entry.last_accessed

        time.sleep(0.001)  # Small delay
        entry.mark_accessed()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_time

    def test_time_to_live_calculation(self):
        """Test TTL calculation."""
        entry = CacheEntry("value", ttl=10)

        ttl = entry.time_to_live()
        assert ttl is not None
        assert 9 <= ttl <= 10  # Should be close to 10 seconds

        # Wait and check again
        time.sleep(0.1)
        ttl2 = entry.time_to_live()
        assert ttl2 < ttl


@pytest.mark.asyncio
class TestMemoryBackend:
    """Test memory cache backend functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.backend = MemoryBackend(max_size=10, default_ttl=3600)

    def teardown_method(self):
        """Cleanup after tests."""
        if hasattr(self.backend, "close"):
            self.backend.close()

    async def test_basic_set_get(self):
        """Test basic set and get operations."""
        key = "test_key"
        value = "test_value"

        # Set value
        success = await self.backend.set(key, value)
        assert success

        # Get value
        retrieved = await self.backend.get(key)
        assert retrieved == value

    async def test_get_nonexistent_key(self):
        """Test getting nonexistent key."""
        result = await self.backend.get("nonexistent")
        assert result is None

    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        key = "expiring_key"
        value = "expiring_value"

        # Set with short TTL
        await self.backend.set(key, value, ttl=0.1)  # 100ms

        # Should be available immediately
        retrieved = await self.backend.get(key)
        assert retrieved == value

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Should be expired
        retrieved = await self.backend.get(key)
        assert retrieved is None

    async def test_default_ttl(self):
        """Test default TTL usage."""
        key = "default_ttl_key"
        value = "default_ttl_value"

        # Set without explicit TTL
        await self.backend.set(key, value)

        # Should use default TTL
        retrieved = await self.backend.get(key)
        assert retrieved == value

    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(10):
            await self.backend.set(f"key_{i}", f"value_{i}")

        # Verify all keys are present
        for i in range(10):
            value = await self.backend.get(f"key_{i}")
            assert value == f"value_{i}"

        # Access some keys to make them more recently used
        await self.backend.get("key_5")
        await self.backend.get("key_7")

        # Add new key (should evict oldest unused key)
        await self.backend.set("new_key", "new_value")

        # key_0 should be evicted (oldest and not recently accessed)
        evicted = await self.backend.get("key_0")
        assert evicted is None

        # Recently accessed keys should still be present
        assert await self.backend.get("key_5") == "value_5"
        assert await self.backend.get("key_7") == "value_7"
        assert await self.backend.get("new_key") == "new_value"

    async def test_delete_operation(self):
        """Test key deletion."""
        key = "delete_key"
        value = "delete_value"

        # Set and verify
        await self.backend.set(key, value)
        assert await self.backend.get(key) == value

        # Delete and verify
        deleted = await self.backend.delete(key)
        assert deleted
        assert await self.backend.get(key) is None

        # Delete nonexistent key
        deleted = await self.backend.delete("nonexistent")
        assert not deleted

    async def test_exists_operation(self):
        """Test key existence check."""
        key = "exists_key"
        value = "exists_value"

        # Key shouldn't exist initially
        assert not await self.backend.exists(key)

        # Set key
        await self.backend.set(key, value)
        assert await self.backend.exists(key)

        # Delete key
        await self.backend.delete(key)
        assert not await self.backend.exists(key)

    async def test_clear_operation(self):
        """Test cache clear operation."""
        # Add some keys
        for i in range(5):
            await self.backend.set(f"key_{i}", f"value_{i}")

        # Verify keys exist
        assert await self.backend.size() == 5

        # Clear cache
        cleared = await self.backend.clear()
        assert cleared
        assert await self.backend.size() == 0

        # Verify keys are gone
        for i in range(5):
            assert await self.backend.get(f"key_{i}") is None

    async def test_size_operation(self):
        """Test cache size tracking."""
        # Initially empty
        assert await self.backend.size() == 0

        # Add keys
        for i in range(3):
            await self.backend.set(f"key_{i}", f"value_{i}")
            assert await self.backend.size() == i + 1

        # Delete key
        await self.backend.delete("key_1")
        assert await self.backend.size() == 2

    async def test_keys_operation(self):
        """Test getting all keys."""
        # Initially empty
        keys = await self.backend.keys()
        assert keys == []

        # Add keys
        test_keys = ["key1", "key2", "key3"]
        for key in test_keys:
            await self.backend.set(key, f"value_{key}")

        # Get all keys
        all_keys = await self.backend.keys()
        assert len(all_keys) == 3
        assert set(all_keys) == set(test_keys)

    async def test_stats_operation(self):
        """Test cache statistics."""
        # Initial stats
        stats = await self.backend.stats()
        assert stats["backend_type"] == "memory"
        assert stats["size"] == 0
        assert stats["max_size"] == 10
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Add some data and access patterns
        await self.backend.set("key1", "value1")
        await self.backend.get("key1")  # Hit
        await self.backend.get("nonexistent")  # Miss

        # Check updated stats
        stats = await self.backend.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0  # 1 hit out of 2 requests

    async def test_overwrite_existing_key(self):
        """Test overwriting existing key."""
        key = "overwrite_key"

        # Set initial value
        await self.backend.set(key, "value1")
        assert await self.backend.get(key) == "value1"

        # Overwrite with new value
        await self.backend.set(key, "value2")
        assert await self.backend.get(key) == "value2"

        # Size should remain 1
        assert await self.backend.size() == 1


@pytest.mark.asyncio
class TestFileBackend:
    """Test file cache backend functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = FileBackend(
            cache_dir=self.temp_dir,
            default_ttl=3600,
            max_files=100,
        )

    def teardown_method(self):
        """Cleanup after tests."""
        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_basic_file_operations(self):
        """Test basic file cache operations."""
        key = "test_key"
        value = {"data": "test_value", "number": 42}

        # Set value
        success = await self.backend.set(key, value)
        assert success

        # Get value
        retrieved = await self.backend.get(key)
        assert retrieved == value

        # Verify file was created
        cache_path = Path(self.temp_dir)
        assert any(cache_path.glob("*.json"))

    async def test_file_ttl_expiration(self):
        """Test file cache TTL expiration."""
        key = "expiring_key"
        value = "expiring_value"

        # Set with short TTL
        await self.backend.set(key, value, ttl=0.1)  # 100ms

        # Should be available immediately
        retrieved = await self.backend.get(key)
        assert retrieved == value

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Should be expired and file should be removed
        retrieved = await self.backend.get(key)
        assert retrieved is None

    async def test_file_persistence(self):
        """Test file cache persistence across instances."""
        key = "persistent_key"
        value = "persistent_value"

        # Set value in first backend instance
        await self.backend.set(key, value)

        # Create new backend instance with same directory
        new_backend = FileBackend(cache_dir=self.temp_dir)

        # Should be able to retrieve value
        retrieved = await new_backend.get(key)
        assert retrieved == value

    async def test_file_nonexistent_key(self):
        """Test getting nonexistent key from file cache."""
        result = await self.backend.get("nonexistent")
        assert result is None

    async def test_file_delete_operation(self):
        """Test file deletion."""
        key = "delete_key"
        value = "delete_value"

        # Set and verify
        await self.backend.set(key, value)
        assert await self.backend.get(key) == value

        # Delete and verify
        deleted = await self.backend.delete(key)
        assert deleted
        assert await self.backend.get(key) is None

    async def test_file_clear_operation(self):
        """Test clearing all files."""
        # Add some keys
        for i in range(3):
            await self.backend.set(f"key_{i}", f"value_{i}")

        # Verify files exist
        cache_path = Path(self.temp_dir)
        json_files = list(cache_path.glob("*.json"))
        assert len(json_files) == 3

        # Clear cache
        cleared = await self.backend.clear()
        assert cleared

        # Verify files are gone
        json_files = list(cache_path.glob("*.json"))
        assert len(json_files) == 0

    async def test_file_size_operation(self):
        """Test file cache size tracking."""
        # Initially empty
        assert await self.backend.size() == 0

        # Add keys
        for i in range(3):
            await self.backend.set(f"key_{i}", f"value_{i}")
            assert await self.backend.size() == i + 1

    async def test_file_keys_operation(self):
        """Test getting all file cache keys."""
        # Add keys
        test_keys = ["key1", "key2", "key3"]
        for key in test_keys:
            await self.backend.set(key, f"value_{key}")

        # Get all keys
        all_keys = await self.backend.keys()
        assert len(all_keys) == 3
        assert set(all_keys) == set(test_keys)

    async def test_file_stats_operation(self):
        """Test file cache statistics."""
        stats = await self.backend.stats()
        assert stats["backend_type"] == "file"
        assert stats["cache_dir"] == self.temp_dir
        assert "size" in stats
        assert "hit_rate" in stats

    async def test_file_special_characters(self):
        """Test handling of special characters in keys."""
        # Keys with special characters
        special_keys = [
            "key:with:colons",
            "key/with/slashes",
            "key with spaces",
            "key@with#symbols",
        ]

        for key in special_keys:
            value = f"value_{key}"
            await self.backend.set(key, value)
            retrieved = await self.backend.get(key)
            assert retrieved == value

    async def test_file_complex_data(self):
        """Test storing complex data structures."""
        complex_data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"inner": "value"},
        }

        await self.backend.set("complex", complex_data)
        retrieved = await self.backend.get("complex")
        assert retrieved == complex_data
