"""Comprehensive caching system for MCP Server Cheap LLM.

This module provides a multi-backend caching system with SHA256-based content
addressing, TTL management, and LRU eviction policies. Designed to achieve
40%+ cache hit rates for cost optimization.

Key components:
    CacheService: Main caching interface
    CacheBackend: Abstract backend interface
    MemoryBackend: In-memory LRU cache with TTL
    RedisBackend: Redis-based distributed cache
    FileBackend: File-based persistent cache
    CacheMetrics: Hit rate monitoring and statistics

Example:
    >>> from mcp_server_cheap_llm.cache import CacheService
    >>> cache = CacheService()
    >>> await cache.get("cache_key")
    >>> await cache.set("cache_key", response_data, ttl=3600)

"""

from .backends import CacheBackend, MemoryBackend
from .key_generator import CacheKeyGenerator
from .metrics import CacheMetrics
from .service import CacheService

__all__ = [
    "CacheBackend",
    "CacheKeyGenerator",
    "CacheMetrics",
    "CacheService",
    "MemoryBackend",
]
