"""Cache metrics and monitoring system.

This module provides comprehensive cache performance monitoring including
hit rates, response times, and usage statistics. Designed to achieve and
track the 40%+ cache hit rate target.

Key components:
    CacheMetrics: Main metrics collection and analysis
    PerformanceTracker: Response time and latency tracking
    HitRateMonitor: Real-time hit rate monitoring
    UsageAnalyzer: Cache usage pattern analysis

Example:
    >>> metrics = CacheMetrics()
    >>> metrics.record_hit("cache_key", 1.5)  # 1.5ms cache response
    >>> metrics.record_miss("cache_key", 150.0)  # 150ms provider response
    >>> hit_rate = metrics.get_hit_rate()

"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from mcp_server_cheap_llm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricSnapshot:
    """Point-in-time cache metrics snapshot.

    Attributes:
        timestamp: When snapshot was taken
        hit_rate: Cache hit rate percentage
        total_requests: Total cache requests
        hits: Number of cache hits
        misses: Number of cache misses
        avg_hit_time_ms: Average cache hit response time
        avg_miss_time_ms: Average cache miss response time
        cache_size: Number of cached items
        evictions: Number of cache evictions

    """

    timestamp: datetime
    hit_rate: float
    total_requests: int
    hits: int
    misses: int
    avg_hit_time_ms: float
    avg_miss_time_ms: float
    cache_size: int
    evictions: int


@dataclass
class ProviderMetrics:
    """Provider-specific cache metrics.

    Tracks cache performance per provider to identify
    which providers benefit most from caching.

    Attributes:
        provider_name: Name of the LLM provider
        requests: Total requests to this provider
        cache_hits: Requests served from cache
        cache_misses: Requests that hit the provider
        hit_rate: Cache hit rate for this provider
        avg_response_time_ms: Average response time
        cost_saved_usd: Estimated cost savings from caching

    """

    provider_name: str
    requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    cost_saved_usd: float = 0.0


class CacheMetrics:
    """Comprehensive cache performance metrics system.

    Tracks cache hit rates, response times, provider-specific metrics,
    and provides analysis to optimize cache performance. Monitors progress
    toward the 40%+ hit rate target.

    Attributes:
        target_hit_rate: Target cache hit rate percentage (default: 40.0)
        window_size: Size of rolling window for recent metrics

    Example:
        >>> metrics = CacheMetrics(target_hit_rate=45.0)
        >>> metrics.record_hit("gemini", "sha256:abc...", 2.1)
        >>> metrics.record_miss("openai", "sha256:def...", 180.5)
        >>> print(f"Hit rate: {metrics.get_hit_rate():.1f}%")

    """

    def __init__(self, target_hit_rate: float = 40.0, window_size: int = 1000) -> None:
        """Initialize cache metrics system.

        Args:
            target_hit_rate: Target hit rate percentage
            window_size: Size of rolling window for metrics

        """
        self.target_hit_rate = target_hit_rate
        self.window_size = window_size

        # Overall metrics
        self._total_hits = 0
        self._total_misses = 0
        self._total_evictions = 0

        # Rolling window for recent performance
        self._recent_hits: deque = deque(maxlen=window_size)
        self._recent_misses: deque = deque(maxlen=window_size)

        # Response time tracking
        self._hit_times: deque = deque(maxlen=window_size)
        self._miss_times: deque = deque(maxlen=window_size)

        # Provider-specific metrics
        self._provider_metrics: dict[str, ProviderMetrics] = {}

        # Key-specific tracking
        self._key_access_counts: dict = defaultdict(int)
        self._key_hit_counts: dict = defaultdict(int)

        # Time-based metrics
        self._hourly_stats: dict = defaultdict(lambda: {"hits": 0, "misses": 0})

        # Performance tracking
        self._start_time = time.time()
        self._snapshots: list[MetricSnapshot] = []

        logger.info(
            f"Cache metrics initialized with {target_hit_rate}% target hit rate",
        )

    def record_hit(
        self,
        provider: str,
        cache_key: str,
        response_time_ms: float,
        cost_saved: float = 0.0,
    ) -> None:
        """Record a cache hit.

        Args:
            provider: Provider name (gemini, openai, etc.)
            cache_key: Cache key that was hit
            response_time_ms: Response time from cache
            cost_saved: Estimated cost savings from cache hit

        """
        current_time = time.time()

        # Update overall metrics
        self._total_hits += 1
        self._recent_hits.append(current_time)
        self._hit_times.append(response_time_ms)

        # Update provider metrics
        if provider not in self._provider_metrics:
            self._provider_metrics[provider] = ProviderMetrics(provider)

        provider_metrics = self._provider_metrics[provider]
        provider_metrics.requests += 1
        provider_metrics.cache_hits += 1
        provider_metrics.cost_saved_usd += cost_saved
        provider_metrics.hit_rate = (
            provider_metrics.cache_hits / provider_metrics.requests * 100
        )

        # Update key-specific metrics
        self._key_access_counts[cache_key] += 1
        self._key_hit_counts[cache_key] += 1

        # Update hourly stats
        hour_key = datetime.now().strftime("%Y-%m-%d:%H")
        self._hourly_stats[hour_key]["hits"] += 1

        # Log significant cache hits
        if response_time_ms < 10.0:  # Very fast cache hit
            logger.debug(f"Fast cache hit for {provider}: {response_time_ms:.2f}ms")

    def record_miss(
        self,
        provider: str,
        cache_key: str,
        response_time_ms: float,
        cache_size: int | None = None,
    ) -> None:
        """Record a cache miss.

        Args:
            provider: Provider name
            cache_key: Cache key that was missed
            response_time_ms: Response time from provider
            cache_size: Current cache size

        """
        current_time = time.time()

        # Update overall metrics
        self._total_misses += 1
        self._recent_misses.append(current_time)
        self._miss_times.append(response_time_ms)

        # Update provider metrics
        if provider not in self._provider_metrics:
            self._provider_metrics[provider] = ProviderMetrics(provider)

        provider_metrics = self._provider_metrics[provider]
        provider_metrics.requests += 1
        provider_metrics.cache_misses += 1
        provider_metrics.hit_rate = (
            provider_metrics.cache_hits / provider_metrics.requests * 100
        )

        # Update key-specific metrics
        self._key_access_counts[cache_key] += 1

        # Update hourly stats
        hour_key = datetime.now().strftime("%Y-%m-%d:%H")
        self._hourly_stats[hour_key]["misses"] += 1

        # Check if we're below target hit rate
        current_hit_rate = self.get_hit_rate()
        if current_hit_rate < self.target_hit_rate:
            logger.warning(
                f"Cache hit rate ({current_hit_rate:.1f}%) below target "
                f"({self.target_hit_rate:.1f}%)",
            )

    def record_eviction(self, evicted_keys: list[str]) -> None:
        """Record cache evictions.

        Args:
            evicted_keys: List of cache keys that were evicted

        """
        self._total_evictions += len(evicted_keys)

        logger.debug(f"Cache evictions: {len(evicted_keys)} keys evicted")

        # Log frequently accessed keys being evicted
        for key in evicted_keys:
            access_count = self._key_access_counts.get(key, 0)
            if access_count > 10:  # Frequently accessed key
                logger.warning(
                    f"Evicting frequently accessed key: {key} "
                    f"(accessed {access_count} times)",
                )

    def get_hit_rate(self, window_minutes: int | None = None) -> float:
        """Calculate cache hit rate.

        Args:
            window_minutes: Calculate hit rate for recent window (None for all-time)

        Returns:
            Hit rate as percentage (0.0 to 100.0)

        """
        if window_minutes:
            # Calculate hit rate for recent window
            cutoff_time = time.time() - (window_minutes * 60)
            recent_hits = sum(1 for t in self._recent_hits if t >= cutoff_time)
            recent_misses = sum(1 for t in self._recent_misses if t >= cutoff_time)
            total_recent = recent_hits + recent_misses

            if total_recent == 0:
                return 0.0
            return (recent_hits / total_recent) * 100.0
        # All-time hit rate
        total_requests = self._total_hits + self._total_misses
        if total_requests == 0:
            return 0.0
        return (self._total_hits / total_requests) * 100.0

    def get_response_time_stats(self) -> dict[str, float]:
        """Get response time statistics.

        Returns:
            Dictionary with response time metrics

        """
        hit_times = list(self._hit_times)
        miss_times = list(self._miss_times)

        stats = {
            "avg_hit_time_ms": sum(hit_times) / len(hit_times) if hit_times else 0.0,
            "avg_miss_time_ms": sum(miss_times) / len(miss_times)
            if miss_times
            else 0.0,
            "min_hit_time_ms": min(hit_times) if hit_times else 0.0,
            "max_hit_time_ms": max(hit_times) if hit_times else 0.0,
            "min_miss_time_ms": min(miss_times) if miss_times else 0.0,
            "max_miss_time_ms": max(miss_times) if miss_times else 0.0,
        }

        # Calculate speed improvement from caching
        if stats["avg_miss_time_ms"] > 0 and stats["avg_hit_time_ms"] > 0:
            speedup = stats["avg_miss_time_ms"] / stats["avg_hit_time_ms"]
            stats["cache_speedup_factor"] = speedup

        return stats

    def get_provider_metrics(self) -> dict[str, ProviderMetrics]:
        """Get provider-specific cache metrics.

        Returns:
            Dictionary mapping provider names to their metrics

        """
        return dict(self._provider_metrics)

    def get_top_keys(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get most frequently accessed cache keys.

        Args:
            limit: Maximum number of keys to return

        Returns:
            List of key statistics sorted by access count

        """
        top_keys = []

        for key, access_count in sorted(
            self._key_access_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]:
            hit_count = self._key_hit_counts.get(key, 0)
            hit_rate = (hit_count / access_count * 100) if access_count > 0 else 0

            top_keys.append(
                {
                    "key": key,
                    "access_count": access_count,
                    "hit_count": hit_count,
                    "hit_rate": round(hit_rate, 1),
                },
            )

        return top_keys

    def get_hourly_stats(self, hours: int = 24) -> dict[str, dict[str, int]]:
        """Get hourly cache statistics.

        Args:
            hours: Number of recent hours to include

        Returns:
            Dictionary mapping hour keys to hit/miss counts

        """
        now = datetime.now()
        recent_stats = {}

        for i in range(hours):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime("%Y-%m-%d:%H")

            if hour_key in self._hourly_stats:
                recent_stats[hour_key] = dict(self._hourly_stats[hour_key])
            else:
                recent_stats[hour_key] = {"hits": 0, "misses": 0}

        return recent_stats

    def create_snapshot(self, cache_size: int = 0) -> MetricSnapshot:
        """Create a point-in-time metrics snapshot.

        Args:
            cache_size: Current cache size

        Returns:
            MetricSnapshot with current metrics

        """
        response_stats = self.get_response_time_stats()

        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            hit_rate=self.get_hit_rate(),
            total_requests=self._total_hits + self._total_misses,
            hits=self._total_hits,
            misses=self._total_misses,
            avg_hit_time_ms=response_stats["avg_hit_time_ms"],
            avg_miss_time_ms=response_stats["avg_miss_time_ms"],
            cache_size=cache_size,
            evictions=self._total_evictions,
        )

        self._snapshots.append(snapshot)

        # Keep only recent snapshots (last 100)
        if len(self._snapshots) > 100:
            self._snapshots = self._snapshots[-100:]

        return snapshot

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary.

        Returns:
            Dictionary with complete cache performance metrics

        """
        total_requests = self._total_hits + self._total_misses
        hit_rate = self.get_hit_rate()
        recent_hit_rate = self.get_hit_rate(window_minutes=60)
        response_stats = self.get_response_time_stats()
        uptime_hours = (time.time() - self._start_time) / 3600

        # Calculate performance vs target
        target_performance = (
            hit_rate / self.target_hit_rate if self.target_hit_rate > 0 else 0
        )

        return {
            "overview": {
                "hit_rate": round(hit_rate, 2),
                "target_hit_rate": self.target_hit_rate,
                "target_performance": round(target_performance, 2),
                "recent_hit_rate_1h": round(recent_hit_rate, 2),
                "total_requests": total_requests,
                "cache_hits": self._total_hits,
                "cache_misses": self._total_misses,
                "evictions": self._total_evictions,
                "uptime_hours": round(uptime_hours, 2),
            },
            "performance": response_stats,
            "providers": {
                name: {
                    "requests": metrics.requests,
                    "hit_rate": round(metrics.hit_rate, 2),
                    "cost_saved_usd": round(metrics.cost_saved_usd, 4),
                }
                for name, metrics in self._provider_metrics.items()
            },
            "top_keys": self.get_top_keys(5),
            "status": self._get_performance_status(hit_rate),
        }

    def _get_performance_status(self, hit_rate: float) -> str:
        """Determine cache performance status.

        Args:
            hit_rate: Current hit rate percentage

        Returns:
            Performance status string

        """
        if hit_rate >= self.target_hit_rate * 1.2:
            return "excellent"
        if hit_rate >= self.target_hit_rate:
            return "good"
        if hit_rate >= self.target_hit_rate * 0.8:
            return "acceptable"
        if hit_rate >= self.target_hit_rate * 0.5:
            return "poor"
        return "critical"

    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        self._total_hits = 0
        self._total_misses = 0
        self._total_evictions = 0
        self._recent_hits.clear()
        self._recent_misses.clear()
        self._hit_times.clear()
        self._miss_times.clear()
        self._provider_metrics.clear()
        self._key_access_counts.clear()
        self._key_hit_counts.clear()
        self._hourly_stats.clear()
        self._snapshots.clear()
        self._start_time = time.time()

        logger.info("Cache metrics reset")
