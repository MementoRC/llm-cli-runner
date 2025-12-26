"""Performance metrics and monitoring for MCP Server Cheap LLM.

This module provides comprehensive performance tracking including response times,
throughput metrics, resource usage tracking, and statistical analysis. Follows
atomic design patterns with clear data structures.

Key classes:
    PerformanceMetrics: Main metrics tracking and aggregation
    TimingDecorator: Decorator for timing async functions
    MetricsAggregator: Statistical aggregation for metrics
    LatencyTracker: p50, p95, p99 latency calculations

Example:
    >>> metrics = PerformanceMetrics()
    >>> await metrics.record_request("gemini", 150.5, success=True)
    >>> stats = metrics.get_latency_stats("gemini")
    >>> print(f"p95: {stats['p95']:.2f}ms")

"""

import asyncio
import bisect
import functools
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from mcp_server_cheap_llm.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for generic async function decoration
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RequestMetric:
    """Individual request metric record.

    Attributes:
        provider: Provider that handled the request
        response_time_ms: Response time in milliseconds
        success: Whether the request succeeded
        timestamp: When the metric was recorded
        tokens_used: Number of tokens consumed (optional)
        error_type: Type of error if failed (optional)

    """

    provider: str
    response_time_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    tokens_used: int = 0
    error_type: str | None = None


class LatencyStats(BaseModel):
    """Latency statistics with percentile calculations.

    Attributes:
        min_ms: Minimum latency in milliseconds
        max_ms: Maximum latency in milliseconds
        avg_ms: Average latency in milliseconds
        median_ms: Median (p50) latency in milliseconds
        p50_ms: 50th percentile latency
        p95_ms: 95th percentile latency
        p99_ms: 99th percentile latency
        std_dev_ms: Standard deviation in milliseconds
        sample_count: Number of samples in calculation

    """

    min_ms: float = Field(default=0.0, ge=0.0)
    max_ms: float = Field(default=0.0, ge=0.0)
    avg_ms: float = Field(default=0.0, ge=0.0)
    median_ms: float = Field(default=0.0, ge=0.0)
    p50_ms: float = Field(default=0.0, ge=0.0)
    p95_ms: float = Field(default=0.0, ge=0.0)
    p99_ms: float = Field(default=0.0, ge=0.0)
    std_dev_ms: float = Field(default=0.0, ge=0.0)
    sample_count: int = Field(default=0, ge=0)


class ThroughputStats(BaseModel):
    """Throughput statistics for a time window.

    Attributes:
        requests_per_second: Average requests per second
        requests_per_minute: Average requests per minute
        successful_requests: Total successful requests in window
        failed_requests: Total failed requests in window
        success_rate: Success rate as percentage
        tokens_per_minute: Average tokens processed per minute
        window_seconds: Duration of measurement window

    """

    requests_per_second: float = Field(default=0.0, ge=0.0)
    requests_per_minute: float = Field(default=0.0, ge=0.0)
    successful_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    tokens_per_minute: float = Field(default=0.0, ge=0.0)
    window_seconds: int = Field(default=60, ge=1)


class PerformanceSnapshot(BaseModel):
    """Point-in-time performance snapshot.

    Attributes:
        timestamp: When the snapshot was taken
        latency_stats: Latency statistics by provider
        throughput_stats: Throughput statistics by provider
        active_requests: Number of currently active requests
        queue_depth: Current queue depth
        resource_usage: Resource usage metrics

    """

    timestamp: datetime = Field(default_factory=datetime.now)
    latency_stats: dict[str, LatencyStats] = Field(default_factory=dict)
    throughput_stats: dict[str, ThroughputStats] = Field(default_factory=dict)
    active_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    resource_usage: dict[str, float] = Field(default_factory=dict)


class LatencyTracker:
    """Tracks latency values with efficient percentile calculation.

    Uses a sorted list approach for accurate percentile calculation
    with configurable sample retention.

    Attributes:
        max_samples: Maximum number of samples to retain
        samples: Sorted list of latency values

    Example:
        >>> tracker = LatencyTracker(max_samples=1000)
        >>> tracker.add_sample(150.5)
        >>> tracker.add_sample(200.3)
        >>> print(f"p95: {tracker.get_percentile(95):.2f}ms")

    """

    def __init__(self, max_samples: int = 1000) -> None:
        """Initialize latency tracker.

        Args:
            max_samples: Maximum number of samples to retain

        """
        self.max_samples = max_samples
        self._samples: list[float] = []
        self._lock = asyncio.Lock()

    async def add_sample(self, latency_ms: float) -> None:
        """Add a latency sample.

        Args:
            latency_ms: Latency value in milliseconds

        """
        async with self._lock:
            bisect.insort(self._samples, latency_ms)

            # Trim to max samples, removing oldest (median values)
            if len(self._samples) > self.max_samples:
                # Remove from middle to preserve distribution shape
                mid = len(self._samples) // 2
                self._samples.pop(mid)

    def add_sample_sync(self, latency_ms: float) -> None:
        """Add a latency sample synchronously (for decorators).

        Args:
            latency_ms: Latency value in milliseconds

        """
        bisect.insort(self._samples, latency_ms)

        if len(self._samples) > self.max_samples:
            mid = len(self._samples) // 2
            self._samples.pop(mid)

    def get_percentile(self, percentile: float) -> float:
        """Get the value at a given percentile.

        Args:
            percentile: Percentile to calculate (0-100)

        Returns:
            Value at the given percentile, or 0.0 if no samples

        """
        if not self._samples:
            return 0.0

        if percentile <= 0:
            return self._samples[0]
        if percentile >= 100:
            return self._samples[-1]

        # Calculate index for percentile
        index = (percentile / 100.0) * (len(self._samples) - 1)
        lower_idx = int(index)
        upper_idx = min(lower_idx + 1, len(self._samples) - 1)

        # Linear interpolation
        fraction = index - lower_idx
        return self._samples[lower_idx] + fraction * (
            self._samples[upper_idx] - self._samples[lower_idx]
        )

    def get_stats(self) -> LatencyStats:
        """Calculate comprehensive latency statistics.

        Returns:
            LatencyStats with all percentiles and statistics

        """
        if not self._samples:
            return LatencyStats()

        try:
            std_dev = statistics.stdev(self._samples) if len(self._samples) > 1 else 0.0
        except statistics.StatisticsError:
            std_dev = 0.0

        return LatencyStats(
            min_ms=self._samples[0],
            max_ms=self._samples[-1],
            avg_ms=statistics.mean(self._samples),
            median_ms=self.get_percentile(50),
            p50_ms=self.get_percentile(50),
            p95_ms=self.get_percentile(95),
            p99_ms=self.get_percentile(99),
            std_dev_ms=std_dev,
            sample_count=len(self._samples),
        )

    def clear(self) -> None:
        """Clear all samples."""
        self._samples.clear()


class MetricsAggregator:
    """Aggregates metrics over time windows for throughput calculation.

    Maintains a sliding window of metrics for accurate throughput
    and success rate calculations.

    Attributes:
        window_seconds: Size of the sliding window in seconds
        max_events: Maximum events to retain

    Example:
        >>> aggregator = MetricsAggregator(window_seconds=60)
        >>> aggregator.record_event(success=True, tokens=100)
        >>> stats = aggregator.get_throughput_stats()

    """

    def __init__(self, window_seconds: int = 60, max_events: int = 10000) -> None:
        """Initialize metrics aggregator.

        Args:
            window_seconds: Size of sliding window in seconds
            max_events: Maximum events to retain

        """
        self.window_seconds = window_seconds
        self.max_events = max_events
        self._events: deque[tuple[float, bool, int]] = deque(maxlen=max_events)
        self._lock = asyncio.Lock()

    async def record_event(self, success: bool, tokens: int = 0) -> None:
        """Record an event in the aggregator.

        Args:
            success: Whether the event was successful
            tokens: Number of tokens processed

        """
        async with self._lock:
            self._events.append((time.time(), success, tokens))

    def record_event_sync(self, success: bool, tokens: int = 0) -> None:
        """Record an event synchronously.

        Args:
            success: Whether the event was successful
            tokens: Number of tokens processed

        """
        self._events.append((time.time(), success, tokens))

    def _cleanup_old_events(self) -> None:
        """Remove events outside the window."""
        cutoff = time.time() - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def get_throughput_stats(self) -> ThroughputStats:
        """Calculate throughput statistics for the current window.

        Returns:
            ThroughputStats for the sliding window

        """
        self._cleanup_old_events()

        if not self._events:
            return ThroughputStats(window_seconds=self.window_seconds)

        successful = sum(1 for _, success, _ in self._events if success)
        failed = len(self._events) - successful
        total_tokens = sum(tokens for _, _, tokens in self._events)

        # Calculate time span
        oldest = self._events[0][0]
        newest = self._events[-1][0]
        time_span = max(newest - oldest, 1.0)  # Avoid division by zero

        requests_per_second = len(self._events) / time_span
        success_rate = (successful / len(self._events) * 100) if self._events else 0.0

        return ThroughputStats(
            requests_per_second=requests_per_second,
            requests_per_minute=requests_per_second * 60,
            successful_requests=successful,
            failed_requests=failed,
            success_rate=success_rate,
            tokens_per_minute=(total_tokens / time_span) * 60 if time_span > 0 else 0.0,
            window_seconds=self.window_seconds,
        )

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()


class PerformanceMetrics:
    """Main performance metrics tracking and aggregation system.

    Provides comprehensive performance monitoring including latency tracking,
    throughput calculation, and historical metrics storage.

    Attributes:
        max_history: Maximum historical snapshots to retain
        window_seconds: Throughput calculation window size

    Example:
        >>> metrics = PerformanceMetrics()
        >>> await metrics.record_request("gemini", 150.5, success=True)
        >>> snapshot = metrics.get_current_snapshot()
        >>> print(f"Active requests: {snapshot.active_requests}")

    """

    def __init__(
        self,
        max_history: int = 100,
        window_seconds: int = 60,
        max_samples_per_provider: int = 1000,
    ) -> None:
        """Initialize performance metrics.

        Args:
            max_history: Maximum historical snapshots to retain
            window_seconds: Throughput calculation window size
            max_samples_per_provider: Maximum latency samples per provider

        """
        self.max_history = max_history
        self.window_seconds = window_seconds
        self.max_samples_per_provider = max_samples_per_provider

        # Per-provider tracking
        self._latency_trackers: dict[str, LatencyTracker] = defaultdict(
            lambda: LatencyTracker(max_samples=self.max_samples_per_provider)
        )
        self._throughput_aggregators: dict[str, MetricsAggregator] = defaultdict(
            lambda: MetricsAggregator(window_seconds=self.window_seconds)
        )

        # Global tracking
        self._global_latency = LatencyTracker(max_samples=max_samples_per_provider * 10)
        self._global_throughput = MetricsAggregator(window_seconds=window_seconds)

        # Request tracking
        self._active_requests: dict[str, int] = defaultdict(int)
        self._total_requests: dict[str, int] = defaultdict(int)

        # History
        self._history: deque[PerformanceSnapshot] = deque(maxlen=max_history)

        # Locks
        self._lock = asyncio.Lock()

        logger.info(
            "Performance metrics initialized",
            max_history=max_history,
            window_seconds=window_seconds,
        )

    async def record_request(
        self,
        provider: str,
        response_time_ms: float,
        success: bool = True,
        tokens_used: int = 0,
        error_type: str | None = None,
    ) -> None:
        """Record a completed request.

        Args:
            provider: Provider that handled the request
            response_time_ms: Response time in milliseconds
            success: Whether the request succeeded
            tokens_used: Number of tokens consumed
            error_type: Type of error if failed

        """
        async with self._lock:
            # Update latency tracking
            await self._latency_trackers[provider].add_sample(response_time_ms)
            await self._global_latency.add_sample(response_time_ms)

            # Update throughput tracking
            await self._throughput_aggregators[provider].record_event(
                success, tokens_used
            )
            await self._global_throughput.record_event(success, tokens_used)

            # Update counters
            self._total_requests[provider] += 1

            logger.debug(
                "Request metric recorded",
                provider=provider,
                response_time_ms=response_time_ms,
                success=success,
            )

    def record_request_sync(
        self,
        provider: str,
        response_time_ms: float,
        success: bool = True,
        tokens_used: int = 0,
    ) -> None:
        """Record a completed request synchronously.

        Args:
            provider: Provider that handled the request
            response_time_ms: Response time in milliseconds
            success: Whether the request succeeded
            tokens_used: Number of tokens consumed

        """
        self._latency_trackers[provider].add_sample_sync(response_time_ms)
        self._global_latency.add_sample_sync(response_time_ms)
        self._throughput_aggregators[provider].record_event_sync(success, tokens_used)
        self._global_throughput.record_event_sync(success, tokens_used)
        self._total_requests[provider] += 1

    async def start_request(self, provider: str) -> None:
        """Mark a request as started (for active request counting).

        Args:
            provider: Provider handling the request

        """
        async with self._lock:
            self._active_requests[provider] += 1

    async def end_request(self, provider: str) -> None:
        """Mark a request as ended.

        Args:
            provider: Provider that handled the request

        """
        async with self._lock:
            self._active_requests[provider] = max(
                0, self._active_requests[provider] - 1
            )

    def get_latency_stats(self, provider: str | None = None) -> LatencyStats:
        """Get latency statistics for a provider or globally.

        Args:
            provider: Provider name or None for global stats

        Returns:
            LatencyStats for the specified scope

        """
        if provider:
            return self._latency_trackers[provider].get_stats()
        return self._global_latency.get_stats()

    def get_throughput_stats(self, provider: str | None = None) -> ThroughputStats:
        """Get throughput statistics for a provider or globally.

        Args:
            provider: Provider name or None for global stats

        Returns:
            ThroughputStats for the specified scope

        """
        if provider:
            return self._throughput_aggregators[provider].get_throughput_stats()
        return self._global_throughput.get_throughput_stats()

    def get_active_requests(self, provider: str | None = None) -> int:
        """Get count of active requests.

        Args:
            provider: Provider name or None for total

        Returns:
            Number of active requests

        """
        if provider:
            return self._active_requests.get(provider, 0)
        return sum(self._active_requests.values())

    def get_total_requests(self, provider: str | None = None) -> int:
        """Get total request count.

        Args:
            provider: Provider name or None for total

        Returns:
            Total number of requests processed

        """
        if provider:
            return self._total_requests.get(provider, 0)
        return sum(self._total_requests.values())

    def get_current_snapshot(self, queue_depth: int = 0) -> PerformanceSnapshot:
        """Create a point-in-time performance snapshot.

        Args:
            queue_depth: Current queue depth to include

        Returns:
            PerformanceSnapshot with current metrics

        """
        latency_stats = {}
        throughput_stats = {}

        for provider in self._latency_trackers:
            latency_stats[provider] = self._latency_trackers[provider].get_stats()
            throughput_stats[provider] = self._throughput_aggregators[
                provider
            ].get_throughput_stats()

        # Add global stats
        latency_stats["_global"] = self._global_latency.get_stats()
        throughput_stats["_global"] = self._global_throughput.get_throughput_stats()

        return PerformanceSnapshot(
            timestamp=datetime.now(),
            latency_stats=latency_stats,
            throughput_stats=throughput_stats,
            active_requests=self.get_active_requests(),
            queue_depth=queue_depth,
        )

    async def take_snapshot(self, queue_depth: int = 0) -> PerformanceSnapshot:
        """Take and store a performance snapshot.

        Args:
            queue_depth: Current queue depth to include

        Returns:
            The stored PerformanceSnapshot

        """
        snapshot = self.get_current_snapshot(queue_depth)

        async with self._lock:
            self._history.append(snapshot)

        return snapshot

    def get_history(self, limit: int | None = None) -> list[PerformanceSnapshot]:
        """Get historical performance snapshots.

        Args:
            limit: Maximum number of snapshots to return

        Returns:
            List of PerformanceSnapshot in chronological order

        """
        snapshots = list(self._history)
        if limit:
            return snapshots[-limit:]
        return snapshots

    def get_provider_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary statistics for all providers.

        Returns:
            Dictionary with provider-level summaries

        """
        summary = {}

        for provider in set(self._latency_trackers.keys()) | set(
            self._throughput_aggregators.keys()
        ):
            latency = self._latency_trackers[provider].get_stats()
            throughput = self._throughput_aggregators[provider].get_throughput_stats()

            summary[provider] = {
                "total_requests": self._total_requests.get(provider, 0),
                "active_requests": self._active_requests.get(provider, 0),
                "latency": {
                    "avg_ms": latency.avg_ms,
                    "p95_ms": latency.p95_ms,
                    "p99_ms": latency.p99_ms,
                },
                "throughput": {
                    "requests_per_minute": throughput.requests_per_minute,
                    "success_rate": throughput.success_rate,
                },
            }

        return summary

    def clear(self) -> None:
        """Clear all metrics data."""
        self._latency_trackers.clear()
        self._throughput_aggregators.clear()
        self._global_latency.clear()
        self._global_throughput.clear()
        self._active_requests.clear()
        self._total_requests.clear()
        self._history.clear()

        logger.info("Performance metrics cleared")


def timing_decorator(
    metrics: PerformanceMetrics,
    provider: str = "default",
) -> Callable[[F], F]:
    """Decorator factory for timing async functions.

    Args:
        metrics: PerformanceMetrics instance to record to
        provider: Provider name for metrics categorization

    Returns:
        Decorator function

    Example:
        >>> metrics = PerformanceMetrics()
        >>> @timing_decorator(metrics, provider="gemini")
        ... async def my_function():
        ...     await asyncio.sleep(0.1)
        ...     return "result"

    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            success = True
            error_type = None

            try:
                await metrics.start_request(provider)
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_type = type(e).__name__
                raise
            finally:
                end_time = time.perf_counter()
                response_time_ms = (end_time - start_time) * 1000

                await metrics.end_request(provider)
                await metrics.record_request(
                    provider=provider,
                    response_time_ms=response_time_ms,
                    success=success,
                    error_type=error_type,
                )

        return wrapper  # type: ignore

    return decorator


def sync_timing_decorator(
    metrics: PerformanceMetrics,
    provider: str = "default",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory for timing synchronous functions.

    Args:
        metrics: PerformanceMetrics instance to record to
        provider: Provider name for metrics categorization

    Returns:
        Decorator function

    Example:
        >>> metrics = PerformanceMetrics()
        >>> @sync_timing_decorator(metrics, provider="local")
        ... def my_sync_function():
        ...     time.sleep(0.1)
        ...     return "result"

    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            success = True

            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                success = False
                raise
            finally:
                end_time = time.perf_counter()
                response_time_ms = (end_time - start_time) * 1000

                metrics.record_request_sync(
                    provider=provider,
                    response_time_ms=response_time_ms,
                    success=success,
                )

        return wrapper

    return decorator


class MetricsReporter:
    """Formats and reports metrics in various formats.

    Provides formatted output for metrics suitable for logging,
    monitoring systems, or API responses.

    Example:
        >>> metrics = PerformanceMetrics()
        >>> reporter = MetricsReporter(metrics)
        >>> report = reporter.generate_report()

    """

    def __init__(self, metrics: PerformanceMetrics) -> None:
        """Initialize metrics reporter.

        Args:
            metrics: PerformanceMetrics instance to report on

        """
        self._metrics = metrics

    def generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive metrics report.

        Returns:
            Dictionary with formatted metrics report

        """
        snapshot = self._metrics.get_current_snapshot()
        provider_summary = self._metrics.get_provider_summary()

        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "summary": {
                "total_requests": self._metrics.get_total_requests(),
                "active_requests": snapshot.active_requests,
                "queue_depth": snapshot.queue_depth,
            },
            "global_latency": snapshot.latency_stats.get(
                "_global", LatencyStats()
            ).model_dump(),
            "global_throughput": snapshot.throughput_stats.get(
                "_global", ThroughputStats()
            ).model_dump(),
            "providers": provider_summary,
        }

    def format_for_logging(self) -> str:
        """Format metrics for structured logging.

        Returns:
            Formatted string suitable for logging

        """
        global_latency = self._metrics.get_latency_stats()
        global_throughput = self._metrics.get_throughput_stats()

        return (
            f"Metrics: requests/min={global_throughput.requests_per_minute:.1f}, "
            f"success_rate={global_throughput.success_rate:.1f}%, "
            f"p95={global_latency.p95_ms:.1f}ms, "
            f"p99={global_latency.p99_ms:.1f}ms, "
            f"active={self._metrics.get_active_requests()}"
        )
