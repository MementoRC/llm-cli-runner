"""Unit tests for performance metrics module - TDD approach."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from mcp_server_llm_cli_runner.utils.performance import (
    LatencyStats,
    LatencyTracker,
    MetricsAggregator,
    MetricsReporter,
    PerformanceMetrics,
    PerformanceSnapshot,
    ThroughputStats,
    sync_timing_decorator,
    timing_decorator,
)


class TestLatencyTracker:
    """Test suite for LatencyTracker class."""

    def test_latency_tracker_initialization(self):
        """Test LatencyTracker can be initialized."""
        tracker = LatencyTracker(max_samples=100)
        assert tracker is not None
        assert tracker.max_samples == 100

    def test_latency_tracker_add_sample_sync(self):
        """Test adding samples synchronously."""
        tracker = LatencyTracker()
        tracker.add_sample_sync(100.0)
        tracker.add_sample_sync(200.0)
        tracker.add_sample_sync(150.0)

        stats = tracker.get_stats()
        assert stats.sample_count == 3
        assert stats.min_ms == 100.0
        assert stats.max_ms == 200.0

    @pytest.mark.asyncio
    async def test_latency_tracker_add_sample_async(self):
        """Test adding samples asynchronously."""
        tracker = LatencyTracker()
        await tracker.add_sample(100.0)
        await tracker.add_sample(200.0)

        stats = tracker.get_stats()
        assert stats.sample_count == 2

    def test_latency_tracker_percentile_calculation(self):
        """Test percentile calculations are accurate."""
        tracker = LatencyTracker()

        # Add 100 samples from 1 to 100
        for i in range(1, 101):
            tracker.add_sample_sync(float(i))

        # Test percentiles
        p50 = tracker.get_percentile(50)
        assert 49 <= p50 <= 51  # Should be around 50

        p95 = tracker.get_percentile(95)
        assert 94 <= p95 <= 96  # Should be around 95

        p99 = tracker.get_percentile(99)
        assert 98 <= p99 <= 100  # Should be around 99

    def test_latency_tracker_empty_stats(self):
        """Test stats for empty tracker."""
        tracker = LatencyTracker()
        stats = tracker.get_stats()

        assert stats.sample_count == 0
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.avg_ms == 0.0
        assert stats.p95_ms == 0.0

    def test_latency_tracker_max_samples_limit(self):
        """Test that tracker respects max samples limit."""
        tracker = LatencyTracker(max_samples=10)

        # Add 20 samples
        for i in range(20):
            tracker.add_sample_sync(float(i))

        stats = tracker.get_stats()
        assert stats.sample_count == 10

    def test_latency_tracker_clear(self):
        """Test clearing the tracker."""
        tracker = LatencyTracker()
        tracker.add_sample_sync(100.0)
        tracker.add_sample_sync(200.0)

        tracker.clear()

        stats = tracker.get_stats()
        assert stats.sample_count == 0

    def test_latency_tracker_standard_deviation(self):
        """Test standard deviation calculation."""
        tracker = LatencyTracker()

        # Add samples with known variance
        samples = [100, 200, 300, 400, 500]
        for s in samples:
            tracker.add_sample_sync(float(s))

        stats = tracker.get_stats()
        assert stats.std_dev_ms > 0  # Should have non-zero std dev


class TestMetricsAggregator:
    """Test suite for MetricsAggregator class."""

    def test_aggregator_initialization(self):
        """Test MetricsAggregator can be initialized."""
        aggregator = MetricsAggregator(window_seconds=60)
        assert aggregator is not None
        assert aggregator.window_seconds == 60

    @pytest.mark.asyncio
    async def test_aggregator_record_event(self):
        """Test recording events."""
        aggregator = MetricsAggregator()
        await aggregator.record_event(success=True, tokens=100)
        await aggregator.record_event(success=True, tokens=50)
        await aggregator.record_event(success=False, tokens=0)

        stats = aggregator.get_throughput_stats()
        assert stats.successful_requests == 2
        assert stats.failed_requests == 1

    def test_aggregator_record_event_sync(self):
        """Test recording events synchronously."""
        aggregator = MetricsAggregator()
        aggregator.record_event_sync(success=True, tokens=100)
        aggregator.record_event_sync(success=False, tokens=0)

        stats = aggregator.get_throughput_stats()
        assert stats.successful_requests == 1
        assert stats.failed_requests == 1

    def test_aggregator_success_rate(self):
        """Test success rate calculation."""
        aggregator = MetricsAggregator()

        # 8 successes, 2 failures = 80% success rate
        for _ in range(8):
            aggregator.record_event_sync(success=True)
        for _ in range(2):
            aggregator.record_event_sync(success=False)

        stats = aggregator.get_throughput_stats()
        assert 79 <= stats.success_rate <= 81  # Allow small variance

    def test_aggregator_empty_stats(self):
        """Test stats for empty aggregator."""
        aggregator = MetricsAggregator()
        stats = aggregator.get_throughput_stats()

        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.requests_per_second == 0.0

    def test_aggregator_clear(self):
        """Test clearing the aggregator."""
        aggregator = MetricsAggregator()
        aggregator.record_event_sync(success=True, tokens=100)

        aggregator.clear()

        stats = aggregator.get_throughput_stats()
        assert stats.successful_requests == 0


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics class."""

    def test_performance_metrics_initialization(self):
        """Test PerformanceMetrics can be initialized."""
        metrics = PerformanceMetrics()
        assert metrics is not None
        assert metrics.max_history == 100

    @pytest.mark.asyncio
    async def test_record_request(self):
        """Test recording a request."""
        metrics = PerformanceMetrics()
        await metrics.record_request(
            provider="gemini",
            response_time_ms=150.5,
            success=True,
            tokens_used=100,
        )

        stats = metrics.get_latency_stats("gemini")
        assert stats.sample_count == 1
        assert stats.avg_ms == 150.5

    @pytest.mark.asyncio
    async def test_record_request_multiple_providers(self):
        """Test recording requests for multiple providers."""
        metrics = PerformanceMetrics()

        await metrics.record_request("gemini", 100.0, True)
        await metrics.record_request("openai", 200.0, True)
        await metrics.record_request("gemini", 150.0, True)

        gemini_stats = metrics.get_latency_stats("gemini")
        openai_stats = metrics.get_latency_stats("openai")

        assert gemini_stats.sample_count == 2
        assert openai_stats.sample_count == 1

    @pytest.mark.asyncio
    async def test_active_request_tracking(self):
        """Test active request counting."""
        metrics = PerformanceMetrics()

        await metrics.start_request("gemini")
        await metrics.start_request("gemini")
        assert metrics.get_active_requests("gemini") == 2

        await metrics.end_request("gemini")
        assert metrics.get_active_requests("gemini") == 1

    @pytest.mark.asyncio
    async def test_global_stats(self):
        """Test global statistics aggregation."""
        metrics = PerformanceMetrics()

        await metrics.record_request("gemini", 100.0, True)
        await metrics.record_request("openai", 200.0, True)

        global_stats = metrics.get_latency_stats()  # No provider = global
        assert global_stats.sample_count == 2

    @pytest.mark.asyncio
    async def test_get_current_snapshot(self):
        """Test getting current performance snapshot."""
        metrics = PerformanceMetrics()

        await metrics.record_request("gemini", 100.0, True)
        await metrics.start_request("gemini")

        snapshot = metrics.get_current_snapshot(queue_depth=5)

        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.active_requests == 1
        assert snapshot.queue_depth == 5
        assert "gemini" in snapshot.latency_stats

    @pytest.mark.asyncio
    async def test_take_snapshot(self):
        """Test taking and storing snapshots."""
        metrics = PerformanceMetrics()

        await metrics.record_request("gemini", 100.0, True)
        snapshot = await metrics.take_snapshot()

        history = metrics.get_history()
        assert len(history) == 1
        assert history[0] == snapshot

    def test_get_provider_summary(self):
        """Test getting provider summary."""
        metrics = PerformanceMetrics()
        metrics.record_request_sync("gemini", 100.0, True, 50)
        metrics.record_request_sync("gemini", 150.0, True, 75)

        summary = metrics.get_provider_summary()

        assert "gemini" in summary
        assert summary["gemini"]["total_requests"] == 2

    def test_clear_metrics(self):
        """Test clearing all metrics."""
        metrics = PerformanceMetrics()
        metrics.record_request_sync("gemini", 100.0, True)

        metrics.clear()

        assert metrics.get_total_requests() == 0


class TestTimingDecorators:
    """Test suite for timing decorators."""

    @pytest.mark.asyncio
    async def test_timing_decorator_basic(self):
        """Test basic timing decorator functionality."""
        metrics = PerformanceMetrics()

        @timing_decorator(metrics, provider="test")
        async def slow_function():
            await asyncio.sleep(0.1)
            return "result"

        result = await slow_function()

        assert result == "result"
        stats = metrics.get_latency_stats("test")
        assert stats.sample_count == 1
        assert stats.avg_ms >= 100  # At least 100ms

    @pytest.mark.asyncio
    async def test_timing_decorator_exception(self):
        """Test timing decorator handles exceptions."""
        metrics = PerformanceMetrics()

        @timing_decorator(metrics, provider="test")
        async def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_function()

        stats = metrics.get_latency_stats("test")
        assert stats.sample_count == 1  # Still recorded

        throughput = metrics.get_throughput_stats("test")
        assert throughput.failed_requests == 1

    def test_sync_timing_decorator(self):
        """Test synchronous timing decorator."""
        metrics = PerformanceMetrics()

        @sync_timing_decorator(metrics, provider="sync_test")
        def slow_sync_function():
            time.sleep(0.05)
            return "sync_result"

        result = slow_sync_function()

        assert result == "sync_result"
        stats = metrics.get_latency_stats("sync_test")
        assert stats.sample_count == 1
        assert stats.avg_ms >= 50


class TestMetricsReporter:
    """Test suite for MetricsReporter class."""

    def test_reporter_initialization(self):
        """Test MetricsReporter can be initialized."""
        metrics = PerformanceMetrics()
        reporter = MetricsReporter(metrics)
        assert reporter is not None

    def test_generate_report(self):
        """Test generating a metrics report."""
        metrics = PerformanceMetrics()
        metrics.record_request_sync("gemini", 100.0, True, 50)

        reporter = MetricsReporter(metrics)
        report = reporter.generate_report()

        assert "timestamp" in report
        assert "summary" in report
        assert "providers" in report
        assert "global_latency" in report

    def test_format_for_logging(self):
        """Test formatting metrics for logging."""
        metrics = PerformanceMetrics()
        metrics.record_request_sync("gemini", 100.0, True)

        reporter = MetricsReporter(metrics)
        log_str = reporter.format_for_logging()

        assert isinstance(log_str, str)
        assert "requests/min" in log_str or "Metrics" in log_str


class TestLatencyStats:
    """Test suite for LatencyStats model."""

    def test_latency_stats_defaults(self):
        """Test LatencyStats default values."""
        stats = LatencyStats()

        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.avg_ms == 0.0
        assert stats.p50_ms == 0.0
        assert stats.p95_ms == 0.0
        assert stats.p99_ms == 0.0
        assert stats.sample_count == 0

    def test_latency_stats_validation(self):
        """Test LatencyStats validation."""
        stats = LatencyStats(
            min_ms=10.0,
            max_ms=100.0,
            avg_ms=50.0,
            p50_ms=45.0,
            p95_ms=90.0,
            p99_ms=95.0,
            sample_count=100,
        )

        assert stats.min_ms == 10.0
        assert stats.sample_count == 100


class TestThroughputStats:
    """Test suite for ThroughputStats model."""

    def test_throughput_stats_defaults(self):
        """Test ThroughputStats default values."""
        stats = ThroughputStats()

        assert stats.requests_per_second == 0.0
        assert stats.requests_per_minute == 0.0
        assert stats.success_rate == 0.0
        assert stats.window_seconds == 60

    def test_throughput_stats_validation(self):
        """Test ThroughputStats validation."""
        stats = ThroughputStats(
            requests_per_second=10.0,
            requests_per_minute=600.0,
            successful_requests=580,
            failed_requests=20,
            success_rate=96.67,
        )

        assert stats.requests_per_second == 10.0
        assert stats.successful_requests == 580


class TestPerformanceSnapshot:
    """Test suite for PerformanceSnapshot model."""

    def test_snapshot_defaults(self):
        """Test PerformanceSnapshot default values."""
        snapshot = PerformanceSnapshot()

        assert snapshot.active_requests == 0
        assert snapshot.queue_depth == 0
        assert snapshot.timestamp is not None

    def test_snapshot_with_data(self):
        """Test PerformanceSnapshot with actual data."""
        latency_stats = {"gemini": LatencyStats(avg_ms=100.0, sample_count=10)}
        throughput_stats = {"gemini": ThroughputStats(requests_per_minute=100.0)}

        snapshot = PerformanceSnapshot(
            latency_stats=latency_stats,
            throughput_stats=throughput_stats,
            active_requests=5,
            queue_depth=10,
        )

        assert snapshot.active_requests == 5
        assert "gemini" in snapshot.latency_stats
