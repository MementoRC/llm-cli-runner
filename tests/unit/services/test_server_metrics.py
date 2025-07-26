"""
Unit tests for server metrics service.

This module provides comprehensive test coverage for the MetricsService
including performance tracking, system health monitoring, and metric
collection functionality.
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from mcp_server_git.services.server_metrics import (
    MetricPoint,
    MetricsService,
    PerformanceMetrics,
    SystemHealthMetrics,
)


class TestMetricPoint:
    """Test MetricPoint dataclass."""

    def test_metric_point_creation(self):
        """Test creating a metric point."""
        timestamp = datetime.now()
        labels = {"environment": "test", "service": "git"}

        metric = MetricPoint(
            name="test_metric",
            value=42.5,
            timestamp=timestamp,
            labels=labels,
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.timestamp == timestamp
        assert metric.labels == labels

    def test_metric_point_to_dict(self):
        """Test converting metric point to dictionary."""
        timestamp = datetime.now()
        labels = {"key": "value"}

        metric = MetricPoint(
            name="test_metric",
            value=100.0,
            timestamp=timestamp,
            labels=labels,
        )

        result = metric.to_dict()

        expected = {
            "name": "test_metric",
            "value": 100.0,
            "timestamp": timestamp.isoformat(),
            "labels": labels,
        }

        assert result == expected

    def test_metric_point_default_labels(self):
        """Test metric point with default empty labels."""
        metric = MetricPoint(
            name="test",
            value=1.0,
            timestamp=datetime.now(),
        )

        assert metric.labels == {}


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics()

        assert metrics.operation_count == 0
        assert metrics.total_duration == 0.0
        assert metrics.min_duration == float("inf")
        assert metrics.max_duration == 0.0
        assert metrics.error_count == 0
        assert metrics.last_operation_time is None

    def test_average_duration_calculation(self):
        """Test average duration calculation."""
        metrics = PerformanceMetrics(
            operation_count=4,
            total_duration=10.0,
        )

        assert metrics.average_duration == 2.5

    def test_average_duration_no_operations(self):
        """Test average duration with no operations."""
        metrics = PerformanceMetrics()
        assert metrics.average_duration == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = PerformanceMetrics(
            operation_count=10,
            error_count=2,
        )

        assert metrics.success_rate == 0.8

    def test_success_rate_no_operations(self):
        """Test success rate with no operations."""
        metrics = PerformanceMetrics()
        assert metrics.success_rate == 1.0

    def test_performance_metrics_to_dict(self):
        """Test converting performance metrics to dictionary."""
        timestamp = datetime.now()
        metrics = PerformanceMetrics(
            operation_count=5,
            total_duration=15.0,
            min_duration=1.0,
            max_duration=5.0,
            error_count=1,
            last_operation_time=timestamp,
        )

        result = metrics.to_dict()

        assert result["operation_count"] == 5
        assert result["total_duration"] == 15.0
        assert result["min_duration"] == 1.0
        assert result["max_duration"] == 5.0
        assert result["average_duration"] == 3.0
        assert result["error_count"] == 1
        assert result["success_rate"] == 0.8
        assert result["last_operation_time"] == timestamp.isoformat()


class TestSystemHealthMetrics:
    """Test SystemHealthMetrics dataclass."""

    def test_system_health_metrics_creation(self):
        """Test creating system health metrics."""
        metrics = SystemHealthMetrics()

        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.disk_usage == 0.0
        assert metrics.active_connections == 0
        assert metrics.request_queue_size == 0
        assert metrics.last_health_check is None

    def test_system_health_metrics_to_dict(self):
        """Test converting system health metrics to dictionary."""
        timestamp = datetime.now()
        metrics = SystemHealthMetrics(
            cpu_usage=25.5,
            memory_usage=60.2,
            disk_usage=45.8,
            active_connections=10,
            request_queue_size=3,
            last_health_check=timestamp,
        )

        result = metrics.to_dict()

        expected = {
            "cpu_usage": 25.5,
            "memory_usage": 60.2,
            "disk_usage": 45.8,
            "active_connections": 10,
            "request_queue_size": 3,
            "last_health_check": timestamp.isoformat(),
        }

        assert result == expected


class TestMetricsService:
    """Test MetricsService class."""

    def test_metrics_service_initialization(self):
        """Test metrics service initialization."""
        service = MetricsService(
            max_metric_history=5000,
            health_check_interval=30.0,
            enable_system_metrics=False,
        )

        assert service._max_metric_history == 5000
        assert service._health_check_interval == 30.0
        assert service._enable_system_metrics is False
        assert service._is_running is False
        assert len(service._metric_points) == 0
        assert len(service._performance_metrics) == 0

    @pytest.mark.asyncio
    async def test_start_stop_service(self):
        """Test starting and stopping the metrics service."""
        service = MetricsService(enable_system_metrics=False)

        # Start service
        await service.start()
        assert service._is_running is True

        # Stop service
        await service.stop()
        assert service._is_running is False

    @pytest.mark.asyncio
    async def test_start_already_running(self, caplog):
        """Test starting service when already running."""
        service = MetricsService(enable_system_metrics=False)

        await service.start()
        await service.start()  # Start again

        assert "already running" in caplog.text
        await service.stop()

    def test_record_metric(self):
        """Test recording a custom metric."""
        service = MetricsService()
        labels = {"test": "value"}

        service.record_metric("test_metric", 42.0, labels)

        assert len(service._metric_points) == 1
        metric = service._metric_points[0]
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.labels == labels

    def test_record_metric_without_labels(self):
        """Test recording a metric without labels."""
        service = MetricsService()

        service.record_metric("simple_metric", 100.0)

        assert len(service._metric_points) == 1
        metric = service._metric_points[0]
        assert metric.name == "simple_metric"
        assert metric.value == 100.0
        assert metric.labels == {}

    def test_record_operation_success(self):
        """Test recording a successful operation."""
        service = MetricsService()

        service.record_operation("git_status", 0.5, success=True)

        # Check performance metrics
        assert "git_status" in service._performance_metrics
        metrics = service._performance_metrics["git_status"]
        assert metrics.operation_count == 1
        assert metrics.total_duration == 0.5
        assert metrics.error_count == 0

        # Check metric point was recorded
        assert len(service._metric_points) == 1
        metric = service._metric_points[0]
        assert metric.name == "operation.git_status.duration"
        assert metric.value == 0.5
        assert metric.labels["success"] == "True"

    def test_record_operation_failure(self):
        """Test recording a failed operation."""
        service = MetricsService()

        service.record_operation("git_push", 2.0, success=False)

        metrics = service._performance_metrics["git_push"]
        assert metrics.operation_count == 1
        assert metrics.total_duration == 2.0
        assert metrics.error_count == 1
        assert metrics.success_rate == 0.0

    def test_record_multiple_operations(self):
        """Test recording multiple operations."""
        service = MetricsService()

        # Record multiple operations
        service.record_operation("test_op", 1.0, success=True)
        service.record_operation("test_op", 2.0, success=True)
        service.record_operation("test_op", 3.0, success=False)

        metrics = service._performance_metrics["test_op"]
        assert metrics.operation_count == 3
        assert metrics.total_duration == 6.0
        assert metrics.min_duration == 1.0
        assert metrics.max_duration == 3.0
        assert metrics.average_duration == 2.0
        assert metrics.error_count == 1
        assert metrics.success_rate == 2.0 / 3.0

    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        service = MetricsService()

        # Record some metrics
        service.record_metric("custom_metric", 50.0)
        service.record_operation("test_operation", 1.5)

        summary = service.get_metrics_summary()

        assert "service_uptime" in summary
        assert summary["total_metric_points"] == 2  # 1 custom + 1 operation
        assert "performance_metrics" in summary
        assert "test_operation" in summary["performance_metrics"]
        assert "system_health" in summary
        assert "recent_metrics" in summary
        assert summary["is_running"] is False

    def test_get_operation_metrics(self):
        """Test getting metrics for specific operation."""
        service = MetricsService()

        service.record_operation("specific_op", 2.5)

        # Get existing operation metrics
        metrics = service.get_operation_metrics("specific_op")
        assert metrics is not None
        assert metrics["operation_count"] == 1
        assert metrics["total_duration"] == 2.5

        # Get non-existent operation metrics
        metrics = service.get_operation_metrics("nonexistent")
        assert metrics is None

    def test_clear_metrics(self):
        """Test clearing all metrics."""
        service = MetricsService()

        # Add some metrics
        service.record_metric("test", 1.0)
        service.record_operation("op", 0.5)

        assert len(service._metric_points) == 2
        assert len(service._performance_metrics) == 1

        # Clear metrics
        service.clear_metrics()

        assert len(service._metric_points) == 0
        assert len(service._performance_metrics) == 0

    def test_metric_history_limit(self):
        """Test metric history respects maximum limit."""
        service = MetricsService(max_metric_history=3)

        # Add more metrics than the limit
        for i in range(5):
            service.record_metric(f"metric_{i}", float(i))

        # Should only keep the last 3
        assert len(service._metric_points) == 3

        # Check that the oldest were removed
        names = [metric.name for metric in service._metric_points]
        assert names == ["metric_2", "metric_3", "metric_4"]

    @pytest.mark.asyncio
    async def test_system_health_collection(self):
        """Test system health metrics collection."""
        service = MetricsService(enable_system_metrics=True, health_check_interval=0.1)

        # Mock the psutil within the collect method
        with patch("builtins.__import__") as mock_import:
            mock_psutil = Mock()
            mock_psutil.cpu_percent.return_value = 25.5
            mock_psutil.virtual_memory.return_value.percent = 60.0
            mock_psutil.disk_usage.return_value.percent = 45.0

            def side_effect(module_name, *args, **kwargs):
                if module_name == "psutil":
                    return mock_psutil
                return __import__(module_name, *args, **kwargs)

            mock_import.side_effect = side_effect

            await service.start()

            # Wait for health check to run
            await asyncio.sleep(0.2)

            await service.stop()

        # Check system health was updated
        assert service._system_health.cpu_usage == 25.5
        assert service._system_health.memory_usage == 60.0
        assert service._system_health.disk_usage == 45.0
        assert service._system_health.last_health_check is not None

    @pytest.mark.asyncio
    async def test_system_health_without_psutil(self):
        """Test system health collection without psutil."""
        service = MetricsService(enable_system_metrics=True, health_check_interval=0.1)

        # Mock psutil import failure
        with patch("builtins.__import__") as mock_import:
            def side_effect(module_name, *args, **kwargs):
                if module_name == "psutil":
                    raise ImportError("No module named 'psutil'")
                return __import__(module_name, *args, **kwargs)

            mock_import.side_effect = side_effect
            await service._collect_system_health()

        # Should still update last_health_check
        assert service._system_health.last_health_check is not None
        assert service._system_health.cpu_usage == 0.0  # Default values
        assert service._system_health.memory_usage == 0.0

    def test_get_component_state(self):
        """Test DebuggableComponent implementation - get_component_state."""
        service = MetricsService()

        state = service.get_component_state()

        assert state.name == "MetricsService"
        assert state.status == "STOPPED"
        assert "is_running" in state.metadata
        assert state.metadata["is_running"] is False
        assert "uptime_seconds" in state.metadata
        assert "total_metrics" in state.metadata

    def test_validate_component_healthy(self):
        """Test component validation when healthy."""
        service = MetricsService()
        service._is_running = True

        result = service.validate_component()

        assert result.is_valid is True
        assert len(result.issues) == 0
        assert result.component_name == "MetricsService"

    def test_validate_component_not_running(self):
        """Test component validation when not running."""
        service = MetricsService()
        # Service not started, should be invalid

        result = service.validate_component()

        assert result.is_valid is False
        assert "not running" in result.issues[0]

    def test_validate_component_storage_full(self):
        """Test component validation when storage approaching capacity."""
        service = MetricsService(max_metric_history=10)
        service._is_running = True

        # Fill storage to 95% capacity
        for i in range(9):
            service.record_metric(f"test_{i}", float(i))

        result = service.validate_component()

        assert result.is_valid is False
        assert any("approaching capacity" in issue for issue in result.issues)

    def test_validate_component_stale_health_checks(self):
        """Test component validation with stale health checks."""
        service = MetricsService(enable_system_metrics=True, health_check_interval=60.0)
        service._is_running = True

        # Set stale health check
        service._system_health.last_health_check = datetime.now() - timedelta(minutes=5)

        result = service.validate_component()

        assert result.is_valid is False
        assert any("stale" in issue for issue in result.issues)

    def test_get_debug_info_basic(self):
        """Test getting basic debug information."""
        service = MetricsService()

        debug_info = service.get_debug_info("INFO")

        assert debug_info["component_type"] == "MetricsService"
        assert "is_running" in debug_info
        assert "configuration" in debug_info
        assert debug_info["configuration"]["max_metric_history"] == 10000

    def test_get_debug_info_detailed(self):
        """Test getting detailed debug information."""
        service = MetricsService()
        service.record_metric("test", 1.0)

        debug_info = service.get_debug_info("DEBUG")

        assert debug_info["component_type"] == "MetricsService"
        assert "service_uptime" in debug_info
        assert "total_metric_points" in debug_info
        assert "performance_metrics" in debug_info
        assert debug_info["total_metric_points"] == 1
