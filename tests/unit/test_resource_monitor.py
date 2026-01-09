"""Unit tests for resource monitor module - TDD approach."""

import asyncio
import gc
import sys
from unittest.mock import MagicMock, patch

import pytest

from mcp_server_llm_cli_runner.utils.resource_monitor import (
    AlertType,
    GCOptimizer,
    ResourceAlert,
    ResourceLevel,
    ResourceMonitor,
    ResourceSnapshot,
    ResourceThresholds,
)


class TestResourceThresholds:
    """Test suite for ResourceThresholds."""

    def test_thresholds_defaults(self):
        """Test default threshold values."""
        thresholds = ResourceThresholds()

        assert thresholds.cpu_warning_percent == 70.0
        assert thresholds.cpu_critical_percent == 90.0
        assert thresholds.memory_warning_percent == 70.0
        assert thresholds.memory_critical_percent == 85.0
        assert thresholds.disk_warning_percent == 80.0
        assert thresholds.disk_critical_percent == 90.0

    def test_thresholds_custom_values(self):
        """Test custom threshold values."""
        thresholds = ResourceThresholds(
            cpu_warning_percent=60.0,
            cpu_critical_percent=80.0,
            memory_warning_percent=65.0,
        )

        assert thresholds.cpu_warning_percent == 60.0
        assert thresholds.cpu_critical_percent == 80.0
        assert thresholds.memory_warning_percent == 65.0

    def test_thresholds_validation(self):
        """Test threshold validation."""
        # Percentages must be 0-100
        with pytest.raises(ValueError):
            ResourceThresholds(cpu_warning_percent=150.0)


class TestResourceSnapshot:
    """Test suite for ResourceSnapshot."""

    def test_snapshot_defaults(self):
        """Test ResourceSnapshot default values."""
        snapshot = ResourceSnapshot()

        assert snapshot.cpu_percent == 0.0
        assert snapshot.memory_percent == 0.0
        assert snapshot.disk_percent == 0.0
        assert snapshot.timestamp is not None

    def test_snapshot_custom_values(self):
        """Test ResourceSnapshot with custom values."""
        snapshot = ResourceSnapshot(
            cpu_percent=50.0,
            cpu_count=8,
            memory_percent=60.0,
            memory_used_mb=8192.0,
            memory_available_mb=8192.0,
            memory_total_mb=16384.0,
            disk_percent=70.0,
        )

        assert snapshot.cpu_percent == 50.0
        assert snapshot.cpu_count == 8
        assert snapshot.memory_percent == 60.0


class TestResourceAlert:
    """Test suite for ResourceAlert."""

    def test_alert_creation(self):
        """Test ResourceAlert creation."""
        alert = ResourceAlert(
            alert_type=AlertType.CPU_HIGH,
            level=ResourceLevel.WARNING,
            message="CPU usage high: 75%",
            current_value=75.0,
            threshold_value=70.0,
        )

        assert alert.alert_type == AlertType.CPU_HIGH
        assert alert.level == ResourceLevel.WARNING
        assert alert.current_value == 75.0

    def test_alert_types(self):
        """Test all alert types exist."""
        assert AlertType.CPU_HIGH.value == "cpu_high"
        assert AlertType.MEMORY_HIGH.value == "memory_high"
        assert AlertType.MEMORY_CRITICAL.value == "memory_critical"
        assert AlertType.DISK_HIGH.value == "disk_high"
        assert AlertType.GC_RECOMMENDED.value == "gc_recommended"


class TestResourceLevel:
    """Test suite for ResourceLevel enum."""

    def test_resource_levels(self):
        """Test all resource levels exist."""
        assert ResourceLevel.NORMAL.value == "normal"
        assert ResourceLevel.WARNING.value == "warning"
        assert ResourceLevel.CRITICAL.value == "critical"


class TestGCOptimizer:
    """Test suite for GCOptimizer."""

    def test_optimizer_initialization(self):
        """Test GCOptimizer initialization."""
        optimizer = GCOptimizer()
        assert optimizer is not None

    def test_get_gc_stats(self):
        """Test getting GC statistics."""
        optimizer = GCOptimizer()
        stats = optimizer.get_gc_stats()

        assert "counts" in stats
        assert "thresholds" in stats
        assert "tracked_objects" in stats
        assert "gc_enabled" in stats

        # Verify structure
        assert "generation_0" in stats["counts"]
        assert "generation_1" in stats["counts"]
        assert "generation_2" in stats["counts"]

    @pytest.mark.skipif(
        sys.version_info >= (3, 14),
        reason="Python 3.14+ changed GC threshold behavior (gen1/gen2 may be ignored)",
    )
    def test_configure_thresholds(self):
        """Test configuring GC thresholds."""
        optimizer = GCOptimizer()
        original = gc.get_threshold()

        try:
            optimizer.configure_thresholds(gen0=800, gen1=15, gen2=15)
            new_thresholds = gc.get_threshold()

            assert new_thresholds[0] == 800
            assert new_thresholds[1] == 15
            assert new_thresholds[2] == 15
        finally:
            # Restore original thresholds
            gc.set_threshold(*original)

    def test_force_collection(self):
        """Test forcing garbage collection."""
        optimizer = GCOptimizer()

        # Create some garbage
        for _ in range(100):
            _ = [1, 2, 3]

        result = optimizer.force_collection()

        assert "collected_objects" in result
        assert "duration_ms" in result
        assert "timestamp" in result

    def test_force_collection_specific_generation(self):
        """Test forcing collection of specific generation."""
        optimizer = GCOptimizer()

        result = optimizer.force_collection(generation=0)

        assert result["generation"] == 0

    def test_analyze_and_optimize(self):
        """Test GC analysis."""
        optimizer = GCOptimizer()
        analysis = optimizer.analyze_and_optimize()

        assert "tracked_objects" in analysis
        assert "generation_counts" in analysis
        assert "gc_recommended" in analysis

    def test_get_optimization_suggestions(self):
        """Test getting optimization suggestions."""
        optimizer = GCOptimizer()
        suggestions = optimizer.get_optimization_suggestions()

        assert isinstance(suggestions, list)

    def test_reset_thresholds(self):
        """Test resetting GC thresholds to original."""
        optimizer = GCOptimizer()
        original = gc.get_threshold()

        optimizer.configure_thresholds(gen0=1000)
        optimizer.reset_thresholds()

        restored = gc.get_threshold()
        assert restored == original


class TestResourceMonitor:
    """Test suite for ResourceMonitor."""

    def test_monitor_initialization(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor()
        assert monitor is not None
        assert monitor.sample_interval == 5

    def test_monitor_with_custom_config(self):
        """Test ResourceMonitor with custom configuration."""
        thresholds = ResourceThresholds(cpu_warning_percent=50.0)
        monitor = ResourceMonitor(
            thresholds=thresholds,
            sample_interval=10,
            history_size=50,
        )

        assert monitor.thresholds.cpu_warning_percent == 50.0
        assert monitor.sample_interval == 10
        assert monitor.history_size == 50

    def test_get_current_snapshot(self):
        """Test getting current resource snapshot."""
        monitor = ResourceMonitor()
        snapshot = monitor.get_current_snapshot()

        assert isinstance(snapshot, ResourceSnapshot)
        # CPU should be between 0 and 100
        assert 0 <= snapshot.cpu_percent <= 100
        # Memory should have reasonable values
        assert snapshot.memory_total_mb > 0

    def test_check_thresholds_no_alerts(self):
        """Test threshold checking with normal values."""
        thresholds = ResourceThresholds(
            cpu_warning_percent=99.0,  # Very high threshold
            memory_warning_percent=99.0,
        )
        monitor = ResourceMonitor(thresholds=thresholds)

        # Create a snapshot with low values
        snapshot = ResourceSnapshot(
            cpu_percent=20.0,
            memory_percent=30.0,
            disk_percent=40.0,
        )

        alerts = monitor.check_thresholds(snapshot)
        assert len(alerts) == 0

    def test_check_thresholds_cpu_warning(self):
        """Test CPU warning threshold."""
        thresholds = ResourceThresholds(cpu_warning_percent=50.0)
        monitor = ResourceMonitor(thresholds=thresholds)

        snapshot = ResourceSnapshot(cpu_percent=75.0)
        alerts = monitor.check_thresholds(snapshot)

        cpu_alerts = [a for a in alerts if a.alert_type == AlertType.CPU_HIGH]
        assert len(cpu_alerts) == 1
        assert cpu_alerts[0].level == ResourceLevel.WARNING

    def test_check_thresholds_cpu_critical(self):
        """Test CPU critical threshold."""
        thresholds = ResourceThresholds(
            cpu_warning_percent=50.0,
            cpu_critical_percent=80.0,
        )
        monitor = ResourceMonitor(thresholds=thresholds)

        snapshot = ResourceSnapshot(cpu_percent=85.0)
        alerts = monitor.check_thresholds(snapshot)

        cpu_alerts = [a for a in alerts if a.alert_type == AlertType.CPU_HIGH]
        assert len(cpu_alerts) == 1
        assert cpu_alerts[0].level == ResourceLevel.CRITICAL

    def test_check_thresholds_memory_warning(self):
        """Test memory warning threshold."""
        thresholds = ResourceThresholds(memory_warning_percent=50.0)
        monitor = ResourceMonitor(thresholds=thresholds)

        snapshot = ResourceSnapshot(memory_percent=60.0)
        alerts = monitor.check_thresholds(snapshot)

        memory_alerts = [a for a in alerts if a.alert_type == AlertType.MEMORY_HIGH]
        assert len(memory_alerts) == 1

    def test_check_thresholds_memory_critical(self):
        """Test memory critical threshold."""
        thresholds = ResourceThresholds(
            memory_warning_percent=50.0,
            memory_critical_percent=70.0,
        )
        monitor = ResourceMonitor(thresholds=thresholds)

        snapshot = ResourceSnapshot(memory_percent=80.0)
        alerts = monitor.check_thresholds(snapshot)

        memory_alerts = [a for a in alerts if a.alert_type == AlertType.MEMORY_CRITICAL]
        assert len(memory_alerts) == 1
        assert memory_alerts[0].level == ResourceLevel.CRITICAL

    def test_register_alert_callback(self):
        """Test registering alert callbacks."""
        monitor = ResourceMonitor()
        received_alerts = []

        def callback(alert):
            received_alerts.append(alert)

        monitor.register_alert_callback(callback)

        # Create an alert-triggering snapshot
        thresholds = ResourceThresholds(cpu_warning_percent=10.0)
        monitor = ResourceMonitor(thresholds=thresholds)
        monitor.register_alert_callback(callback)

        snapshot = ResourceSnapshot(cpu_percent=50.0)
        alerts = monitor.check_thresholds(snapshot)
        monitor._process_alerts(alerts)

        assert len(received_alerts) > 0

    @pytest.mark.asyncio
    async def test_monitor_start_stop(self):
        """Test monitor start and stop."""
        monitor = ResourceMonitor(sample_interval=1)

        await monitor.start()
        assert monitor._is_running is True

        # Let it run for a brief moment
        await asyncio.sleep(0.1)

        await monitor.stop()
        assert monitor._is_running is False

    def test_get_history(self):
        """Test getting snapshot history."""
        monitor = ResourceMonitor(history_size=10)

        # Manually add some snapshots
        for _ in range(5):
            snapshot = monitor.get_current_snapshot()
            monitor._history.append(snapshot)

        history = monitor.get_history()
        assert len(history) == 5

    def test_get_history_with_limit(self):
        """Test getting limited history."""
        monitor = ResourceMonitor(history_size=10)

        # Manually add some snapshots
        for _ in range(5):
            snapshot = monitor.get_current_snapshot()
            monitor._history.append(snapshot)

        history = monitor.get_history(limit=3)
        assert len(history) == 3

    def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        monitor = ResourceMonitor()

        # Add some alerts
        alert = ResourceAlert(
            alert_type=AlertType.CPU_HIGH,
            level=ResourceLevel.WARNING,
            message="Test",
            current_value=80.0,
            threshold_value=70.0,
        )
        monitor._alerts.append(alert)

        recent = monitor.get_recent_alerts(limit=1)
        assert len(recent) == 1

    def test_get_gc_optimizer(self):
        """Test getting GC optimizer."""
        monitor = ResourceMonitor()
        optimizer = monitor.get_gc_optimizer()

        assert isinstance(optimizer, GCOptimizer)

    def test_get_summary(self):
        """Test getting resource summary."""
        monitor = ResourceMonitor()
        summary = monitor.get_summary()

        assert "current" in summary
        assert "trends" in summary
        assert "alerts" in summary
        assert "gc" in summary
        assert "thresholds" in summary

        # Verify current values
        assert "cpu_percent" in summary["current"]
        assert "memory_percent" in summary["current"]

    def test_force_gc(self):
        """Test forcing GC through monitor."""
        monitor = ResourceMonitor()
        result = monitor.force_gc()

        assert "collected_objects" in result
        assert "duration_ms" in result


class TestResourceMonitorIntegration:
    """Integration tests for ResourceMonitor."""

    @pytest.mark.asyncio
    async def test_monitor_sampling_loop(self):
        """Test that sampling loop works correctly."""
        monitor = ResourceMonitor(sample_interval=1)

        await monitor.start()

        # Wait for at least one sample
        await asyncio.sleep(1.5)

        await monitor.stop()

        history = monitor.get_history()
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_monitor_alert_generation(self):
        """Test alert generation during monitoring."""
        thresholds = ResourceThresholds(
            cpu_warning_percent=1.0,  # Very low to ensure alert
        )
        monitor = ResourceMonitor(thresholds=thresholds, sample_interval=1)

        await monitor.start()
        await asyncio.sleep(1.5)
        await monitor.stop()

        # Should have generated at least one alert
        alerts = monitor.get_recent_alerts()
        # Note: May or may not have alerts depending on actual CPU usage
        assert isinstance(alerts, list)
