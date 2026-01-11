"""System resource monitoring for MCP Server LLM CLI Runner.

This module provides CPU, memory, disk monitoring with alerts and
garbage collection optimization. Follows atomic design patterns.

Key classes:
    ResourceMonitor: Main resource monitoring implementation
    ResourceThresholds: Configurable thresholds for alerts
    ResourceSnapshot: Point-in-time resource usage data

Example:
    >>> monitor = ResourceMonitor()
    >>> await monitor.start()
    >>> snapshot = monitor.get_current_snapshot()
    >>> print(f"CPU: {snapshot.cpu_percent:.1f}%")

"""

import asyncio
import contextlib
import gc
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import psutil
from pydantic import BaseModel, Field

from mcp_server_llm_cli_runner.utils.logging import get_logger

logger = get_logger(__name__)


class ResourceLevel(str, Enum):
    """Resource usage level classification."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of resource alerts."""

    CPU_HIGH = "cpu_high"
    MEMORY_HIGH = "memory_high"
    MEMORY_CRITICAL = "memory_critical"
    DISK_HIGH = "disk_high"
    DISK_CRITICAL = "disk_critical"
    GC_RECOMMENDED = "gc_recommended"


class ResourceThresholds(BaseModel):
    """Configurable thresholds for resource alerts.

    Attributes:
        cpu_warning_percent: CPU usage warning threshold
        cpu_critical_percent: CPU usage critical threshold
        memory_warning_percent: Memory usage warning threshold
        memory_critical_percent: Memory usage critical threshold
        disk_warning_percent: Disk usage warning threshold
        disk_critical_percent: Disk usage critical threshold
        gc_memory_threshold_mb: Memory threshold to trigger GC
        gc_object_threshold: Object count threshold for GC

    """

    cpu_warning_percent: float = Field(default=70.0, ge=0.0, le=100.0)
    cpu_critical_percent: float = Field(default=90.0, ge=0.0, le=100.0)
    memory_warning_percent: float = Field(default=70.0, ge=0.0, le=100.0)
    memory_critical_percent: float = Field(default=85.0, ge=0.0, le=100.0)
    disk_warning_percent: float = Field(default=80.0, ge=0.0, le=100.0)
    disk_critical_percent: float = Field(default=90.0, ge=0.0, le=100.0)
    gc_memory_threshold_mb: int = Field(default=500, ge=100, le=10000)
    gc_object_threshold: int = Field(default=10000, ge=1000, le=1000000)


class ResourceSnapshot(BaseModel):
    """Point-in-time resource usage snapshot.

    Attributes:
        timestamp: When the snapshot was taken
        cpu_percent: Current CPU usage percentage
        cpu_count: Number of CPU cores
        memory_percent: Current memory usage percentage
        memory_used_mb: Memory used in megabytes
        memory_available_mb: Memory available in megabytes
        memory_total_mb: Total system memory in megabytes
        disk_percent: Disk usage percentage
        disk_used_gb: Disk space used in gigabytes
        disk_free_gb: Disk space free in gigabytes
        open_file_descriptors: Number of open file descriptors
        thread_count: Number of active threads
        gc_objects: Number of tracked objects
        gc_collections: GC collection counts by generation
        process_cpu_percent: Process-specific CPU usage
        process_memory_mb: Process memory usage in MB

    """

    timestamp: datetime = Field(default_factory=datetime.now)
    cpu_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    cpu_count: int = Field(default=1, ge=1)
    memory_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_used_mb: float = Field(default=0.0, ge=0.0)
    memory_available_mb: float = Field(default=0.0, ge=0.0)
    memory_total_mb: float = Field(default=0.0, ge=0.0)
    disk_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    disk_used_gb: float = Field(default=0.0, ge=0.0)
    disk_free_gb: float = Field(default=0.0, ge=0.0)
    open_file_descriptors: int = Field(default=0, ge=0)
    thread_count: int = Field(default=0, ge=0)
    gc_objects: int = Field(default=0, ge=0)
    gc_collections: dict[str, int] = Field(default_factory=dict)
    process_cpu_percent: float = Field(default=0.0, ge=0.0)
    process_memory_mb: float = Field(default=0.0, ge=0.0)


@dataclass
class ResourceAlert:
    """Resource usage alert.

    Attributes:
        alert_type: Type of alert
        level: Severity level
        message: Human-readable message
        current_value: Current resource value
        threshold_value: Threshold that was exceeded
        timestamp: When the alert was generated

    """

    alert_type: AlertType
    level: ResourceLevel
    message: str
    current_value: float
    threshold_value: float
    timestamp: float = field(default_factory=time.time)


class GCOptimizer:
    """Garbage collection optimization hooks.

    Provides methods for analyzing and optimizing garbage collection
    based on resource usage patterns.

    Example:
        >>> optimizer = GCOptimizer()
        >>> optimizer.configure_thresholds(
        ...     gen0=700, gen1=10, gen2=10
        ... )
        >>> result = optimizer.analyze_and_optimize()

    """

    def __init__(self) -> None:
        """Initialize GC optimizer."""
        self._last_gc_time = time.time()
        self._gc_history: deque[dict[str, Any]] = deque(maxlen=100)
        self._original_thresholds = gc.get_threshold()

    def get_gc_stats(self) -> dict[str, Any]:
        """Get current garbage collection statistics.

        Returns:
            Dictionary with GC statistics

        """
        counts = gc.get_count()
        thresholds = gc.get_threshold()
        stats = gc.get_stats()

        return {
            "counts": {
                "generation_0": counts[0],
                "generation_1": counts[1],
                "generation_2": counts[2],
            },
            "thresholds": {
                "generation_0": thresholds[0],
                "generation_1": thresholds[1],
                "generation_2": thresholds[2],
            },
            "stats": stats,
            "tracked_objects": len(gc.get_objects()),
            "garbage_count": len(gc.garbage),
            "gc_enabled": gc.isenabled(),
        }

    def configure_thresholds(
        self,
        gen0: int | None = None,
        gen1: int | None = None,
        gen2: int | None = None,
    ) -> None:
        """Configure GC thresholds.

        Args:
            gen0: Threshold for generation 0 (allocations before collection)
            gen1: Threshold for generation 1
            gen2: Threshold for generation 2

        """
        current = gc.get_threshold()

        new_thresholds = (
            gen0 if gen0 is not None else current[0],
            gen1 if gen1 is not None else current[1],
            gen2 if gen2 is not None else current[2],
        )

        gc.set_threshold(*new_thresholds)
        logger.info(f"GC thresholds set to: {new_thresholds}")

    def force_collection(self, generation: int | None = None) -> dict[str, Any]:
        """Force a garbage collection run.

        Args:
            generation: Specific generation to collect (None for full)

        Returns:
            Dictionary with collection results

        """
        start_time = time.time()
        objects_before = len(gc.get_objects())

        if generation is not None:
            collected = gc.collect(generation)
        else:
            collected = gc.collect()

        duration_ms = (time.time() - start_time) * 1000
        objects_after = len(gc.get_objects())

        result = {
            "collected_objects": collected,
            "objects_before": objects_before,
            "objects_after": objects_after,
            "objects_freed": objects_before - objects_after,
            "duration_ms": duration_ms,
            "generation": generation,
            "timestamp": time.time(),
        }

        self._gc_history.append(result)
        self._last_gc_time = time.time()

        logger.info(
            f"GC collected {collected} objects, freed {result['objects_freed']}, took {duration_ms:.2f}ms"
        )

        return result

    def analyze_and_optimize(self) -> dict[str, Any]:
        """Analyze memory and optimize GC if needed.

        Returns:
            Dictionary with analysis results and actions taken

        """
        gc_stats = self.get_gc_stats()
        tracked_objects = gc_stats["tracked_objects"]
        counts = gc_stats["counts"]

        analysis = {
            "tracked_objects": tracked_objects,
            "generation_counts": counts,
            "gc_recommended": False,
            "actions_taken": [],
        }

        # Check if gen0 count is high
        if counts["generation_0"] > gc_stats["thresholds"]["generation_0"] * 0.8:
            analysis["gc_recommended"] = True
            analysis["reason"] = "Generation 0 count approaching threshold"

        # Check if tracked objects are very high
        if tracked_objects > 100000:
            analysis["gc_recommended"] = True
            analysis["reason"] = "High number of tracked objects"

        # Check if garbage list has uncollectable objects
        if len(gc.garbage) > 0:
            analysis["uncollectable_objects"] = len(gc.garbage)
            logger.warning(f"Found {len(gc.garbage)} uncollectable objects")

        return analysis

    def get_optimization_suggestions(self) -> list[str]:
        """Get suggestions for GC optimization.

        Returns:
            List of optimization suggestions

        """
        suggestions = []
        gc_stats = self.get_gc_stats()

        # Check for high object count
        if gc_stats["tracked_objects"] > 50000:
            suggestions.append(
                "Consider reviewing object lifecycle - high tracked object count"
            )

        # Check for uncollectable objects
        if gc_stats["garbage_count"] > 0:
            suggestions.append(
                "Investigate uncollectable objects (circular references with __del__)"
            )

        # Check generation statistics
        stats = gc_stats["stats"]
        if stats:
            # Check for high collection times
            for i, gen_stats in enumerate(stats):
                if gen_stats.get("collected", 0) > 10000:
                    suggestions.append(
                        f"Generation {i} collecting many objects - consider object pooling"
                    )

        return suggestions

    def reset_thresholds(self) -> None:
        """Reset GC thresholds to original values."""
        gc.set_threshold(*self._original_thresholds)
        logger.info(f"GC thresholds reset to: {self._original_thresholds}")


class ResourceMonitor:
    """System resource monitoring with alerts and GC optimization.

    Provides comprehensive system resource monitoring including CPU,
    memory, disk, and process-specific metrics. Supports configurable
    thresholds and alert callbacks.

    Attributes:
        thresholds: Resource alert thresholds
        sample_interval: Seconds between samples

    Example:
        >>> monitor = ResourceMonitor()
        >>> await monitor.start()
        >>> snapshot = monitor.get_current_snapshot()
        >>> alerts = monitor.check_thresholds()

    """

    def __init__(
        self,
        thresholds: ResourceThresholds | None = None,
        sample_interval: int = 5,
        history_size: int = 100,
    ) -> None:
        """Initialize resource monitor.

        Args:
            thresholds: Alert thresholds (uses defaults if None)
            sample_interval: Seconds between automatic samples
            history_size: Number of snapshots to retain

        """
        self.thresholds = thresholds or ResourceThresholds()
        self.sample_interval = sample_interval
        self.history_size = history_size

        # Process handle
        self._process = psutil.Process(os.getpid())

        # GC optimizer
        self._gc_optimizer = GCOptimizer()

        # State
        self._is_running = False
        self._sample_task: asyncio.Task[None] | None = None
        self._history: deque[ResourceSnapshot] = deque(maxlen=history_size)
        self._alerts: deque[ResourceAlert] = deque(maxlen=100)
        self._alert_callbacks: list[Callable[[ResourceAlert], None]] = []

        # Cached values for CPU calculations
        self._last_cpu_sample = time.time()

        logger.info(
            "Resource monitor initialized",
            sample_interval=sample_interval,
            history_size=history_size,
        )

    async def start(self) -> None:
        """Start the resource monitor background task."""
        if self._is_running:
            return

        self._is_running = True
        self._sample_task = asyncio.create_task(self._sample_loop())
        logger.info("Resource monitor started")

    async def stop(self) -> None:
        """Stop the resource monitor."""
        if not self._is_running:
            return

        self._is_running = False

        if self._sample_task:
            self._sample_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sample_task

        logger.info("Resource monitor stopped")

    def get_current_snapshot(self) -> ResourceSnapshot:
        """Get current resource usage snapshot.

        Returns:
            ResourceSnapshot with current values

        """
        try:
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=0)
            cpu_count = psutil.cpu_count() or 1
            memory = psutil.virtual_memory()

            # Disk metrics (root partition)
            try:
                disk = psutil.disk_usage("/")
                disk_percent = disk.percent
                disk_used_gb = disk.used / (1024**3)
                disk_free_gb = disk.free / (1024**3)
            except (PermissionError, OSError):
                disk_percent = 0.0
                disk_used_gb = 0.0
                disk_free_gb = 0.0

            # Process-specific metrics
            try:
                process_cpu = self._process.cpu_percent(interval=0)
                process_memory = self._process.memory_info().rss / (1024**2)
                thread_count = self._process.num_threads()
                open_fds = (
                    self._process.num_fds() if hasattr(self._process, "num_fds") else 0
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process_cpu = 0.0
                process_memory = 0.0
                thread_count = 0
                open_fds = 0

            # GC metrics
            gc_stats = gc.get_stats()
            gc_collections = {
                "gen0": gc_stats[0].get("collections", 0),
                "gen1": gc_stats[1].get("collections", 0),
                "gen2": gc_stats[2].get("collections", 0),
            }

            return ResourceSnapshot(
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024**2),
                memory_available_mb=memory.available / (1024**2),
                memory_total_mb=memory.total / (1024**2),
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_free_gb=disk_free_gb,
                open_file_descriptors=open_fds,
                thread_count=thread_count,
                gc_objects=len(gc.get_objects()),
                gc_collections=gc_collections,
                process_cpu_percent=process_cpu,
                process_memory_mb=process_memory,
            )
        except Exception as e:
            logger.exception(f"Error getting resource snapshot: {e}")
            return ResourceSnapshot()

    def check_thresholds(
        self, snapshot: ResourceSnapshot | None = None
    ) -> list[ResourceAlert]:
        """Check resource thresholds and generate alerts.

        Args:
            snapshot: Snapshot to check (uses current if None)

        Returns:
            List of generated alerts

        """
        if snapshot is None:
            snapshot = self.get_current_snapshot()

        alerts = []

        # CPU checks
        if snapshot.cpu_percent >= self.thresholds.cpu_critical_percent:
            alerts.append(
                ResourceAlert(
                    alert_type=AlertType.CPU_HIGH,
                    level=ResourceLevel.CRITICAL,
                    message=f"CPU usage critical: {snapshot.cpu_percent:.1f}%",
                    current_value=snapshot.cpu_percent,
                    threshold_value=self.thresholds.cpu_critical_percent,
                )
            )
        elif snapshot.cpu_percent >= self.thresholds.cpu_warning_percent:
            alerts.append(
                ResourceAlert(
                    alert_type=AlertType.CPU_HIGH,
                    level=ResourceLevel.WARNING,
                    message=f"CPU usage high: {snapshot.cpu_percent:.1f}%",
                    current_value=snapshot.cpu_percent,
                    threshold_value=self.thresholds.cpu_warning_percent,
                )
            )

        # Memory checks
        if snapshot.memory_percent >= self.thresholds.memory_critical_percent:
            alerts.append(
                ResourceAlert(
                    alert_type=AlertType.MEMORY_CRITICAL,
                    level=ResourceLevel.CRITICAL,
                    message=f"Memory usage critical: {snapshot.memory_percent:.1f}%",
                    current_value=snapshot.memory_percent,
                    threshold_value=self.thresholds.memory_critical_percent,
                )
            )
        elif snapshot.memory_percent >= self.thresholds.memory_warning_percent:
            alerts.append(
                ResourceAlert(
                    alert_type=AlertType.MEMORY_HIGH,
                    level=ResourceLevel.WARNING,
                    message=f"Memory usage high: {snapshot.memory_percent:.1f}%",
                    current_value=snapshot.memory_percent,
                    threshold_value=self.thresholds.memory_warning_percent,
                )
            )

        # Disk checks
        if snapshot.disk_percent >= self.thresholds.disk_critical_percent:
            alerts.append(
                ResourceAlert(
                    alert_type=AlertType.DISK_CRITICAL,
                    level=ResourceLevel.CRITICAL,
                    message=f"Disk usage critical: {snapshot.disk_percent:.1f}%",
                    current_value=snapshot.disk_percent,
                    threshold_value=self.thresholds.disk_critical_percent,
                )
            )
        elif snapshot.disk_percent >= self.thresholds.disk_warning_percent:
            alerts.append(
                ResourceAlert(
                    alert_type=AlertType.DISK_HIGH,
                    level=ResourceLevel.WARNING,
                    message=f"Disk usage high: {snapshot.disk_percent:.1f}%",
                    current_value=snapshot.disk_percent,
                    threshold_value=self.thresholds.disk_warning_percent,
                )
            )

        # GC check
        if snapshot.process_memory_mb >= self.thresholds.gc_memory_threshold_mb:
            gc_analysis = self._gc_optimizer.analyze_and_optimize()
            if gc_analysis.get("gc_recommended"):
                alerts.append(
                    ResourceAlert(
                        alert_type=AlertType.GC_RECOMMENDED,
                        level=ResourceLevel.WARNING,
                        message=f"GC recommended: {gc_analysis.get('reason', 'high memory')}",
                        current_value=snapshot.process_memory_mb,
                        threshold_value=float(self.thresholds.gc_memory_threshold_mb),
                    )
                )

        return alerts

    def register_alert_callback(
        self, callback: Callable[[ResourceAlert], None]
    ) -> None:
        """Register a callback for resource alerts.

        Args:
            callback: Function to call when alerts are generated

        """
        self._alert_callbacks.append(callback)

    def _process_alerts(self, alerts: list[ResourceAlert]) -> None:
        """Process and distribute alerts.

        Args:
            alerts: List of alerts to process

        """
        for alert in alerts:
            self._alerts.append(alert)

            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.exception(f"Error in alert callback: {e}")

            # Log alerts
            if alert.level == ResourceLevel.CRITICAL:
                logger.error(alert.message)
            else:
                logger.warning(alert.message)

    async def _sample_loop(self) -> None:
        """Background sampling loop."""
        while self._is_running:
            try:
                # Take snapshot
                snapshot = self.get_current_snapshot()
                self._history.append(snapshot)

                # Check thresholds
                alerts = self.check_thresholds(snapshot)
                if alerts:
                    self._process_alerts(alerts)

                await asyncio.sleep(self.sample_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in sample loop: {e}")
                await asyncio.sleep(self.sample_interval)

    def get_history(self, limit: int | None = None) -> list[ResourceSnapshot]:
        """Get historical snapshots.

        Args:
            limit: Maximum number of snapshots to return

        Returns:
            List of snapshots in chronological order

        """
        snapshots = list(self._history)
        if limit:
            return snapshots[-limit:]
        return snapshots

    def get_recent_alerts(self, limit: int = 10) -> list[ResourceAlert]:
        """Get recent alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts

        """
        return list(self._alerts)[-limit:]

    def get_gc_optimizer(self) -> GCOptimizer:
        """Get the GC optimizer instance.

        Returns:
            GCOptimizer instance

        """
        return self._gc_optimizer

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of resource usage.

        Returns:
            Dictionary with resource summary

        """
        snapshot = self.get_current_snapshot()
        alerts = self.check_thresholds(snapshot)

        # Calculate trends from history
        if len(self._history) >= 2:
            recent = list(self._history)[-10:]
            cpu_trend = [s.cpu_percent for s in recent]
            memory_trend = [s.memory_percent for s in recent]
        else:
            cpu_trend = [snapshot.cpu_percent]
            memory_trend = [snapshot.memory_percent]

        return {
            "current": {
                "cpu_percent": snapshot.cpu_percent,
                "memory_percent": snapshot.memory_percent,
                "disk_percent": snapshot.disk_percent,
                "process_memory_mb": snapshot.process_memory_mb,
            },
            "trends": {
                "cpu_avg": sum(cpu_trend) / len(cpu_trend),
                "memory_avg": sum(memory_trend) / len(memory_trend),
            },
            "alerts": {
                "active_count": len(alerts),
                "critical_count": sum(
                    1 for a in alerts if a.level == ResourceLevel.CRITICAL
                ),
                "warning_count": sum(
                    1 for a in alerts if a.level == ResourceLevel.WARNING
                ),
            },
            "gc": {
                "objects": snapshot.gc_objects,
                "suggestions": self._gc_optimizer.get_optimization_suggestions(),
            },
            "thresholds": self.thresholds.model_dump(),
        }

    def force_gc(self, generation: int | None = None) -> dict[str, Any]:
        """Force garbage collection.

        Args:
            generation: Specific generation to collect

        Returns:
            GC collection results

        """
        return self._gc_optimizer.force_collection(generation)
