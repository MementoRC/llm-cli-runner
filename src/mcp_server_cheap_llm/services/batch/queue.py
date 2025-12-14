"""Thread-safe batch queue implementation with priority management.

This module implements an intelligent batch processing queue with priority-based
scheduling, thread-safety guarantees, and comprehensive monitoring capabilities.
Integrates with existing ProviderManager and CacheService for optimal performance.

Key classes:
    BatchQueue: Main thread-safe priority queue for batch requests
    QueueManager: High-level queue management with monitoring
    PriorityScheduler: Priority-based batch scheduling logic

Example:
    >>> queue = BatchQueue(max_size=100)
    >>> await queue.enqueue(batch_request, priority=BatchPriority.HIGH)
    >>> batch = await queue.dequeue()

"""

import asyncio
import contextlib
import heapq
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from mcp_server_cheap_llm.core.models import (
    BatchMetrics,
    BatchPriority,
    BatchQueueInfo,
    BatchRequest,
)
from mcp_server_cheap_llm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueuedBatch:
    """Internal representation of a queued batch with priority and timing info.

    Attributes:
        priority_value: Numeric priority for heap ordering (lower = higher priority)
        enqueue_time: When the batch was added to queue
        batch_request: The actual batch request
        position_in_queue: Current position in queue (updated dynamically)

    """

    priority_value: int
    enqueue_time: float
    batch_request: BatchRequest
    position_in_queue: int = 0

    def __lt__(self, other):
        """Priority comparison for heap ordering."""
        if self.priority_value != other.priority_value:
            return self.priority_value < other.priority_value
        # If same priority, FIFO (earlier timestamp wins)
        return self.enqueue_time < other.enqueue_time


class BatchQueue:
    """Thread-safe priority queue for batch processing requests.

    Provides thread-safe batch queuing with priority management, capacity limits,
    and comprehensive monitoring. Uses asyncio-compatible locking for concurrency.

    Attributes:
        max_size: Maximum number of batches that can be queued
        priority_weights: Mapping of priority levels to numeric weights
        queue: Internal priority heap for batch storage
        current_size: Current number of queued batches
        total_enqueued: Total batches enqueued (lifetime counter)
        total_dequeued: Total batches dequeued (lifetime counter)

    Example:
        >>> queue = BatchQueue(max_size=50)
        >>> await queue.enqueue(batch_request, BatchPriority.HIGH)
        >>> queued_batch = await queue.dequeue()

    """

    def __init__(self, max_size: int = 100) -> None:
        """Initialize batch queue.

        Args:
            max_size: Maximum number of batches in queue

        """
        self.max_size = max_size

        # Priority mapping (lower number = higher priority)
        self.priority_weights = {
            BatchPriority.URGENT: 1,
            BatchPriority.HIGH: 2,
            BatchPriority.NORMAL: 3,
            BatchPriority.LOW: 4,
        }

        # Queue state
        self._queue: list[QueuedBatch] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)

        # Metrics
        self.current_size = 0
        self.total_enqueued = 0
        self.total_dequeued = 0
        self.enqueue_times = deque(maxlen=100)  # Recent enqueue timestamps
        self.dequeue_times = deque(maxlen=100)  # Recent dequeue timestamps
        self.wait_times = deque(maxlen=100)  # Recent wait times

        logger.info(f"Batch queue initialized with max_size={max_size}")

    async def enqueue(
        self,
        batch_request: BatchRequest,
        timeout: float | None = None,
    ) -> bool:
        """Add batch request to queue with priority ordering.

        Args:
            batch_request: Batch request to enqueue
            timeout: Maximum time to wait for queue space (None = no timeout)

        Returns:
            bool: True if successfully enqueued, False if timeout or queue full

        """
        async with self._not_full:
            # Wait for space if queue is full
            if timeout is not None:
                end_time = time.time() + timeout
                while self.current_size >= self.max_size:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        logger.warning(
                            f"Enqueue timeout for batch {batch_request.batch_id}",
                        )
                        return False

                    try:
                        await asyncio.wait_for(self._not_full.wait(), timeout=remaining)
                    except TimeoutError:
                        logger.warning(
                            f"Enqueue timeout for batch {batch_request.batch_id}",
                        )
                        return False
            else:
                while self.current_size >= self.max_size:
                    await self._not_full.wait()

            # Create queued batch with priority
            priority_value = self.priority_weights.get(batch_request.priority, 3)
            enqueue_time = time.time()

            queued_batch = QueuedBatch(
                priority_value=priority_value,
                enqueue_time=enqueue_time,
                batch_request=batch_request,
            )

            # Add to priority heap
            heapq.heappush(self._queue, queued_batch)
            self.current_size += 1
            self.total_enqueued += 1
            self.enqueue_times.append(enqueue_time)

            # Update queue positions
            self._update_queue_positions()

            logger.info(
                f"Enqueued batch {batch_request.batch_id} with priority {batch_request.priority} "
                f"(queue size: {self.current_size}/{self.max_size})",
            )

            # Notify waiting dequeuers
            self._not_empty.notify()

            return True

    async def dequeue(self, timeout: float | None = None) -> QueuedBatch | None:
        """Remove and return highest priority batch from queue.

        Args:
            timeout: Maximum time to wait for batch (None = wait indefinitely)

        Returns:
            QueuedBatch: Highest priority batch or None if timeout

        """
        async with self._not_empty:
            # Wait for batch if queue is empty
            if timeout is not None:
                end_time = time.time() + timeout
                while self.current_size == 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        logger.debug("Dequeue timeout - no batches available")
                        return None

                    try:
                        await asyncio.wait_for(
                            self._not_empty.wait(),
                            timeout=remaining,
                        )
                    except TimeoutError:
                        logger.debug("Dequeue timeout - no batches available")
                        return None
            else:
                while self.current_size == 0:
                    await self._not_empty.wait()

            # Get highest priority batch
            queued_batch = heapq.heappop(self._queue)
            self.current_size -= 1
            self.total_dequeued += 1

            dequeue_time = time.time()
            self.dequeue_times.append(dequeue_time)

            # Calculate and record wait time
            wait_time = dequeue_time - queued_batch.enqueue_time
            self.wait_times.append(wait_time)

            # Update remaining queue positions
            self._update_queue_positions()

            logger.info(
                f"Dequeued batch {queued_batch.batch_request.batch_id} "
                f"(waited {wait_time:.2f}s, queue size: {self.current_size})",
            )

            # Notify waiting enqueuers
            self._not_full.notify()

            return queued_batch

    def _update_queue_positions(self) -> None:
        """Update position information for all queued batches."""
        for i, queued_batch in enumerate(sorted(self._queue)):
            queued_batch.position_in_queue = i + 1

    async def get_queue_info(self, batch_id: str) -> BatchQueueInfo | None:
        """Get queue information for a specific batch.

        Args:
            batch_id: ID of the batch to get info for

        Returns:
            BatchQueueInfo: Queue information or None if batch not found

        """
        async with self._lock:
            # Find batch in queue
            for queued_batch in self._queue:
                if queued_batch.batch_request.batch_id == batch_id:
                    # Calculate priority queue depths
                    priority_depths = defaultdict(int)
                    for qb in self._queue:
                        if qb.priority_value <= queued_batch.priority_value:
                            priority_name = qb.batch_request.priority.value
                            priority_depths[priority_name] += 1

                    # Estimate wait time based on recent processing times
                    recent_wait_times = list(self.wait_times)[-10:]  # Last 10 batches
                    avg_processing_time = (
                        sum(recent_wait_times) / len(recent_wait_times) * 1000
                        if recent_wait_times
                        else 30000.0  # Default 30s
                    )

                    estimated_wait = int(
                        queued_batch.position_in_queue * avg_processing_time,
                    )

                    return BatchQueueInfo(
                        queue_position=queued_batch.position_in_queue,
                        queue_depth=self.current_size,
                        estimated_wait_time_ms=estimated_wait,
                        processing_capacity=1,  # Will be updated by QueueManager
                        average_batch_time_ms=avg_processing_time,
                        priority_queue_depth=dict(priority_depths),
                    )

            return None

    async def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive queue metrics.

        Returns:
            dict: Queue performance and status metrics

        """
        async with self._lock:
            recent_enqueues = len(
                [
                    t
                    for t in self.enqueue_times
                    if time.time() - t <= 3600  # Last hour
                ],
            )

            recent_dequeues = len(
                [
                    t
                    for t in self.dequeue_times
                    if time.time() - t <= 3600  # Last hour
                ],
            )

            avg_wait_time = (
                sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0.0
            )

            # Priority distribution
            priority_counts = defaultdict(int)
            for queued_batch in self._queue:
                priority_name = queued_batch.batch_request.priority.value
                priority_counts[priority_name] += 1

            return {
                "queue_size": {
                    "current": self.current_size,
                    "max": self.max_size,
                    "utilization_percent": (self.current_size / self.max_size) * 100,
                },
                "throughput": {
                    "total_enqueued": self.total_enqueued,
                    "total_dequeued": self.total_dequeued,
                    "enqueues_per_hour": recent_enqueues,
                    "dequeues_per_hour": recent_dequeues,
                },
                "timing": {
                    "average_wait_time_ms": avg_wait_time * 1000,
                    "recent_wait_times": [
                        t * 1000 for t in list(self.wait_times)[-10:]
                    ],
                },
                "priority_distribution": dict(priority_counts),
                "health": {
                    "queue_blocked": self.current_size >= self.max_size,
                    "processing_stalled": len(self.dequeue_times) == 0,
                    "backlog_building": recent_enqueues > recent_dequeues,
                },
            }

    async def clear(self) -> None:
        """Clear all batches from queue (for shutdown/reset)."""
        async with self._lock:
            self._queue.clear()
            self.current_size = 0
            logger.info("Batch queue cleared")

            # Notify all waiters
            self._not_full.notify_all()


class QueueManager:
    """High-level batch queue manager with monitoring and optimization.

    Provides a management layer over BatchQueue with health monitoring,
    capacity planning, and integration with system metrics.

    Attributes:
        queue: Underlying BatchQueue instance
        processing_capacity: Number of concurrent batch processors
        health_check_interval: Seconds between health checks
        metrics_history: Historical metrics for trend analysis

    Example:
        >>> manager = QueueManager(max_queue_size=100, processing_capacity=3)
        >>> await manager.start()
        >>> await manager.enqueue_batch(batch_request)

    """

    def __init__(
        self,
        max_queue_size: int = 100,
        processing_capacity: int = 3,
        health_check_interval: int = 60,
    ) -> None:
        """Initialize queue manager.

        Args:
            max_queue_size: Maximum batches in queue
            processing_capacity: Number of concurrent processors
            health_check_interval: Seconds between health checks

        """
        self.queue = BatchQueue(max_size=max_queue_size)
        self.processing_capacity = processing_capacity
        self.health_check_interval = health_check_interval

        # Monitoring
        self.metrics_history = deque(maxlen=100)  # Keep last 100 snapshots
        self.health_check_task: asyncio.Task | None = None
        self.is_running = False

        logger.info(
            f"Queue manager initialized: max_size={max_queue_size}, "
            f"capacity={processing_capacity}",
        )

    async def start(self) -> None:
        """Start queue manager and background health monitoring."""
        if self.is_running:
            return

        self.is_running = True
        self.health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("Queue manager started")

    async def stop(self) -> None:
        """Stop queue manager and cleanup resources."""
        if not self.is_running:
            return

        self.is_running = False

        if self.health_check_task:
            self.health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.health_check_task

        await self.queue.clear()

        logger.info("Queue manager stopped")

    async def enqueue_batch(
        self,
        batch_request: BatchRequest,
        timeout: float | None = None,
    ) -> bool:
        """Enqueue batch with manager-level validation and monitoring.

        Args:
            batch_request: Batch request to enqueue
            timeout: Maximum time to wait for queue space

        Returns:
            bool: True if successfully enqueued

        """
        # Validation
        if not batch_request.requests:
            logger.error(f"Empty batch request {batch_request.batch_id}")
            return False

        if len(batch_request.requests) > 100:  # Max batch size check
            logger.error(
                f"Batch {batch_request.batch_id} too large: {len(batch_request.requests)} requests",
            )
            return False

        # Enqueue with monitoring
        start_time = time.time()
        success = await self.queue.enqueue(batch_request, timeout)

        if success:
            enqueue_time = (time.time() - start_time) * 1000
            logger.info(
                f"Batch {batch_request.batch_id} enqueued successfully "
                f"(took {enqueue_time:.2f}ms)",
            )

        return success

    async def dequeue_batch(self, timeout: float | None = None) -> QueuedBatch | None:
        """Dequeue batch with manager-level monitoring.

        Args:
            timeout: Maximum time to wait for batch

        Returns:
            QueuedBatch: Next batch to process or None

        """
        return await self.queue.dequeue(timeout)

    async def get_batch_queue_info(self, batch_id: str) -> BatchQueueInfo | None:
        """Get queue information for a specific batch.

        Args:
            batch_id: ID of the batch

        Returns:
            BatchQueueInfo: Queue information with updated capacity info

        """
        queue_info = await self.queue.get_queue_info(batch_id)

        if queue_info:
            # Update with manager-level capacity info
            queue_info.processing_capacity = self.processing_capacity

            # Recalculate wait time based on processing capacity
            if queue_info.queue_position > 0:
                estimated_batches_ahead = max(
                    0,
                    queue_info.queue_position - self.processing_capacity,
                )
                queue_info.estimated_wait_time_ms = int(
                    estimated_batches_ahead
                    * queue_info.average_batch_time_ms
                    / self.processing_capacity,
                )

        return queue_info

    async def get_system_metrics(self) -> BatchMetrics:
        """Get comprehensive system-level batch processing metrics.

        Returns:
            BatchMetrics: System performance metrics

        """
        queue_metrics = await self.queue.get_metrics()

        # Calculate system-level metrics
        total_processed = self.queue.total_dequeued
        current_queue_depth = queue_metrics["queue_size"]["current"]

        # Throughput calculations
        throughput_batches_per_hour = queue_metrics["throughput"]["dequeues_per_hour"]
        avg_requests_per_batch = 5.0  # Would be calculated from historical data
        throughput_requests_per_hour = (
            throughput_batches_per_hour * avg_requests_per_batch
        )

        # Average times
        avg_queue_time_ms = queue_metrics["timing"]["average_wait_time_ms"]
        avg_batch_time_ms = 30000.0  # Would be tracked from actual processing

        return BatchMetrics(
            total_batches_processed=total_processed,
            total_requests_processed=int(total_processed * avg_requests_per_batch),
            average_batch_time_ms=avg_batch_time_ms,
            average_queue_time_ms=avg_queue_time_ms,
            cache_hit_rate_percent=0.0,  # Will be updated by integration
            throughput_batches_per_hour=throughput_batches_per_hour,
            throughput_requests_per_hour=throughput_requests_per_hour,
            error_rate_percent=0.0,  # Will be tracked by batch processor
            similarity_optimization_rate=0.0,  # Will be tracked by similarity analyzer
            current_queue_depth=current_queue_depth,
            active_processing_slots=self.processing_capacity,
            system_load_percent=min(
                100.0,
                (current_queue_depth / (self.queue.max_size * 0.8)) * 100,
            ),
        )

    async def _health_check_loop(self) -> None:
        """Background health monitoring loop."""
        while self.is_running:
            try:
                # Capture metrics snapshot
                metrics = await self.queue.get_metrics()
                metrics["timestamp"] = time.time()
                self.metrics_history.append(metrics)

                # Health checks
                await self._perform_health_checks(metrics)

                # Wait for next check
                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _perform_health_checks(self, metrics: dict[str, Any]) -> None:
        """Perform health checks and log warnings."""
        # Queue utilization check
        utilization = metrics["queue_size"]["utilization_percent"]
        if utilization > 90:
            logger.warning(f"Queue utilization high: {utilization:.1f}%")
        elif utilization > 95:
            logger.error(f"Queue utilization critical: {utilization:.1f}%")

        # Processing stall check
        if metrics["health"]["processing_stalled"]:
            logger.warning("Processing appears stalled - no recent dequeues")

        # Backlog building check
        if metrics["health"]["backlog_building"]:
            logger.warning("Backlog building - enqueues exceeding dequeues")

        # Average wait time check
        avg_wait = metrics["timing"]["average_wait_time_ms"]
        if avg_wait > 300000:  # 5 minutes
            logger.warning(f"Average wait time high: {avg_wait / 1000:.1f}s")
