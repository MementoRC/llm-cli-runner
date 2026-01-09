"""Thread-safe batch queue implementation with priority management and backpressure.

This module implements an intelligent batch processing queue with priority-based
scheduling, thread-safety guarantees, backpressure handling, and comprehensive
monitoring capabilities. Integrates with existing ProviderManager and CacheService
for optimal performance.

Key classes:
    BatchQueue: Main thread-safe priority queue for batch requests
    QueueManager: High-level queue management with monitoring
    BackpressureController: Manages queue backpressure and adaptive sizing
    QueueMetrics: Detailed queue metrics including backpressure state

Example:
    >>> queue = BatchQueue(max_size=100, enable_backpressure=True)
    >>> await queue.enqueue(batch_request, priority=BatchPriority.HIGH)
    >>> batch = await queue.dequeue()

"""

import asyncio
import contextlib
import heapq
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from mcp_server_llm_cli_runner.core.models import (
    BatchMetrics,
    BatchPriority,
    BatchQueueInfo,
    BatchRequest,
)
from mcp_server_llm_cli_runner.utils.logging import get_logger

logger = get_logger(__name__)


class BackpressureLevel(str, Enum):
    """Backpressure severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BackpressureAction(str, Enum):
    """Actions to take under backpressure."""

    ACCEPT = "accept"  # Accept the request normally
    DELAY = "delay"  # Accept with delay
    REJECT_LOW_PRIORITY = "reject_low_priority"  # Reject low priority requests
    REJECT_ALL = "reject_all"  # Reject all new requests


class BackpressureConfig(BaseModel):
    """Configuration for backpressure handling.

    Attributes:
        enabled: Whether backpressure is enabled
        low_threshold_percent: Queue utilization for low backpressure
        medium_threshold_percent: Queue utilization for medium backpressure
        high_threshold_percent: Queue utilization for high backpressure
        critical_threshold_percent: Queue utilization for critical backpressure
        delay_base_ms: Base delay for delayed requests
        delay_multiplier: Multiplier for delay based on level
        adaptive_sizing: Whether to use adaptive queue sizing
        min_size: Minimum queue size for adaptive sizing
        max_size: Maximum queue size for adaptive sizing
        size_adjustment_rate: Rate of size adjustment

    """

    enabled: bool = Field(default=True)
    low_threshold_percent: float = Field(default=50.0, ge=0.0, le=100.0)
    medium_threshold_percent: float = Field(default=70.0, ge=0.0, le=100.0)
    high_threshold_percent: float = Field(default=85.0, ge=0.0, le=100.0)
    critical_threshold_percent: float = Field(default=95.0, ge=0.0, le=100.0)
    delay_base_ms: int = Field(default=100, ge=0, le=10000)
    delay_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    adaptive_sizing: bool = Field(default=False)
    min_size: int = Field(default=50, ge=10, le=1000)
    max_size: int = Field(default=500, ge=50, le=10000)
    size_adjustment_rate: float = Field(default=0.1, ge=0.01, le=0.5)


class BackpressureState(BaseModel):
    """Current backpressure state.

    Attributes:
        level: Current backpressure level
        action: Recommended action for new requests
        queue_utilization: Current queue utilization percentage
        delay_ms: Delay to apply if action is DELAY
        rejected_count: Number of requests rejected due to backpressure
        delayed_count: Number of requests delayed due to backpressure
        current_queue_size: Current effective queue size

    """

    level: BackpressureLevel = BackpressureLevel.NONE
    action: BackpressureAction = BackpressureAction.ACCEPT
    queue_utilization: float = Field(default=0.0, ge=0.0, le=100.0)
    delay_ms: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    delayed_count: int = Field(default=0, ge=0)
    current_queue_size: int = Field(default=100, ge=1)


class BackpressureController:
    """Controls backpressure behavior for the queue.

    Monitors queue utilization and determines appropriate backpressure
    responses including delays, rejections, and adaptive sizing.

    Attributes:
        config: Backpressure configuration
        state: Current backpressure state

    Example:
        >>> controller = BackpressureController(config)
        >>> action = controller.evaluate(current_size=80, max_size=100)
        >>> if action == BackpressureAction.REJECT_ALL:
        ...     raise QueueFullError()

    """

    def __init__(self, config: BackpressureConfig | None = None) -> None:
        """Initialize backpressure controller.

        Args:
            config: Backpressure configuration

        """
        self.config = config or BackpressureConfig()
        self._state = BackpressureState(current_queue_size=self.config.max_size)
        self._lock = asyncio.Lock()

        # Metrics
        self._rejected_count = 0
        self._delayed_count = 0

        # Adaptive sizing history
        self._utilization_history: deque[float] = deque(maxlen=60)  # Last minute

        logger.info("Backpressure controller initialized", enabled=self.config.enabled)

    def evaluate(
        self,
        current_size: int,
        max_size: int,
        priority: BatchPriority = BatchPriority.NORMAL,
    ) -> tuple[BackpressureAction, int]:
        """Evaluate backpressure for a new request.

        Args:
            current_size: Current queue size
            max_size: Maximum queue size
            priority: Priority of the incoming request

        Returns:
            Tuple of (action to take, delay in ms if applicable)

        """
        if not self.config.enabled:
            return (BackpressureAction.ACCEPT, 0)

        utilization = (current_size / max_size) * 100 if max_size > 0 else 0
        self._utilization_history.append(utilization)

        # Determine backpressure level
        if utilization >= self.config.critical_threshold_percent:
            level = BackpressureLevel.CRITICAL
        elif utilization >= self.config.high_threshold_percent:
            level = BackpressureLevel.HIGH
        elif utilization >= self.config.medium_threshold_percent:
            level = BackpressureLevel.MEDIUM
        elif utilization >= self.config.low_threshold_percent:
            level = BackpressureLevel.LOW
        else:
            level = BackpressureLevel.NONE

        # Determine action based on level and priority
        action, delay = self._determine_action(level, priority)

        # Update state
        self._state = BackpressureState(
            level=level,
            action=action,
            queue_utilization=utilization,
            delay_ms=delay,
            rejected_count=self._rejected_count,
            delayed_count=self._delayed_count,
            current_queue_size=max_size,
        )

        return (action, delay)

    def _determine_action(
        self, level: BackpressureLevel, priority: BatchPriority
    ) -> tuple[BackpressureAction, int]:
        """Determine the action to take based on backpressure level and priority.

        Args:
            level: Current backpressure level
            priority: Request priority

        Returns:
            Tuple of (action, delay_ms)

        """
        if level == BackpressureLevel.NONE:
            return (BackpressureAction.ACCEPT, 0)

        if level == BackpressureLevel.LOW:
            # Slight delay for normal/low priority
            if priority in (BatchPriority.NORMAL, BatchPriority.LOW):
                delay = self.config.delay_base_ms
                return (BackpressureAction.DELAY, delay)
            return (BackpressureAction.ACCEPT, 0)

        if level == BackpressureLevel.MEDIUM:
            # Delay for all but urgent, longer delays for lower priority
            if priority == BatchPriority.URGENT:
                return (BackpressureAction.ACCEPT, 0)
            elif priority == BatchPriority.HIGH:
                delay = self.config.delay_base_ms
                return (BackpressureAction.DELAY, delay)
            else:
                delay = int(self.config.delay_base_ms * self.config.delay_multiplier)
                return (BackpressureAction.DELAY, delay)

        if level == BackpressureLevel.HIGH:
            # Reject low priority, delay others
            if priority == BatchPriority.LOW:
                self._rejected_count += 1
                return (BackpressureAction.REJECT_LOW_PRIORITY, 0)
            elif priority == BatchPriority.URGENT:
                delay = self.config.delay_base_ms
                return (BackpressureAction.DELAY, delay)
            else:
                delay = int(
                    self.config.delay_base_ms * (self.config.delay_multiplier**2)
                )
                return (BackpressureAction.DELAY, delay)

        if level == BackpressureLevel.CRITICAL:
            # Reject all except urgent
            if priority == BatchPriority.URGENT:
                delay = int(
                    self.config.delay_base_ms * (self.config.delay_multiplier**2)
                )
                self._delayed_count += 1
                return (BackpressureAction.DELAY, delay)
            self._rejected_count += 1
            return (BackpressureAction.REJECT_ALL, 0)

        return (BackpressureAction.ACCEPT, 0)

    def get_state(self) -> BackpressureState:
        """Get current backpressure state.

        Returns:
            Current BackpressureState

        """
        return self._state.model_copy()

    def get_adaptive_size(self, current_max: int) -> int:
        """Calculate adaptive queue size based on utilization history.

        Args:
            current_max: Current maximum queue size

        Returns:
            Recommended new maximum size

        """
        if not self.config.adaptive_sizing or not self._utilization_history:
            return current_max

        avg_utilization = sum(self._utilization_history) / len(
            self._utilization_history
        )

        # If consistently high utilization, increase size
        if avg_utilization > 80:
            new_size = int(current_max * (1 + self.config.size_adjustment_rate))
            return min(new_size, self.config.max_size)

        # If consistently low utilization, decrease size
        if avg_utilization < 30:
            new_size = int(current_max * (1 - self.config.size_adjustment_rate))
            return max(new_size, self.config.min_size)

        return current_max

    def reset_metrics(self) -> None:
        """Reset backpressure metrics."""
        self._rejected_count = 0
        self._delayed_count = 0
        self._utilization_history.clear()


class QueueMetrics(BaseModel):
    """Detailed queue metrics including backpressure state.

    Attributes:
        current_size: Current queue size
        max_size: Maximum queue size
        utilization_percent: Queue utilization as percentage
        total_enqueued: Total batches enqueued
        total_dequeued: Total batches dequeued
        total_rejected: Total batches rejected (backpressure)
        total_delayed: Total batches delayed (backpressure)
        average_wait_time_ms: Average wait time in queue
        throughput_per_minute: Batches processed per minute
        backpressure_state: Current backpressure state

    """

    current_size: int = Field(default=0, ge=0)
    max_size: int = Field(default=100, ge=1)
    utilization_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    total_enqueued: int = Field(default=0, ge=0)
    total_dequeued: int = Field(default=0, ge=0)
    total_rejected: int = Field(default=0, ge=0)
    total_delayed: int = Field(default=0, ge=0)
    average_wait_time_ms: float = Field(default=0.0, ge=0.0)
    throughput_per_minute: float = Field(default=0.0, ge=0.0)
    backpressure_state: BackpressureState = Field(default_factory=BackpressureState)


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
    """Thread-safe priority queue for batch processing requests with backpressure.

    Provides thread-safe batch queuing with priority management, capacity limits,
    backpressure handling, and comprehensive monitoring. Uses asyncio-compatible
    locking for concurrency.

    Attributes:
        max_size: Maximum number of batches that can be queued
        priority_weights: Mapping of priority levels to numeric weights
        queue: Internal priority heap for batch storage
        current_size: Current number of queued batches
        total_enqueued: Total batches enqueued (lifetime counter)
        total_dequeued: Total batches dequeued (lifetime counter)
        backpressure_controller: Controller for backpressure handling

    Example:
        >>> queue = BatchQueue(max_size=50, enable_backpressure=True)
        >>> await queue.enqueue(batch_request, BatchPriority.HIGH)
        >>> queued_batch = await queue.dequeue()

    """

    def __init__(
        self,
        max_size: int = 100,
        enable_backpressure: bool = True,
        backpressure_config: BackpressureConfig | None = None,
    ) -> None:
        """Initialize batch queue.

        Args:
            max_size: Maximum number of batches in queue
            enable_backpressure: Whether to enable backpressure handling
            backpressure_config: Configuration for backpressure behavior

        """
        self.max_size = max_size
        self._initial_max_size = max_size

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

        # Backpressure handling
        bp_config = backpressure_config or BackpressureConfig(
            enabled=enable_backpressure
        )
        self.backpressure_controller = BackpressureController(bp_config)

        # Metrics
        self.current_size = 0
        self.total_enqueued = 0
        self.total_dequeued = 0
        self.total_rejected = 0
        self.total_delayed = 0
        self.enqueue_times = deque(maxlen=100)  # Recent enqueue timestamps
        self.dequeue_times = deque(maxlen=100)  # Recent dequeue timestamps
        self.wait_times = deque(maxlen=100)  # Recent wait times

        logger.info(
            f"Batch queue initialized with max_size={max_size}, "
            f"backpressure={'enabled' if enable_backpressure else 'disabled'}"
        )

    async def enqueue(
        self,
        batch_request: BatchRequest,
        timeout: float | None = None,
    ) -> bool:
        """Add batch request to queue with priority ordering and backpressure.

        Args:
            batch_request: Batch request to enqueue
            timeout: Maximum time to wait for queue space (None = no timeout)

        Returns:
            bool: True if successfully enqueued, False if rejected/timeout

        """
        # Evaluate backpressure before acquiring lock
        action, delay_ms = self.backpressure_controller.evaluate(
            self.current_size,
            self.max_size,
            batch_request.priority,
        )

        # Handle backpressure rejection
        if action in (
            BackpressureAction.REJECT_LOW_PRIORITY,
            BackpressureAction.REJECT_ALL,
        ):
            self.total_rejected += 1
            logger.warning(
                f"Batch {batch_request.batch_id} rejected due to backpressure "
                f"(action={action.value}, queue={self.current_size}/{self.max_size})",
            )
            return False

        # Apply delay if needed
        if action == BackpressureAction.DELAY and delay_ms > 0:
            self.total_delayed += 1
            logger.debug(
                f"Delaying batch {batch_request.batch_id} by {delay_ms}ms due to backpressure",
            )
            await asyncio.sleep(delay_ms / 1000.0)

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
        """Get comprehensive queue metrics including backpressure state.

        Returns:
            dict: Queue performance, status, and backpressure metrics

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

            # Get backpressure state
            bp_state = self.backpressure_controller.get_state()

            return {
                "queue_size": {
                    "current": self.current_size,
                    "max": self.max_size,
                    "utilization_percent": (self.current_size / self.max_size) * 100,
                },
                "throughput": {
                    "total_enqueued": self.total_enqueued,
                    "total_dequeued": self.total_dequeued,
                    "total_rejected": self.total_rejected,
                    "total_delayed": self.total_delayed,
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
                "backpressure": {
                    "level": bp_state.level.value,
                    "action": bp_state.action.value,
                    "utilization": bp_state.queue_utilization,
                    "delay_ms": bp_state.delay_ms,
                    "rejected_count": bp_state.rejected_count,
                    "delayed_count": bp_state.delayed_count,
                },
            }

    def get_detailed_metrics(self) -> QueueMetrics:
        """Get detailed queue metrics as a Pydantic model.

        Returns:
            QueueMetrics with all queue statistics

        """
        # Calculate throughput
        recent_dequeues = len([t for t in self.dequeue_times if time.time() - t <= 60])

        avg_wait_time = (
            sum(self.wait_times) / len(self.wait_times) * 1000
            if self.wait_times
            else 0.0
        )

        return QueueMetrics(
            current_size=self.current_size,
            max_size=self.max_size,
            utilization_percent=(self.current_size / self.max_size) * 100
            if self.max_size > 0
            else 0,
            total_enqueued=self.total_enqueued,
            total_dequeued=self.total_dequeued,
            total_rejected=self.total_rejected,
            total_delayed=self.total_delayed,
            average_wait_time_ms=avg_wait_time,
            throughput_per_minute=recent_dequeues,
            backpressure_state=self.backpressure_controller.get_state(),
        )

    def adjust_max_size(self) -> int:
        """Adjust queue max size based on adaptive sizing.

        Returns:
            New maximum size (unchanged if adaptive sizing disabled)

        """
        new_size = self.backpressure_controller.get_adaptive_size(self.max_size)
        if new_size != self.max_size:
            old_size = self.max_size
            self.max_size = new_size
            logger.info(f"Queue max size adjusted: {old_size} -> {new_size}")
        return self.max_size

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
