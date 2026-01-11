"""Main batch processing service integrating queue, similarity, and provider management.

This module provides the main BatchProcessor class that orchestrates the entire
batch processing pipeline, integrating with ProviderManager, CacheService, and
other system components for intelligent request processing.

Key classes:
    BatchProcessor: Main batch processing orchestrator
    ProcessingResult: Container for batch processing results
    BatchWorker: Individual batch processing worker

Example:
    >>> processor = BatchProcessor(provider_manager, cache_service)
    >>> await processor.start()
    >>> result = await processor.process_batch(batch_request)

"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

from mcp_server_llm_cli_runner.cache.service import CacheService
from mcp_server_llm_cli_runner.core.models import (
    BatchMetrics,
    BatchRequest,
    BatchResponse,
    BatchStatus,
    LLMRequest,
    LLMResponse,
)
from mcp_server_llm_cli_runner.providers.manager import ProviderManager
from mcp_server_llm_cli_runner.utils.logging import get_logger

from .queue import QueuedBatch, QueueManager
from .similarity import SimilarityAnalyzer

logger = get_logger(__name__)


class ProcessingResult:
    """Container for batch processing results with detailed metrics.

    Attributes:
        batch_id: ID of the processed batch
        status: Final processing status
        responses: List of successful responses
        failed_indices: Indices of failed requests
        processing_time_ms: Total processing time
        cache_hits: Number of cache hits
        similarity_groups: Identified similarity groups
        error_details: Error information if processing failed

    """

    def __init__(self, batch_id: str) -> None:
        """Initialize processing result.

        Args:
            batch_id: ID of the batch being processed

        """
        self.batch_id = batch_id
        self.status = BatchStatus.PROCESSING
        self.responses: list[LLMResponse] = []
        self.failed_indices: list[int] = []
        self.processing_time_ms = 0
        self.cache_hits = 0
        self.similarity_groups: list[Any] = []
        self.error_details: str | None = None
        self.start_time = time.time()

    def mark_completed(self) -> None:
        """Mark processing as completed and calculate timing."""
        self.status = BatchStatus.COMPLETED
        self.processing_time_ms = int((time.time() - self.start_time) * 1000)

    def mark_failed(self, error: str) -> None:
        """Mark processing as failed with error details."""
        self.status = BatchStatus.FAILED
        self.error_details = error
        self.processing_time_ms = int((time.time() - self.start_time) * 1000)

    def add_response(self, response: LLMResponse, is_cache_hit: bool = False) -> None:
        """Add successful response to results."""
        self.responses.append(response)
        if is_cache_hit:
            self.cache_hits += 1

    def add_failure(self, index: int) -> None:
        """Add failed request index."""
        self.failed_indices.append(index)


class BatchWorker:
    """Individual worker for processing batches.

    Handles the actual processing of individual batches including:
    - Request routing through ProviderManager
    - Cache integration
    - Error handling and retries
    - Response aggregation
    """

    def __init__(
        self,
        worker_id: str,
        provider_manager: ProviderManager,
        cache_service: CacheService | None = None,
        similarity_analyzer: SimilarityAnalyzer | None = None,
    ) -> None:
        """Initialize batch worker.

        Args:
            worker_id: Unique identifier for this worker
            provider_manager: Provider management service
            cache_service: Optional cache service
            similarity_analyzer: Optional similarity analysis service

        """
        self.worker_id = worker_id
        self.provider_manager = provider_manager
        self.cache_service = cache_service
        self.similarity_analyzer = similarity_analyzer

        # Worker statistics
        self.batches_processed = 0
        self.total_processing_time = 0.0
        self.total_requests_processed = 0
        self.is_busy = False

        logger.info(f"Batch worker {worker_id} initialized")

    async def process_batch(self, queued_batch: QueuedBatch) -> ProcessingResult:
        """Process a single batch request.

        Args:
            queued_batch: Queued batch to process

        Returns:
            ProcessingResult: Processing results and metrics

        """
        self.is_busy = True
        batch_request = queued_batch.batch_request
        result = ProcessingResult(batch_request.batch_id)

        try:
            logger.info(
                f"Worker {self.worker_id} starting batch {batch_request.batch_id} "
                f"({len(batch_request.requests)} requests)",
            )

            # Similarity analysis and optimization
            if self.similarity_analyzer:
                (
                    optimized_batch,
                    similarity_groups,
                ) = await self.similarity_analyzer.optimize_batch(batch_request)
                result.similarity_groups = similarity_groups
                batch_request = optimized_batch

            # Process individual requests
            await self._process_requests(batch_request, result)

            # Mark as completed
            result.mark_completed()

            # Update worker statistics
            self.batches_processed += 1
            self.total_processing_time += result.processing_time_ms / 1000.0
            self.total_requests_processed += len(batch_request.requests)

            logger.info(
                f"Worker {self.worker_id} completed batch {batch_request.batch_id} "
                f"in {result.processing_time_ms}ms "
                f"({len(result.responses)} successful, {len(result.failed_indices)} failed, "
                f"{result.cache_hits} cache hits)",
            )

        except Exception as e:
            error_msg = f"Batch processing failed: {e!s}"
            logger.exception(f"Worker {self.worker_id} error: {error_msg}")
            result.mark_failed(error_msg)

        finally:
            self.is_busy = False

        return result

    async def _process_requests(
        self,
        batch_request: BatchRequest,
        result: ProcessingResult,
    ) -> None:
        """Process individual requests in the batch.

        Args:
            batch_request: Batch request to process
            result: Result container to populate

        """
        # Determine parallelism level
        max_parallel = min(
            batch_request.max_parallel,
            len(batch_request.requests),
            10,  # Global max parallel limit
        )

        # Create semaphore for parallel processing
        semaphore = asyncio.Semaphore(max_parallel)

        # Create tasks for parallel processing
        tasks = []
        for i, request in enumerate(batch_request.requests):
            task = asyncio.create_task(
                self._process_single_request(request, i, semaphore, result),
            )
            tasks.append(task)

        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_single_request(
        self,
        request: LLMRequest,
        index: int,
        semaphore: asyncio.Semaphore,
        result: ProcessingResult,
    ) -> None:
        """Process a single request within the batch.

        Args:
            request: Individual LLM request
            index: Index of request in batch
            semaphore: Semaphore for controlling parallelism
            result: Result container

        """
        async with semaphore:
            try:
                # Check cache first if available
                if self.cache_service:
                    cached_response = await self.cache_service.get_cached_response(
                        request,
                    )
                    if cached_response:
                        result.add_response(cached_response, is_cache_hit=True)
                        logger.debug(f"Cache hit for request {index}")
                        return

                # Route request through provider manager
                response = await self.provider_manager.route_request(request)

                # Cache the response if cache service is available
                if self.cache_service:
                    await self.cache_service.cache_response(request, response)

                result.add_response(response, is_cache_hit=False)
                logger.debug(f"Successfully processed request {index}")

            except Exception as e:
                logger.exception(f"Failed to process request {index}: {e!s}")
                result.add_failure(index)

    def get_worker_stats(self) -> dict[str, Any]:
        """Get worker performance statistics.

        Returns:
            Dict[str, Any]: Worker performance metrics

        """
        avg_batch_time = (
            self.total_processing_time / self.batches_processed
            if self.batches_processed > 0
            else 0.0
        )

        avg_requests_per_batch = (
            self.total_requests_processed / self.batches_processed
            if self.batches_processed > 0
            else 0.0
        )

        return {
            "worker_id": self.worker_id,
            "is_busy": self.is_busy,
            "batches_processed": self.batches_processed,
            "total_requests_processed": self.total_requests_processed,
            "total_processing_time_seconds": self.total_processing_time,
            "average_batch_time_seconds": avg_batch_time,
            "average_requests_per_batch": avg_requests_per_batch,
        }


class BatchProcessor:
    """Main batch processing orchestrator.

    Coordinates the entire batch processing pipeline including:
    - Queue management
    - Worker pool management
    - Similarity analysis integration
    - Provider and cache service integration
    - Metrics collection and reporting

    Attributes:
        provider_manager: Provider management service
        cache_service: Optional cache service
        queue_manager: Batch queue manager
        similarity_analyzer: Similarity analysis service
        workers: Pool of batch processing workers

    Example:
        >>> processor = BatchProcessor(provider_manager, cache_service)
        >>> await processor.start()
        >>> response = await processor.submit_batch(batch_request)

    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        cache_service: CacheService | None = None,
        max_queue_size: int = 100,
        num_workers: int = 3,
        similarity_threshold: float = 0.7,
    ) -> None:
        """Initialize batch processor.

        Args:
            provider_manager: Provider management service
            cache_service: Optional cache service for optimization
            max_queue_size: Maximum batches in queue
            num_workers: Number of processing workers
            similarity_threshold: Threshold for similarity analysis

        """
        self.provider_manager = provider_manager
        self.cache_service = cache_service

        # Initialize components
        self.queue_manager = QueueManager(
            max_queue_size=max_queue_size,
            processing_capacity=num_workers,
        )

        self.similarity_analyzer = SimilarityAnalyzer(
            similarity_threshold=similarity_threshold,
        )

        # Worker management
        self.num_workers = num_workers
        self.workers: list[BatchWorker] = []
        self.worker_tasks: list[asyncio.Task] = []

        # State
        self.is_running = False
        self.processing_results: dict[str, BatchResponse] = {}

        # Metrics
        self.total_batches_submitted = 0
        self.total_batches_completed = 0
        self.total_batches_failed = 0

        logger.info(
            f"Batch processor initialized: {num_workers} workers, "
            f"queue_size={max_queue_size}, similarity_threshold={similarity_threshold}",
        )

    async def start(self) -> None:
        """Start the batch processor and all components."""
        if self.is_running:
            return

        logger.info("Starting batch processor...")

        # Start queue manager
        await self.queue_manager.start()

        # Initialize workers
        for i in range(self.num_workers):
            worker = BatchWorker(
                worker_id=f"worker-{i}",
                provider_manager=self.provider_manager,
                cache_service=self.cache_service,
                similarity_analyzer=self.similarity_analyzer,
            )
            self.workers.append(worker)

            # Start worker task
            task = asyncio.create_task(self._worker_loop(worker))
            self.worker_tasks.append(task)

        self.is_running = True
        logger.info(f"Batch processor started with {len(self.workers)} workers")

    async def stop(self) -> None:
        """Stop the batch processor and cleanup resources."""
        if not self.is_running:
            return

        logger.info("Stopping batch processor...")

        self.is_running = False

        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        # Stop queue manager
        await self.queue_manager.stop()

        # Clear state
        self.workers.clear()
        self.worker_tasks.clear()

        logger.info("Batch processor stopped")

    async def submit_batch(self, batch_request: BatchRequest) -> str:
        """Submit batch request for processing.

        Args:
            batch_request: Batch request to process

        Returns:
            str: Batch ID for tracking

        Raises:
            RuntimeError: If processor is not running
            ValueError: If batch request is invalid

        """
        if not self.is_running:
            msg = "Batch processor is not running"
            raise RuntimeError(msg)

        # Validate batch request
        if not batch_request.requests:
            msg = "Batch request cannot be empty"
            raise ValueError(msg)

        if len(batch_request.requests) > 100:
            msg = "Batch size exceeds maximum limit (100)"
            raise ValueError(msg)

        # Enqueue batch
        success = await self.queue_manager.enqueue_batch(batch_request, timeout=30.0)

        if not success:
            msg = "Failed to enqueue batch - queue may be full"
            raise RuntimeError(msg)

        self.total_batches_submitted += 1

        logger.info(
            f"Submitted batch {batch_request.batch_id} "
            f"({len(batch_request.requests)} requests, priority: {batch_request.priority})",
        )

        return batch_request.batch_id

    async def get_batch_status(self, batch_id: str) -> BatchResponse | None:
        """Get status of a batch request.

        Args:
            batch_id: ID of the batch to check

        Returns:
            BatchResponse: Current batch status or None if not found

        """
        # Check completed results first
        if batch_id in self.processing_results:
            return self.processing_results[batch_id]

        # Check if batch is still in queue
        queue_info = await self.queue_manager.get_batch_queue_info(batch_id)
        if queue_info:
            return BatchResponse(
                batch_id=batch_id,
                status=BatchStatus.QUEUED,
                queue_time_ms=int(time.time() * 1000)
                - int(queue_info.estimated_wait_time_ms),
                metadata={
                    "queue_position": queue_info.queue_position,
                    "estimated_wait_time_ms": queue_info.estimated_wait_time_ms,
                },
            )

        # Check if any worker is currently processing this batch
        for worker in self.workers:
            if worker.is_busy:
                # Note: In a real implementation, we'd track which batch each worker is processing
                return BatchResponse(
                    batch_id=batch_id,
                    status=BatchStatus.PROCESSING,
                    metadata={"worker_id": worker.worker_id},
                )

        return None

    async def get_system_metrics(self) -> BatchMetrics:
        """Get comprehensive system metrics.

        Returns:
            BatchMetrics: System performance metrics

        """
        # Get queue metrics
        queue_metrics = await self.queue_manager.get_system_metrics()

        # Calculate worker metrics
        sum(worker.total_processing_time for worker in self.workers)

        sum(worker.total_requests_processed for worker in self.workers)

        # Get similarity analyzer stats
        similarity_stats = self.similarity_analyzer.get_cache_stats()

        # Update queue metrics with additional processor data
        queue_metrics.total_batches_processed = self.total_batches_completed
        queue_metrics.error_rate_percent = (
            self.total_batches_failed / max(1, self.total_batches_submitted)
        ) * 100
        queue_metrics.similarity_optimization_rate = similarity_stats[
            "average_optimization_potential"
        ]

        # Calculate cache hit rate if cache service is available
        if self.cache_service:
            cache_stats = await self.cache_service.get_cache_stats()
            if "metrics" in cache_stats and "overview" in cache_stats["metrics"]:
                queue_metrics.cache_hit_rate_percent = cache_stats["metrics"][
                    "overview"
                ]["hit_rate"]

        return queue_metrics

    async def _worker_loop(self, worker: BatchWorker) -> None:
        """Main processing loop for a worker.

        Args:
            worker: Worker instance to run

        """
        logger.info(f"Starting worker loop for {worker.worker_id}")

        try:
            while self.is_running:
                try:
                    # Get next batch from queue (with timeout)
                    queued_batch = await self.queue_manager.dequeue_batch(timeout=5.0)

                    if queued_batch is None:
                        continue  # Timeout, try again

                    # Process the batch
                    result = await worker.process_batch(queued_batch)

                    # Convert to BatchResponse and store
                    batch_response = self._create_batch_response(queued_batch, result)

                    self.processing_results[result.batch_id] = batch_response

                    # Update global metrics
                    if result.status == BatchStatus.COMPLETED:
                        self.total_batches_completed += 1
                    else:
                        self.total_batches_failed += 1

                    # Clean up old results (keep last 1000)
                    if len(self.processing_results) > 1000:
                        oldest_keys = list(self.processing_results.keys())[:100]
                        for key in oldest_keys:
                            del self.processing_results[key]

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception(f"Worker {worker.worker_id} error: {e!s}")
                    await asyncio.sleep(1.0)  # Brief pause before retrying

        except asyncio.CancelledError:
            pass

        logger.info(f"Worker loop ended for {worker.worker_id}")

    def _create_batch_response(
        self,
        queued_batch: QueuedBatch,
        result: ProcessingResult,
    ) -> BatchResponse:
        """Create BatchResponse from processing result.

        Args:
            queued_batch: Original queued batch
            result: Processing result

        Returns:
            BatchResponse: Complete batch response

        """
        # Calculate queue time
        queue_time_ms = int((result.start_time - queued_batch.enqueue_time) * 1000)

        # Calculate total tokens and cost
        total_tokens = sum(response.tokens_used for response in result.responses)
        total_cost = 0.0  # Would be calculated based on provider pricing

        return BatchResponse(
            batch_id=result.batch_id,
            status=result.status,
            responses=result.responses,
            failed_requests=result.failed_indices,
            processing_time_ms=result.processing_time_ms,
            queue_time_ms=queue_time_ms,
            cache_hits=result.cache_hits,
            total_tokens_used=total_tokens,
            total_cost_usd=total_cost,
            similarity_groups=[
                group.to_dict() if hasattr(group, "to_dict") else group
                for group in result.similarity_groups
            ],
            completed_at=datetime.now(UTC),
            error_details=result.error_details,
            metadata={
                "worker_processed": True,
                "similarity_analyzed": len(result.similarity_groups) > 0,
                "optimization_applied": any(
                    group.get("optimization_potential", 0) > 0
                    for group in result.similarity_groups
                    if isinstance(group, dict)
                ),
            },
        )
