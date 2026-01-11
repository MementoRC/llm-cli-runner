"""Test batch processing import resolution."""

import pytest


class TestBatchImports:
    """Test all batch processing imports work correctly."""

    def test_batch_processor_imports(self):
        """Test that BatchProcessor can import all required models."""
        from mcp_server_llm_cli_runner.core.models import (
            BatchMetrics,
            BatchRequest,
            BatchResponse,
            BatchStatus,
        )
        from mcp_server_llm_cli_runner.services.batch.processor import BatchProcessor

        # Verify BatchStatus has QUEUED attribute
        assert hasattr(BatchStatus, "QUEUED")
        assert BatchStatus.QUEUED == "queued"

        # Verify other expected batch statuses exist
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.PROCESSING == "processing"
        assert BatchStatus.COMPLETED == "completed"
        assert BatchStatus.FAILED == "failed"
        assert BatchStatus.CANCELLED == "cancelled"

    def test_batch_queue_imports(self):
        """Test that QueueManager can import all required models."""
        from mcp_server_llm_cli_runner.core.models import (
            BatchMetrics,
            BatchPriority,
            BatchQueueInfo,
            BatchRequest,
        )
        from mcp_server_llm_cli_runner.services.batch.queue import QueueManager

        # Verify BatchPriority enum works
        assert BatchPriority.URGENT == "urgent"
        assert BatchPriority.HIGH == "high"
        assert BatchPriority.NORMAL == "normal"
        assert BatchPriority.LOW == "low"

    def test_batch_similarity_imports(self):
        """Test that SimilarityAnalyzer can import all required models."""
        from mcp_server_llm_cli_runner.core.models import BatchRequest
        from mcp_server_llm_cli_runner.services.batch.similarity import (
            SimilarityAnalyzer,
        )

        # Verify BatchRequest model exists and has expected fields in __annotations__
        assert "batch_id" in BatchRequest.__annotations__
        assert "requests" in BatchRequest.__annotations__
        assert "priority" in BatchRequest.__annotations__
        assert "similarity_threshold" in BatchRequest.__annotations__

    def test_all_batch_models_available(self):
        """Test that all batch-related models are properly defined."""
        from mcp_server_llm_cli_runner.core.models import (
            BatchMetrics,
            BatchPriority,
            BatchQueueInfo,
            BatchRequest,
            BatchResponse,
            BatchStatus,
        )

        # Verify models can be instantiated with basic data
        status = BatchStatus.QUEUED
        priority = BatchPriority.NORMAL

        # These should not raise errors
        assert isinstance(status, str)
        assert isinstance(priority, str)

        # Verify the models exist as classes
        assert isinstance(BatchMetrics, type)
        assert isinstance(BatchQueueInfo, type)
        assert isinstance(BatchRequest, type)
        assert isinstance(BatchResponse, type)
