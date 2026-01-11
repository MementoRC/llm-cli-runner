"""Unit tests for queue backpressure functionality - TDD approach."""

import asyncio

import pytest

from mcp_server_llm_cli_runner.core.models import BatchPriority, BatchRequest
from mcp_server_llm_cli_runner.services.batch.queue import (
    BackpressureAction,
    BackpressureConfig,
    BackpressureController,
    BackpressureLevel,
    BackpressureState,
    BatchQueue,
    QueueMetrics,
)


class TestBackpressureLevel:
    """Test suite for BackpressureLevel enum."""

    def test_backpressure_levels(self):
        """Test all backpressure levels exist."""
        assert BackpressureLevel.NONE.value == "none"
        assert BackpressureLevel.LOW.value == "low"
        assert BackpressureLevel.MEDIUM.value == "medium"
        assert BackpressureLevel.HIGH.value == "high"
        assert BackpressureLevel.CRITICAL.value == "critical"


class TestBackpressureAction:
    """Test suite for BackpressureAction enum."""

    def test_backpressure_actions(self):
        """Test all backpressure actions exist."""
        assert BackpressureAction.ACCEPT.value == "accept"
        assert BackpressureAction.DELAY.value == "delay"
        assert BackpressureAction.REJECT_LOW_PRIORITY.value == "reject_low_priority"
        assert BackpressureAction.REJECT_ALL.value == "reject_all"


class TestBackpressureConfig:
    """Test suite for BackpressureConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = BackpressureConfig()

        assert config.enabled is True
        assert config.low_threshold_percent == 50.0
        assert config.medium_threshold_percent == 70.0
        assert config.high_threshold_percent == 85.0
        assert config.critical_threshold_percent == 95.0
        assert config.delay_base_ms == 100
        assert config.delay_multiplier == 2.0

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = BackpressureConfig(
            enabled=False,
            low_threshold_percent=40.0,
            high_threshold_percent=80.0,
            delay_base_ms=200,
        )

        assert config.enabled is False
        assert config.low_threshold_percent == 40.0
        assert config.delay_base_ms == 200

    def test_config_adaptive_sizing(self):
        """Test adaptive sizing configuration."""
        config = BackpressureConfig(
            adaptive_sizing=True,
            min_size=100,
            max_size=1000,
            size_adjustment_rate=0.2,
        )

        assert config.adaptive_sizing is True
        assert config.min_size == 100
        assert config.max_size == 1000
        assert config.size_adjustment_rate == 0.2


class TestBackpressureState:
    """Test suite for BackpressureState."""

    def test_state_defaults(self):
        """Test default state values."""
        state = BackpressureState()

        assert state.level == BackpressureLevel.NONE
        assert state.action == BackpressureAction.ACCEPT
        assert state.queue_utilization == 0.0
        assert state.delay_ms == 0
        assert state.rejected_count == 0
        assert state.delayed_count == 0

    def test_state_custom_values(self):
        """Test state with custom values."""
        state = BackpressureState(
            level=BackpressureLevel.HIGH,
            action=BackpressureAction.DELAY,
            queue_utilization=85.0,
            delay_ms=400,
            rejected_count=5,
            delayed_count=10,
        )

        assert state.level == BackpressureLevel.HIGH
        assert state.queue_utilization == 85.0


class TestBackpressureController:
    """Test suite for BackpressureController."""

    def test_controller_initialization(self):
        """Test BackpressureController initialization."""
        controller = BackpressureController()
        assert controller is not None
        assert controller.config.enabled is True

    def test_controller_disabled(self):
        """Test controller when disabled."""
        config = BackpressureConfig(enabled=False)
        controller = BackpressureController(config)

        action, delay = controller.evaluate(
            current_size=95,
            max_size=100,
            priority=BatchPriority.LOW,
        )

        assert action == BackpressureAction.ACCEPT
        assert delay == 0

    def test_evaluate_no_backpressure(self):
        """Test evaluation with low utilization."""
        controller = BackpressureController()

        action, delay = controller.evaluate(
            current_size=30,
            max_size=100,
            priority=BatchPriority.NORMAL,
        )

        assert action == BackpressureAction.ACCEPT
        assert delay == 0

    def test_evaluate_low_backpressure(self):
        """Test evaluation with low backpressure."""
        controller = BackpressureController()

        action, delay = controller.evaluate(
            current_size=55,  # 55% utilization
            max_size=100,
            priority=BatchPriority.NORMAL,
        )

        assert action == BackpressureAction.DELAY
        assert delay > 0

    def test_evaluate_low_backpressure_urgent_priority(self):
        """Test that urgent requests bypass low backpressure."""
        controller = BackpressureController()

        action, delay = controller.evaluate(
            current_size=55,
            max_size=100,
            priority=BatchPriority.URGENT,
        )

        assert action == BackpressureAction.ACCEPT
        assert delay == 0

    def test_evaluate_medium_backpressure(self):
        """Test evaluation with medium backpressure."""
        controller = BackpressureController()

        action, delay = controller.evaluate(
            current_size=75,  # 75% utilization
            max_size=100,
            priority=BatchPriority.NORMAL,
        )

        assert action == BackpressureAction.DELAY
        state = controller.get_state()
        assert state.level == BackpressureLevel.MEDIUM

    def test_evaluate_high_backpressure_low_priority(self):
        """Test that low priority requests are rejected under high backpressure."""
        controller = BackpressureController()

        action, delay = controller.evaluate(
            current_size=88,  # 88% utilization
            max_size=100,
            priority=BatchPriority.LOW,
        )

        assert action == BackpressureAction.REJECT_LOW_PRIORITY

    def test_evaluate_high_backpressure_normal_priority(self):
        """Test that normal priority requests are delayed under high backpressure."""
        controller = BackpressureController()

        action, delay = controller.evaluate(
            current_size=88,
            max_size=100,
            priority=BatchPriority.NORMAL,
        )

        assert action == BackpressureAction.DELAY
        assert delay > 0

    def test_evaluate_critical_backpressure(self):
        """Test evaluation with critical backpressure."""
        controller = BackpressureController()

        action, delay = controller.evaluate(
            current_size=96,  # 96% utilization
            max_size=100,
            priority=BatchPriority.NORMAL,
        )

        assert action == BackpressureAction.REJECT_ALL

    def test_evaluate_critical_backpressure_urgent(self):
        """Test that urgent requests are delayed (not rejected) at critical."""
        controller = BackpressureController()

        action, delay = controller.evaluate(
            current_size=96,
            max_size=100,
            priority=BatchPriority.URGENT,
        )

        assert action == BackpressureAction.DELAY
        assert delay > 0

    def test_get_state(self):
        """Test getting current state."""
        controller = BackpressureController()
        controller.evaluate(current_size=50, max_size=100)

        state = controller.get_state()

        assert isinstance(state, BackpressureState)
        assert state.queue_utilization == 50.0

    def test_adaptive_sizing_increase(self):
        """Test adaptive sizing increases queue size under high load."""
        config = BackpressureConfig(
            adaptive_sizing=True,
            min_size=50,
            max_size=200,
            size_adjustment_rate=0.1,
        )
        controller = BackpressureController(config)

        # Simulate high utilization history
        for _ in range(60):
            controller.evaluate(current_size=90, max_size=100)

        new_size = controller.get_adaptive_size(current_max=100)

        assert new_size > 100

    def test_adaptive_sizing_decrease(self):
        """Test adaptive sizing decreases queue size under low load."""
        config = BackpressureConfig(
            adaptive_sizing=True,
            min_size=50,
            max_size=200,
            size_adjustment_rate=0.1,
        )
        controller = BackpressureController(config)

        # Simulate low utilization history
        for _ in range(60):
            controller.evaluate(current_size=20, max_size=100)

        new_size = controller.get_adaptive_size(current_max=100)

        assert new_size < 100

    def test_reset_metrics(self):
        """Test resetting metrics."""
        controller = BackpressureController()

        # Generate some metrics
        controller.evaluate(current_size=96, max_size=100, priority=BatchPriority.LOW)

        controller.reset_metrics()

        state = controller.get_state()
        # After reset, the controller's internal counters should be reset
        # but state is updated on next evaluate
        assert isinstance(state, BackpressureState)


class TestQueueMetrics:
    """Test suite for QueueMetrics model."""

    def test_queue_metrics_defaults(self):
        """Test QueueMetrics default values."""
        metrics = QueueMetrics()

        assert metrics.current_size == 0
        assert metrics.max_size == 100
        assert metrics.utilization_percent == 0.0
        assert metrics.total_enqueued == 0
        assert metrics.total_rejected == 0
        assert isinstance(metrics.backpressure_state, BackpressureState)

    def test_queue_metrics_custom_values(self):
        """Test QueueMetrics with custom values."""
        bp_state = BackpressureState(level=BackpressureLevel.MEDIUM)
        metrics = QueueMetrics(
            current_size=50,
            max_size=100,
            utilization_percent=50.0,
            total_enqueued=1000,
            total_rejected=10,
            total_delayed=50,
            backpressure_state=bp_state,
        )

        assert metrics.current_size == 50
        assert metrics.total_rejected == 10
        assert metrics.backpressure_state.level == BackpressureLevel.MEDIUM


class TestBatchQueueBackpressure:
    """Test suite for BatchQueue backpressure integration."""

    @pytest.fixture
    def sample_request(self):
        """Create a sample batch request."""
        return BatchRequest(
            batch_id="test-batch-1",
            requests=[{"prompt": "test"}],
            priority=BatchPriority.NORMAL,
        )

    def test_queue_with_backpressure_disabled(self, sample_request):
        """Test queue with backpressure disabled."""
        queue = BatchQueue(max_size=100, enable_backpressure=False)
        assert queue.backpressure_controller.config.enabled is False

    def test_queue_with_backpressure_enabled(self, sample_request):
        """Test queue with backpressure enabled."""
        queue = BatchQueue(max_size=100, enable_backpressure=True)
        assert queue.backpressure_controller.config.enabled is True

    def test_queue_with_custom_backpressure_config(self, sample_request):
        """Test queue with custom backpressure config."""
        config = BackpressureConfig(
            low_threshold_percent=40.0,
            high_threshold_percent=80.0,
        )
        queue = BatchQueue(max_size=100, backpressure_config=config)

        assert queue.backpressure_controller.config.low_threshold_percent == 40.0

    @pytest.mark.asyncio
    async def test_queue_enqueue_under_backpressure(self, sample_request):
        """Test enqueue behavior under backpressure."""
        queue = BatchQueue(max_size=100, enable_backpressure=True)

        # Should succeed with low utilization
        success = await queue.enqueue(sample_request)
        assert success is True

    @pytest.mark.asyncio
    async def test_queue_rejection_critical_backpressure(self):
        """Test that requests are rejected under critical backpressure."""
        config = BackpressureConfig(
            critical_threshold_percent=50.0,  # Low threshold for testing
        )
        queue = BatchQueue(max_size=10, backpressure_config=config)

        # Fill queue to trigger critical backpressure
        for i in range(6):  # 60% utilization - above our 50% critical threshold
            request = BatchRequest(
                batch_id=f"test-batch-{i}",
                requests=[{"prompt": "test"}],
                priority=BatchPriority.URGENT,  # Urgent bypasses rejection initially
            )
            await queue.enqueue(request)

        # Now try to add a low priority request - should be rejected
        low_priority_request = BatchRequest(
            batch_id="test-batch-low",
            requests=[{"prompt": "test"}],
            priority=BatchPriority.LOW,
        )
        success = await queue.enqueue(low_priority_request)

        assert success is False
        assert queue.total_rejected > 0

    @pytest.mark.asyncio
    async def test_get_detailed_metrics(self):
        """Test getting detailed metrics with backpressure state."""
        queue = BatchQueue(max_size=100, enable_backpressure=True)

        request = BatchRequest(
            batch_id="test-batch-1",
            requests=[{"prompt": "test"}],
            priority=BatchPriority.NORMAL,
        )
        await queue.enqueue(request)

        metrics = queue.get_detailed_metrics()

        assert isinstance(metrics, QueueMetrics)
        assert metrics.current_size == 1
        assert metrics.total_enqueued == 1

    @pytest.mark.asyncio
    async def test_metrics_include_backpressure_state(self):
        """Test that queue metrics include backpressure state."""
        queue = BatchQueue(max_size=100, enable_backpressure=True)

        metrics = await queue.get_metrics()

        assert "backpressure" in metrics
        assert "level" in metrics["backpressure"]
        assert "action" in metrics["backpressure"]

    def test_adjust_max_size_disabled(self):
        """Test adjust_max_size when adaptive sizing is disabled."""
        queue = BatchQueue(max_size=100, enable_backpressure=True)

        # Should return same size when adaptive sizing is disabled
        new_size = queue.adjust_max_size()
        assert new_size == 100

    def test_adjust_max_size_enabled(self):
        """Test adjust_max_size when adaptive sizing is enabled."""
        config = BackpressureConfig(
            adaptive_sizing=True,
            min_size=50,
            max_size=200,
        )
        queue = BatchQueue(max_size=100, backpressure_config=config)

        # Without history, should return same size
        new_size = queue.adjust_max_size()
        assert new_size == 100
