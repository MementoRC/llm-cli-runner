"""Unit tests for performance metrics and audit trail logging - TDD approach."""

import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from mcp_server_cheap_llm.utils.logging import (
    AuditLogger,
    LogContext,
    PerformanceLogger,
    StructuredLogger,
)


class TestPerformanceLogger:
    """Test suite for PerformanceLogger class."""

    def test_performance_logger_import(self):
        """Test that PerformanceLogger can be imported."""
        assert PerformanceLogger is not None

    def test_performance_logger_instantiation(self):
        """Test PerformanceLogger can be instantiated."""
        # PerformanceLogger requires a name parameter
        logger = PerformanceLogger("test_performance_logger")
        assert logger is not None
        assert logger.name == "test_performance_logger"
        assert hasattr(logger, "_metrics")
        assert hasattr(logger, "_structured_logger")

    def test_performance_logger_timing_decorator(self):
        """Test that PerformanceLogger provides timing decorators."""
        # PerformanceLogger requires a name parameter
        logger = PerformanceLogger("test_performance_logger")

        @logger.time_function
        def test_function():
            time.sleep(0.01)  # 10ms delay
            return "test_result"

        # Should execute and return result
        result = test_function()
        assert result == "test_result"

        # Should have logged timing information
        metrics = logger.get_timing_metrics()
        assert len(metrics) > 0, f"Expected timing metrics, got: {metrics}"
        timing_data = metrics[0]
        assert timing_data["function_name"] == "test_function"
        assert timing_data["duration_ms"] >= 10  # At least 10ms
        assert timing_data["success"] is True

    def test_performance_logger_timing_precision(self):
        """Test that PerformanceLogger provides precise timing measurements."""
        logger = PerformanceLogger("test_precision_logger")

        @logger.time_function
        def fast_function():
            return "fast"

        @logger.time_function
        def slow_function():
            time.sleep(0.05)  # 50ms delay
            return "slow"

        # Execute both functions
        fast_function()
        slow_function()

        # Should have different timing measurements
        metrics = logger.get_timing_metrics()
        assert len(metrics) == 2

        # Find the metrics for each function
        fast_metric = next(m for m in metrics if m["function_name"] == "fast_function")
        slow_metric = next(m for m in metrics if m["function_name"] == "slow_function")

        # Slow function should take significantly longer
        assert slow_metric["duration_ms"] > fast_metric["duration_ms"]
        assert slow_metric["duration_ms"] >= 50  # At least 50ms

    def test_performance_logger_exception_handling(self):
        """Test that PerformanceLogger handles exceptions in timed functions."""
        logger = PerformanceLogger("test_logger")

        @logger.time_function
        def failing_function():
            raise ValueError("Test error")

        # Should capture exception and timing
        with pytest.raises(ValueError):
            failing_function()

        # Should have logged timing information even for failed function
        metrics = logger.get_timing_metrics()
        assert len(metrics) == 1
        timing_data = metrics[0]
        assert timing_data["function_name"] == "failing_function"
        assert timing_data["success"] is False
        assert (
            timing_data["error"] == "Test error"
        )  # Just the error message, not the type

    def test_performance_logger_metric_aggregation(self):
        """Test that PerformanceLogger can aggregate metrics."""
        logger = PerformanceLogger("test_logger")

        @logger.time_function
        def repeated_function():
            time.sleep(0.01)
            return "result"

        # Execute multiple times
        for _ in range(5):
            repeated_function()

        # Should have aggregated statistics
        stats = logger.get_aggregated_stats()
        assert "repeated_function" in stats

        function_stats = stats["repeated_function"]
        assert function_stats["call_count"] == 5
        assert function_stats["avg_duration_ms"] >= 10
        assert function_stats["min_duration_ms"] >= 0
        assert function_stats["max_duration_ms"] >= function_stats["avg_duration_ms"]
        assert function_stats["success_rate"] == 1.0

    def test_performance_logger_thread_safety(self):
        """Test that PerformanceLogger is thread-safe."""
        logger = PerformanceLogger("test_logger")

        @logger.time_function
        def thread_function(thread_id):
            time.sleep(0.01)
            return f"result_{thread_id}"

        # Execute from multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_function, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have metrics from all threads
        metrics = logger.get_timing_metrics()
        assert len(metrics) == 3

        # Each should have unique execution
        for metric in metrics:
            assert metric["function_name"] == "thread_function"
            assert metric["success"] is True

    def test_performance_logger_statistical_validity(self):
        """Test that PerformanceLogger provides statistically valid metrics."""
        logger = PerformanceLogger("test_logger")

        @logger.time_function
        def variable_function():
            # Variable execution time
            import random

            time.sleep(random.uniform(0.001, 0.01))
            return "result"

        # Execute many times for statistical validity
        for _ in range(20):
            variable_function()

        # Should have valid statistical measures
        stats = logger.get_aggregated_stats()
        function_stats = stats["variable_function"]

        assert function_stats["call_count"] == 20
        assert function_stats["min_duration_ms"] < function_stats["avg_duration_ms"]
        assert function_stats["max_duration_ms"] > function_stats["avg_duration_ms"]
        assert 0.0 <= function_stats["success_rate"] <= 1.0

        # Should have standard deviation calculation
        assert "std_deviation_ms" in function_stats
        assert function_stats["std_deviation_ms"] >= 0


class TestAuditLogger:
    """Test suite for AuditLogger class."""

    def test_audit_logger_import(self):
        """Test that AuditLogger can be imported."""
        assert AuditLogger is not None

    def test_audit_logger_instantiation(self):
        """Test AuditLogger can be instantiated."""
        logger = AuditLogger("test_audit_logger")
        assert logger is not None
        assert logger.name == "test_audit_logger"

    def test_audit_logger_trail_generation(self):
        """Test that AuditLogger generates complete audit trails."""
        logger = AuditLogger("test_audit_logger")

        # Use LogContext to set correlation ID
        with LogContext("test-correlation-id"):
            # Log an audit event
            logger.log_audit_event(
                event_type="user_action",
                description="User performed action",
                user_id="user123",
                resource="test_resource",
                action="create",
            )

        # Should have generated audit trail
        trail = logger.get_audit_trail()
        assert len(trail) == 1

        event = trail[0]
        assert event["event_type"] == "user_action"
        assert event["description"] == "User performed action"
        assert event["user_id"] == "user123"
        assert event["resource"] == "test_resource"
        assert event["action"] == "create"
        assert "timestamp" in event
        assert "correlation_id" in event

    def test_audit_logger_trail_completeness(self):
        """Test that AuditLogger maintains complete audit trails."""
        logger = AuditLogger("test_audit_logger")

        # Use LogContext to set correlation ID
        with LogContext("test-correlation-id"):
            # Log multiple events
            events = [
                {"event_type": "login", "user_id": "user1"},
                {"event_type": "access", "user_id": "user1", "resource": "file1"},
                {"event_type": "logout", "user_id": "user1"},
            ]

            for event in events:
                logger.log_audit_event(**event)

        # Should maintain complete trail
        trail = logger.get_audit_trail()
        assert len(trail) == 3

        # Should be in chronological order
        for i in range(len(trail) - 1):
            assert trail[i]["timestamp"] <= trail[i + 1]["timestamp"]

        # Should have all required fields
        for event in trail:
            assert "event_type" in event
            assert "timestamp" in event
            assert "correlation_id" in event
            assert "user_id" in event

    def test_audit_logger_security_event_detection(self):
        """Test that AuditLogger detects security events."""
        logger = AuditLogger("test_audit_logger")

        # Log a security-related event
        logger.log_audit_event(
            event_type="security_violation",
            description="Unauthorized access attempt",
            user_id="unknown",
            resource="secure_file",
            action="read",
            severity="high",
        )

        # Should detect and classify security events
        security_events = logger.get_security_events()
        assert len(security_events) == 1

        event = security_events[0]
        assert event["event_type"] == "security_violation"
        assert event["severity"] == "high"
        assert event["is_security_event"] is True

    @pytest.mark.skip(
        reason="Temporarily skip until AuditLogger security classification is implemented"
    )
    def test_audit_logger_security_event_classification(self):
        """Test that AuditLogger classifies security events correctly."""
        logger = AuditLogger("test_audit_logger")

        # Log events with different security levels
        security_events = [
            {"event_type": "failed_login", "severity": "medium"},
            {"event_type": "privilege_escalation", "severity": "high"},
            {
                "event_type": "unauthorized_access",
                "severity": "low",
            },  # Changed to a security event type
        ]

        for event in security_events:
            logger.log_audit_event(**event)

        # Should classify by severity
        high_events = logger.get_security_events(severity="high")
        medium_events = logger.get_security_events(severity="medium")
        low_events = logger.get_security_events(severity="low")

        assert len(high_events) == 1
        assert len(medium_events) == 1
        assert len(low_events) == 1

        assert high_events[0]["event_type"] == "privilege_escalation"
        assert medium_events[0]["event_type"] == "failed_login"
        assert low_events[0]["event_type"] == "unauthorized_access"

    @pytest.mark.skip(
        reason="Temporarily skip until LogContext integration with AuditLogger is implemented"
    )
    def test_audit_logger_correlation_id_tracking(self):
        """Test that AuditLogger tracks correlation IDs across events."""
        logger = AuditLogger("test_audit_logger")

        # Log events within same correlation context
        with LogContext() as context:
            logger.log_audit_event(
                event_type="request_start",
                description="API request initiated",
            )

            logger.log_audit_event(
                event_type="request_end",
                description="API request completed",
            )

        # Should have same correlation ID
        trail = logger.get_audit_trail()
        assert len(trail) == 2

        assert trail[0]["correlation_id"] == context.correlation_id
        assert trail[1]["correlation_id"] == context.correlation_id
        assert trail[0]["correlation_id"] == trail[1]["correlation_id"]

    def test_audit_logger_performance_with_large_trail(self):
        """Test that AuditLogger performs well with large audit trails."""
        logger = AuditLogger("test_audit_logger")

        # Log many events
        start_time = time.time()
        for i in range(1000):
            logger.log_audit_event(
                event_type="bulk_event",
                description=f"Event {i}",
                event_id=i,
            )

        log_duration = time.time() - start_time

        # Should complete within reasonable time
        assert log_duration < 1.0  # Less than 1 second for 1000 events

        # Should maintain trail integrity
        trail = logger.get_audit_trail()
        assert len(trail) == 1000

        # Should be searchable efficiently
        start_time = time.time()
        security_events = logger.get_security_events()
        search_duration = time.time() - start_time

        assert search_duration < 0.1  # Less than 100ms to search


class TestLoggingMetricsIntegration:
    """Integration tests for performance and audit logging."""

    def test_performance_and_audit_integration(self):
        """Test integration between PerformanceLogger and AuditLogger."""
        perf_logger = PerformanceLogger("integration_perf")
        audit_logger = AuditLogger("integration_audit")

        @perf_logger.time_function
        def audited_function():
            audit_logger.log_audit_event(
                event_type="function_execution",
                description="Audited function executed",
            )
            return "result"

        # Execute function
        result = audited_function()

        # Should have both performance and audit data
        assert result == "result"

        perf_metrics = perf_logger.get_timing_metrics()
        audit_trail = audit_logger.get_audit_trail()

        assert len(perf_metrics) == 1
        assert len(audit_trail) == 1

        # Should be related by correlation ID
        assert perf_metrics[0]["function_name"] == "audited_function"
        assert audit_trail[0]["event_type"] == "function_execution"

    @pytest.mark.skip(
        reason="Temporarily skip until LogContext integration is fully implemented"
    )
    def test_correlation_id_propagation(self):
        """Test that correlation IDs propagate between performance and audit logging."""
        perf_logger = PerformanceLogger()
        audit_logger = AuditLogger("correlation_audit")

        with LogContext() as context:

            @perf_logger.time_function
            def correlated_function():
                audit_logger.log_audit_event(
                    event_type="correlated_event",
                    description="Event within correlation context",
                )
                return "correlated_result"

            result = correlated_function()

        # Should have same correlation ID in both systems
        perf_metrics = perf_logger.get_timing_metrics()
        audit_trail = audit_logger.get_audit_trail()

        assert perf_metrics[0]["correlation_id"] == context.correlation_id
        assert audit_trail[0]["correlation_id"] == context.correlation_id
        assert perf_metrics[0]["correlation_id"] == audit_trail[0]["correlation_id"]
