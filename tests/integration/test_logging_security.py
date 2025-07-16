"""Integration tests for security logging and error handling - TDD approach.

RED: These tests are designed to fail first, then be made to pass.
This follows the TDD methodology for security-critical functionality.
"""

import json
import logging
import threading
import time
from unittest.mock import Mock, patch
from uuid import UUID

import pytest

from mcp_server_cheap_llm.utils.errors import (
    CheapLLMError,
    ConfigurationError,
    ProviderError,
    SecurityError,
    ValidationError,
)
from mcp_server_cheap_llm.utils.logging import LogContext, StructuredLogger


class TestSecurityLoggerIntegration:
    """Test suite for SecurityLogger integration with error handling."""

    def test_security_logger_import(self):
        """Test that SecurityLogger can be imported.

        RED: This will fail until SecurityLogger is implemented.
        """
        from mcp_server_cheap_llm.utils.logging import SecurityLogger

        assert SecurityLogger is not None

    def test_security_logger_instantiation(self):
        """Test SecurityLogger instantiation with default configuration.

        RED: This will fail until SecurityLogger is implemented.
        """
        from mcp_server_cheap_llm.utils.logging import SecurityLogger

        security_logger = SecurityLogger("test_security")
        assert security_logger.name == "test_security"
        assert hasattr(security_logger, "log_security_event")
        assert hasattr(security_logger, "monitor_error_rate")
        assert hasattr(security_logger, "detect_sensitive_data")

    def test_security_event_logging_with_error_integration(self):
        """Test security event logging with integrated error handling.

        RED: This will fail until SecurityLogger error integration is implemented.
        """
        from mcp_server_cheap_llm.utils.logging import SecurityLogger

        security_logger = SecurityLogger("test_security")

        # Test logging security event with error context
        test_error = SecurityError("Unauthorized access detected", error_code="SEC001")

        with LogContext() as context:
            security_logger.log_security_event(
                event_type="unauthorized_access",
                description="Failed authentication attempt",
                error=test_error,
                source="api_endpoint",
                severity="high",
            )

        # SecurityLogger should have captured the event with error details
        assert hasattr(security_logger, "_events")
        assert len(security_logger._events) > 0

        event = security_logger._events[-1]
        assert event["event_type"] == "unauthorized_access"
        assert event["error_code"] == "SEC001"
        assert event["severity"] == "high"
        assert "correlation_id" in event

    def test_sensitive_data_filtering_in_error_logs(self):
        """Test that sensitive data is filtered from error logs.

        RED: This will fail until sensitive data filtering is implemented.
        """
        from mcp_server_cheap_llm.utils.logging import SecurityLogger

        security_logger = SecurityLogger("test_security")

        # Test error with sensitive data
        sensitive_context = {
            "api_key": "sk-1234567890abcdef",
            "password": "secret123",
            "token": "bearer_token_xyz",
            "user_id": "user_123",  # This should NOT be filtered
            "request_id": "req_456",  # This should NOT be filtered
        }

        test_error = ProviderError(
            "API call failed",
            provider="test_provider",
            error_code="PROV001",
            context=sensitive_context,
        )

        with LogContext() as context:
            security_logger.log_error_with_filtering(test_error)

        # Check that sensitive data was filtered (values should be [FILTERED])
        logged_event = security_logger._events[-1]
        error_context = logged_event.get("error_context", {})
        assert error_context.get("api_key") == "[FILTERED]"
        assert error_context.get("password") == "[FILTERED]"
        assert error_context.get("token") == "[FILTERED]"
        assert error_context.get("user_id") == "user_123"  # Should be preserved
        assert error_context.get("request_id") == "req_456"  # Should be preserved

    def test_error_rate_monitoring_and_threshold_detection(self):
        """Test error rate monitoring with configurable thresholds.

        RED: This will fail until error rate monitoring is implemented.
        """
        from mcp_server_cheap_llm.utils.logging import SecurityLogger

        # Configure SecurityLogger with threshold
        security_logger = SecurityLogger(
            "test_security",
            error_rate_threshold=5,  # 5 errors per minute
            threshold_window_minutes=1,
        )

        # Simulate multiple errors within threshold window
        for i in range(3):
            error = ProviderError(
                f"Error {i}", provider="test_provider", error_code=f"PROV{i:03d}"
            )
            security_logger.monitor_error_rate(error)

        # Should not trigger threshold yet
        assert not security_logger.is_threshold_exceeded()

        # Add more errors to exceed threshold
        for i in range(3, 7):
            error = ProviderError(
                f"Error {i}", provider="test_provider", error_code=f"PROV{i:03d}"
            )
            security_logger.monitor_error_rate(error)

        # Should trigger threshold
        assert security_logger.is_threshold_exceeded()

        # Should log alert event
        alert_events = [
            e
            for e in security_logger._events
            if e.get("event_type") == "error_rate_threshold_exceeded"
        ]
        assert len(alert_events) > 0
        assert alert_events[0]["error_count"] >= 5

    def test_security_event_detection_accuracy(self):
        """Test security event detection with various error types.

        RED: This will fail until security event classification is implemented.
        """
        from mcp_server_cheap_llm.utils.logging import SecurityLogger

        security_logger = SecurityLogger("test_security")

        # Test different error types for security classification
        test_cases = [
            (SecurityError("Unauthorized access"), "high"),
            (ValidationError("Invalid input format"), "medium"),
            (ConfigurationError("Missing API key"), "low"),
            (ProviderError("Rate limit exceeded", provider="test_provider"), "medium"),
            (CheapLLMError("General error"), "low"),
        ]

        for error, expected_severity in test_cases:
            with LogContext() as context:
                detected_severity = security_logger.detect_security_severity(error)
                assert detected_severity == expected_severity

    def test_alert_threshold_triggering_mechanism(self):
        """Test alert threshold triggering with different configurations.

        RED: This will fail until alert threshold mechanism is implemented.
        """
        from mcp_server_cheap_llm.utils.logging import SecurityLogger

        # Test different threshold configurations
        configs = [
            {"error_rate_threshold": 3, "threshold_window_minutes": 1},
            {"error_rate_threshold": 10, "threshold_window_minutes": 5},
        ]

        for config in configs:
            security_logger = SecurityLogger("test_security", **config)

            # Generate errors up to threshold
            for i in range(config["error_rate_threshold"]):
                error = ProviderError(f"Error {i}", provider="test_provider")
                security_logger.monitor_error_rate(error)

            # Threshold should be reached
            assert security_logger.is_threshold_exceeded()

            # Alert should be triggered
            alerts = [
                e
                for e in security_logger._events
                if e.get("event_type") == "error_rate_threshold_exceeded"
            ]
            assert len(alerts) > 0

    def test_comprehensive_security_and_error_integration(self):
        """Test comprehensive integration of security logging with error handling.

        RED: This will fail until full integration is implemented.
        """
        from mcp_server_cheap_llm.utils.logging import SecurityLogger

        security_logger = SecurityLogger(
            "comprehensive_test",
            error_rate_threshold=3,
            threshold_window_minutes=1,
            enable_sensitive_data_filtering=True,
        )

        # Simulate a complex scenario with multiple types of events
        with LogContext() as context:
            # 1. Security event
            security_error = SecurityError(
                "Suspicious activity detected",
                error_code="SEC002",
                context={"source_ip": "192.168.1.100", "api_key": "sk-secret"},
            )
            security_logger.log_security_event(
                event_type="suspicious_activity",
                description="Multiple failed auth attempts",
                error=security_error,
                source="authentication_service",
            )

            # 2. Multiple provider errors to trigger threshold
            for i in range(4):
                provider_error = ProviderError(
                    f"API failure {i}",
                    provider="test_provider",
                    error_code=f"PROV{i:03d}",
                    context={"password": "secret", "user_id": f"user_{i}"},
                )
                security_logger.monitor_error_rate(provider_error)

            # 3. Validation error with sensitive data
            validation_error = ValidationError(
                "Invalid credentials",
                context={"token": "bearer_xyz", "request_id": "req_789"},
            )
            security_logger.log_error_with_filtering(validation_error)

        # Verify comprehensive logging occurred
        events = security_logger._events
        assert (
            len(events) >= 3
        )  # At least 3 events logged (security event + threshold alert + validation error)

        # Verify threshold was exceeded
        assert security_logger.is_threshold_exceeded()

        # Verify sensitive data filtering by checking filtered values
        for event in events:
            if "error_context" in event:
                context = event["error_context"]
                if "api_key" in context:
                    assert context["api_key"] == "[FILTERED]"
                if "password" in context:
                    assert context["password"] == "[FILTERED]"
                if "token" in context:
                    assert context["token"] == "[FILTERED]"
                # Preserved fields should have original values
                if "user_id" in context:
                    assert context["user_id"] != "[FILTERED]"
                if "request_id" in context:
                    assert context["request_id"] != "[FILTERED]"

        # Verify alert was generated
        alerts = [
            e for e in events if e.get("event_type") == "error_rate_threshold_exceeded"
        ]
        assert len(alerts) > 0

    def test_performance_with_high_volume_security_events(self):
        """Test SecurityLogger performance with high volume of security events.

        RED: This will fail until performance optimization is implemented.
        """
        from mcp_server_cheap_llm.utils.logging import SecurityLogger

        security_logger = SecurityLogger("performance_test")

        start_time = time.time()

        # Generate high volume of events
        for i in range(1000):
            error = SecurityError(f"Event {i}", error_code=f"SEC{i:04d}")
            security_logger.log_security_event(
                event_type="bulk_test",
                description=f"Test event {i}",
                error=error,
                source="performance_test",
            )

        end_time = time.time()
        duration = end_time - start_time

        # Should process 1000 events in reasonable time (< 5 seconds)
        assert duration < 5.0
        assert len(security_logger._events) == 1000

    def test_thread_safety_with_concurrent_security_logging(self):
        """Test thread safety with concurrent security logging operations.

        RED: This will fail until thread safety is implemented.
        """
        from mcp_server_cheap_llm.utils.logging import SecurityLogger

        security_logger = SecurityLogger("thread_safety_test")
        results = []

        def log_security_events(thread_id):
            for i in range(50):
                error = SecurityError(f"Thread {thread_id} Event {i}")
                security_logger.log_security_event(
                    event_type="thread_test",
                    description=f"Thread {thread_id} event {i}",
                    error=error,
                    source=f"thread_{thread_id}",
                )
            results.append(f"thread_{thread_id}_complete")

        # Create multiple threads
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=log_security_events, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all threads completed
        assert len(results) == 5

        # Verify all events were logged (5 threads * 50 events each)
        assert len(security_logger._events) == 250

        # Verify no data corruption
        thread_counts = {}
        for event in security_logger._events:
            source = event.get("source", "")
            if source.startswith("thread_"):
                thread_counts[source] = thread_counts.get(source, 0) + 1

        # Each thread should have logged exactly 50 events
        for thread_id in range(5):
            assert thread_counts.get(f"thread_{thread_id}", 0) == 50
