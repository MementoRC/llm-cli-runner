"""Unit tests for logging infrastructure - TDD approach."""

import json
import logging
import threading
import time
from unittest.mock import Mock, patch
from uuid import UUID

import pytest

from mcp_server_cheap_llm.utils.logging import LogContext, StructuredLogger


class TestLogContext:
    """Test suite for LogContext class."""

    def test_log_context_import(self):
        """Test that LogContext can be imported."""
        assert LogContext is not None

    def test_log_context_generates_unique_correlation_ids(self):
        """Test that LogContext generates unique correlation IDs."""
        context1 = LogContext()
        context2 = LogContext()

        assert context1.correlation_id != context2.correlation_id
        assert isinstance(context1.correlation_id, str)
        assert len(context1.correlation_id) > 0

        # Should be valid UUID format
        UUID(context1.correlation_id)  # Will raise ValueError if invalid
        UUID(context2.correlation_id)

    def test_log_context_thread_safety(self):
        """Test that LogContext is thread-safe."""
        contexts = []

        def create_context():
            context = LogContext()
            contexts.append(context.correlation_id)
            time.sleep(0.01)  # Small delay to test concurrency

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_context)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All correlation IDs should be unique
        assert len(contexts) == 10
        assert len(set(contexts)) == 10

    def test_log_context_with_statement(self):
        """Test that LogContext works as context manager."""
        with LogContext() as context:
            assert context.correlation_id is not None
            assert isinstance(context.correlation_id, str)

    def test_log_context_preserves_correlation_id(self):
        """Test that correlation ID is preserved within context."""
        context = LogContext()
        correlation_id = context.correlation_id

        # Should maintain same ID
        assert context.correlation_id == correlation_id
        assert context.correlation_id == correlation_id  # Multiple calls


class TestStructuredLogger:
    """Test suite for StructuredLogger class."""

    def test_structured_logger_import(self):
        """Test that StructuredLogger can be imported."""
        assert StructuredLogger is not None

    def test_structured_logger_instantiation(self):
        """Test StructuredLogger can be instantiated."""
        logger = StructuredLogger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"

    def test_structured_logger_json_format(self):
        """Test that StructuredLogger outputs JSON format."""
        logger = StructuredLogger("test_logger")

        with patch.object(logger._logger, "info") as mock_info:
            logger.info("Test message", extra_field="test_value")

            # Should call the underlying logger
            mock_info.assert_called_once()

            # The message should be JSON formatted
            call_args = mock_info.call_args
            assert call_args is not None

            # The logged message should be valid JSON
            logged_message = call_args[0][0]
            parsed = json.loads(logged_message)
            assert parsed["message"] == "Test message"
            assert parsed["extra_field"] == "test_value"
            assert parsed["level"] == "INFO"

    def test_structured_logger_correlation_id_injection(self):
        """Test that StructuredLogger injects correlation IDs."""
        logger = StructuredLogger("test_logger")

        with patch.object(logger._logger, "info") as mock_info:
            with LogContext() as context:
                logger.info("Test message")

                # Should inject correlation ID
                mock_info.assert_called_once()
                call_args = mock_info.call_args

                # Should contain correlation ID in the logged data
                logged_message = call_args[0][0]
                parsed = json.loads(logged_message)
                assert parsed["correlation_id"] == context.correlation_id

    def test_structured_logger_log_levels(self):
        """Test that StructuredLogger supports all log levels."""
        with patch.dict("os.environ", {"MCP_LOG_LEVEL": "DEBUG"}):
            logger = StructuredLogger("test_logger")

            with (
                patch.object(logger._logger, "debug") as mock_debug,
                patch.object(logger._logger, "info") as mock_info,
                patch.object(logger._logger, "warning") as mock_warning,
                patch.object(logger._logger, "error") as mock_error,
                patch.object(logger._logger, "critical") as mock_critical,
            ):
                logger.debug("Debug message")
                logger.info("Info message")
                logger.warning("Warning message")
                logger.error("Error message")
                logger.critical("Critical message")

                # Should call appropriate methods
                mock_debug.assert_called_once()
                mock_info.assert_called_once()
                mock_warning.assert_called_once()
                mock_error.assert_called_once()
                mock_critical.assert_called_once()

    def test_structured_logger_custom_fields(self):
        """Test that StructuredLogger supports custom fields."""
        logger = StructuredLogger("test_logger")

        with patch.object(logger._logger, "info") as mock_info:
            logger.info(
                "Test message",
                user_id="user123",
                request_id="req456",
                extra_data={"key": "value"},
            )

            # Should include custom fields
            mock_info.assert_called_once()
            call_args = mock_info.call_args

            # Custom fields should be in the JSON
            logged_message = call_args[0][0]
            parsed = json.loads(logged_message)
            assert parsed["user_id"] == "user123"
            assert parsed["request_id"] == "req456"
            assert parsed["extra_data"] == {"key": "value"}

    def test_structured_logger_thread_safety(self):
        """Test that StructuredLogger is thread-safe."""
        logger = StructuredLogger("test_logger")
        results = []

        def log_with_context():
            with LogContext() as context:
                logger.info("Thread message")
                results.append(context.correlation_id)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=log_with_context)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have unique correlation IDs from different threads
        assert len(results) == 5
        assert len(set(results)) == 5

    def test_structured_logger_configuration_loading(self):
        """Test that StructuredLogger can load configuration."""
        with patch.dict("os.environ", {"MCP_LOG_LEVEL": "DEBUG"}):
            logger = StructuredLogger("test_logger")

            # Should respect environment configuration
            assert logger.level == logging.DEBUG

    def test_structured_logger_performance_optimization(self):
        """Test that StructuredLogger is performant."""
        logger = StructuredLogger("test_logger")

        # Should handle rapid logging without significant delay
        start_time = time.time()

        for i in range(100):
            logger.info(f"Performance test {i}")

        elapsed_time = time.time() - start_time

        # Should complete 100 logs in reasonable time (< 1 second)
        assert elapsed_time < 1.0

    def test_structured_logger_exception_handling(self):
        """Test that StructuredLogger handles exceptions gracefully."""
        logger = StructuredLogger("test_logger")

        # Should not raise exceptions on invalid input
        try:
            logger.info("Test", invalid_field=object())  # Non-serializable object
            logger.info(None)  # None message
            logger.info("")  # Empty message
        except Exception as e:
            pytest.fail(f"StructuredLogger raised exception: {e}")


class TestLoggingIntegration:
    """Integration tests for logging components."""

    def test_log_context_with_structured_logger(self):
        """Test LogContext integration with StructuredLogger."""
        logger = StructuredLogger("integration_test")

        with patch.object(logger._logger, "info") as mock_info:
            with LogContext() as context:
                logger.info("Integration test message")

                # Should work together seamlessly
                mock_info.assert_called_once()
                call_args = mock_info.call_args

                # Context should be preserved
                logged_message = call_args[0][0]
                parsed = json.loads(logged_message)
                assert parsed["correlation_id"] == context.correlation_id

    def test_nested_log_contexts(self):
        """Test nested LogContext behavior."""
        logger = StructuredLogger("nested_test")

        with patch("logging.getLogger") as mock_get_logger:
            mock_log_instance = Mock()
            mock_get_logger.return_value = mock_log_instance

            with LogContext() as outer_context:
                logger.info("Outer message")

                with LogContext() as inner_context:
                    logger.info("Inner message")

                    # Inner context should have different correlation ID
                    assert outer_context.correlation_id != inner_context.correlation_id

    def test_log_context_propagation_across_threads(self):
        """Test that LogContext doesn't leak between threads."""
        logger = StructuredLogger("thread_test")
        thread_contexts = {}

        def log_in_thread(thread_id):
            with LogContext() as context:
                logger.info(f"Thread {thread_id} message")
                thread_contexts[thread_id] = context.correlation_id

        threads = []
        for i in range(3):
            thread = threading.Thread(target=log_in_thread, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Each thread should have unique correlation ID
        correlation_ids = list(thread_contexts.values())
        assert len(correlation_ids) == 3
        assert len(set(correlation_ids)) == 3
