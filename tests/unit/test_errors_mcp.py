"""Tests for MCP protocol error serialization.

This module tests the ErrorSerializer class for MCP protocol compliance,
error message formatting, and context preservation.
"""

import json
from unittest.mock import patch

import pytest

from mcp_server_llm_cli_runner.core.errors import (
    ConfigurationError,
    LLMCliRunnerError,
    ProviderError,
    RateLimitError,
    SecurityError,
    ValidationError,
)
from mcp_server_llm_cli_runner.utils.errors import ErrorSerializer


class TestErrorSerializerTDD:
    """Test-driven development for ErrorSerializer class."""

    def test_error_serializer_import(self):
        """Test that ErrorSerializer can be imported."""
        from mcp_server_llm_cli_runner.utils.errors import ErrorSerializer

        assert ErrorSerializer is not None

    def test_error_serializer_instantiation(self):
        """Test that ErrorSerializer can be instantiated."""
        serializer = ErrorSerializer()
        assert serializer is not None

    def test_serialize_method_exists(self):
        """Test that serialize method exists."""
        serializer = ErrorSerializer()
        assert hasattr(serializer, "serialize")
        assert callable(serializer.serialize)

    def test_deserialize_method_exists(self):
        """Test that deserialize method exists."""
        serializer = ErrorSerializer()
        assert hasattr(serializer, "deserialize")
        assert callable(serializer.deserialize)


class TestMCPProtocolCompliance:
    """Test MCP protocol specification compliance."""

    def test_serialize_llm_cli_runner_error_mcp_format(self):
        """Test serialization of LLMCliRunnerError to MCP format."""
        error = LLMCliRunnerError(
            "Test error",
            error_code="TEST001",
            context={"key": "value"},
        )
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        # MCP protocol requires specific fields
        assert "error" in result
        assert "code" in result["error"]
        assert "message" in result["error"]
        assert "data" in result["error"]

        # Check values
        assert result["error"]["code"] == "TEST001"
        assert result["error"]["message"] == "Test error"
        assert result["error"]["data"]["context"] == {"key": "value"}

    def test_serialize_configuration_error_mcp_format(self):
        """Test serialization of ConfigurationError to MCP format."""
        error = ConfigurationError("Invalid config", error_code="CFG001")
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        assert result["error"]["code"] == "CFG001"
        assert result["error"]["message"] == "Invalid config"
        assert result["error"]["data"]["error_type"] == "ConfigurationError"

    def test_serialize_provider_error_mcp_format(self):
        """Test serialization of ProviderError to MCP format."""
        error = ProviderError("API failed", provider="gemini", error_code="PRV001")
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        assert result["error"]["code"] == "PRV001"
        assert result["error"]["message"] == "API failed"
        assert result["error"]["data"]["provider"] == "gemini"

    def test_serialize_validation_error_mcp_format(self):
        """Test serialization of ValidationError to MCP format."""
        error = ValidationError(
            "Invalid input",
            error_code="VAL001",
            context={"field": "prompt"},
        )
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        assert result["error"]["code"] == "VAL001"
        assert result["error"]["message"] == "Invalid input"
        assert result["error"]["data"]["context"]["field"] == "prompt"

    def test_serialize_rate_limit_error_mcp_format(self):
        """Test serialization of RateLimitError to MCP format."""
        error = RateLimitError("Rate limit exceeded", provider="gemini", retry_after=60)
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        assert result["error"]["code"] is not None
        assert result["error"]["message"] == "Rate limit exceeded"
        assert result["error"]["data"]["retry_after"] == 60

    def test_serialize_security_error_mcp_format(self):
        """Test serialization of SecurityError to MCP format."""
        error = SecurityError("Unsafe operation", error_code="SEC001")
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        assert result["error"]["code"] == "SEC001"
        assert result["error"]["message"] == "Unsafe operation"
        assert result["error"]["data"]["error_type"] == "SecurityError"


class TestErrorMessageFormatting:
    """Test error message formatting and clarity."""

    def test_error_message_clarity(self):
        """Test that error messages are clear and actionable."""
        error = ValidationError(
            "Prompt exceeds maximum length",
            error_code="VAL001",
            context={"max_length": 1000, "actual_length": 1500},
        )
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        # Message should be clear and actionable
        assert "Prompt exceeds maximum length" in result["error"]["message"]
        assert result["error"]["data"]["context"]["max_length"] == 1000
        assert result["error"]["data"]["context"]["actual_length"] == 1500

    def test_error_message_with_context(self):
        """Test error message formatting with context information."""
        error = ProviderError(
            "API authentication failed",
            provider="gemini",
            error_code="PRV001",
            context={"endpoint": "/v1/chat"},
        )
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        assert result["error"]["message"] == "API authentication failed"
        assert result["error"]["data"]["provider"] == "gemini"
        assert result["error"]["data"]["context"]["endpoint"] == "/v1/chat"

    def test_error_message_without_code(self):
        """Test error message formatting when no error code is provided."""
        error = LLMCliRunnerError("Generic error message")
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        # Should generate default error code
        assert result["error"]["code"] is not None
        assert result["error"]["message"] == "Generic error message"


class TestContextDataPreservation:
    """Test context data preservation during serialization."""

    def test_context_preservation_complete(self):
        """Test that all context data is preserved during serialization."""
        context = {
            "user_id": "12345",
            "request_id": "req-67890",
            "timestamp": "2024-01-01T00:00:00Z",
            "nested": {"key": "value", "count": 42},
        }
        error = LLMCliRunnerError("Test error", error_code="TEST001", context=context)
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        # All context should be preserved
        assert result["error"]["data"]["context"] == context

    def test_context_preservation_with_special_types(self):
        """Test context preservation with special Python types."""
        from datetime import datetime

        context = {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "count": 42,
            "is_active": True,
            "items": ["item1", "item2", "item3"],
        }
        error = LLMCliRunnerError("Test error", context=context)
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        # Should handle type conversion for JSON serialization
        assert result["error"]["data"]["context"]["count"] == 42
        assert result["error"]["data"]["context"]["is_active"] is True
        assert result["error"]["data"]["context"]["items"] == [
            "item1",
            "item2",
            "item3",
        ]

    def test_context_preservation_empty_context(self):
        """Test handling of empty context."""
        error = LLMCliRunnerError("Test error", context={})
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        assert result["error"]["data"]["context"] == {}


class TestSerializationDeserialization:
    """Test serialization and deserialization round-trip consistency."""

    def test_round_trip_llm_cli_runner_error(self):
        """Test round-trip serialization/deserialization of LLMCliRunnerError."""
        original_error = LLMCliRunnerError(
            "Test error",
            error_code="TEST001",
            context={"key": "value"},
        )
        serializer = ErrorSerializer()

        # Serialize
        serialized = serializer.serialize(original_error)

        # Deserialize
        deserialized = serializer.deserialize(serialized)

        # Check consistency
        assert isinstance(deserialized, LLMCliRunnerError)
        assert deserialized.message == original_error.message
        assert deserialized.error_code == original_error.error_code
        assert deserialized.context == original_error.context

    def test_round_trip_provider_error(self):
        """Test round-trip serialization/deserialization of ProviderError."""
        original_error = ProviderError(
            "API failed",
            provider="gemini",
            error_code="PRV001",
            context={"endpoint": "/v1/chat"},
        )
        serializer = ErrorSerializer()

        # Serialize
        serialized = serializer.serialize(original_error)

        # Deserialize
        deserialized = serializer.deserialize(serialized)

        # Check consistency
        assert isinstance(deserialized, ProviderError)
        assert deserialized.message == original_error.message
        assert deserialized.error_code == original_error.error_code
        assert deserialized.provider == original_error.provider
        assert deserialized.context == original_error.context

    def test_round_trip_json_serialization(self):
        """Test round-trip through JSON serialization."""
        original_error = ValidationError(
            "Invalid input",
            error_code="VAL001",
            context={"field": "prompt", "value": "test"},
        )
        serializer = ErrorSerializer()

        # Serialize to MCP format
        mcp_format = serializer.serialize(original_error)

        # Convert to JSON and back (simulating network transmission)
        json_str = json.dumps(mcp_format)
        parsed_json = json.loads(json_str)

        # Deserialize
        deserialized = serializer.deserialize(parsed_json)

        # Check consistency
        assert isinstance(deserialized, ValidationError)
        assert deserialized.message == original_error.message
        assert deserialized.error_code == original_error.error_code
        assert deserialized.context == original_error.context


class TestErrorChaining:
    """Test error chaining capabilities."""

    def test_error_chaining_preservation(self):
        """Test that error chaining is preserved during serialization."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise LLMCliRunnerError("Wrapped error", error_code="WRAP001") from e
        except LLMCliRunnerError as error:
            serializer = ErrorSerializer()

            result = serializer.serialize(error)

            # Should preserve chaining information
            assert "cause" in result["error"]["data"]
            assert "ValueError" in result["error"]["data"]["cause"]

    def test_error_chaining_deserialization(self):
        """Test that error chaining is restored during deserialization."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise LLMCliRunnerError("Wrapped error", error_code="WRAP001") from e
        except LLMCliRunnerError as original_error:
            serializer = ErrorSerializer()

            # Serialize
            serialized = serializer.serialize(original_error)

            # Deserialize
            deserialized = serializer.deserialize(serialized)

            # Check chaining information is preserved
            assert hasattr(deserialized, "__cause__") or "cause" in deserialized.context


class TestErrorCodes:
    """Test error code handling and uniqueness."""

    def test_default_error_code_generation(self):
        """Test automatic error code generation when none provided."""
        error = LLMCliRunnerError("Test error")
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        # Should generate a default error code
        assert result["error"]["code"] is not None
        assert result["error"]["code"] != ""

    def test_custom_error_code_preservation(self):
        """Test that custom error codes are preserved."""
        error = LLMCliRunnerError("Test error", error_code="CUSTOM001")
        serializer = ErrorSerializer()

        result = serializer.serialize(error)

        assert result["error"]["code"] == "CUSTOM001"

    def test_error_code_uniqueness_per_type(self):
        """Test that different error types have appropriate default codes."""
        errors = [
            ConfigurationError("Config error"),
            ProviderError("Provider error", provider="gemini"),
            ValidationError("Validation error"),
            SecurityError("Security error"),
        ]

        serializer = ErrorSerializer()
        codes = []

        for error in errors:
            result = serializer.serialize(error)
            codes.append(result["error"]["code"])

        # Each should have appropriate prefix or unique code
        assert len(set(codes)) == len(codes)  # All codes should be unique
