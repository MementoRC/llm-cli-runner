"""Unit tests for custom exception hierarchy - TDD approach."""

import pytest

from mcp_server_cheap_llm.utils.errors import (
    CheapLLMError,
    ConfigurationError,
    ProviderError,
    RateLimitError,
    SecurityError,
    ValidationError,
)


class TestCheapLLMError:
    """Test suite for base CheapLLMError class."""

    def test_cheap_llm_error_import(self):
        """Test that CheapLLMError can be imported."""
        assert CheapLLMError is not None

    def test_cheap_llm_error_basic_instantiation(self):
        """Test basic CheapLLMError instantiation."""
        error = CheapLLMError("Test error message")
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.context == {}
        assert str(error) == "Test error message"

    def test_cheap_llm_error_with_error_code(self):
        """Test CheapLLMError with error code."""
        error = CheapLLMError("Test error", error_code="E001")
        assert error.message == "Test error"
        assert error.error_code == "E001"
        assert error.context == {}

    def test_cheap_llm_error_with_context(self):
        """Test CheapLLMError with context information."""
        context = {"user_id": "123", "request_id": "req456"}
        error = CheapLLMError("Test error", context=context)
        assert error.message == "Test error"
        assert error.error_code is None
        assert error.context == context

    def test_cheap_llm_error_full_instantiation(self):
        """Test CheapLLMError with all parameters."""
        context = {"detail": "extra info"}
        error = CheapLLMError("Full error", error_code="E999", context=context)
        assert error.message == "Full error"
        assert error.error_code == "E999"
        assert error.context == context

    def test_cheap_llm_error_to_dict(self):
        """Test CheapLLMError to_dict method."""
        context = {"key": "value"}
        error = CheapLLMError("Test error", error_code="E001", context=context)
        error_dict = error.to_dict()

        expected_dict = {
            "error_type": "CheapLLMError",
            "message": "Test error",
            "error_code": "E001",
            "context": {"key": "value"},
        }
        assert error_dict == expected_dict

    def test_cheap_llm_error_inheritance(self):
        """Test that CheapLLMError inherits from Exception."""
        error = CheapLLMError("Test error")
        assert isinstance(error, Exception)
        assert isinstance(error, CheapLLMError)

    def test_cheap_llm_error_can_be_raised(self):
        """Test that CheapLLMError can be raised and caught."""
        with pytest.raises(CheapLLMError) as exc_info:
            raise CheapLLMError("Test error", error_code="E001")

        assert exc_info.value.message == "Test error"
        assert exc_info.value.error_code == "E001"


class TestConfigurationError:
    """Test suite for ConfigurationError class."""

    def test_configuration_error_import(self):
        """Test that ConfigurationError can be imported."""
        assert ConfigurationError is not None

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from CheapLLMError."""
        error = ConfigurationError("Config error")
        assert isinstance(error, CheapLLMError)
        assert isinstance(error, ConfigurationError)

    def test_configuration_error_basic_instantiation(self):
        """Test basic ConfigurationError instantiation."""
        error = ConfigurationError("Invalid configuration")
        assert error.message == "Invalid configuration"
        assert error.error_code is None
        assert error.context == {}

    def test_configuration_error_with_error_code(self):
        """Test ConfigurationError with error code."""
        error = ConfigurationError("Missing field", error_code="CFG001")
        assert error.message == "Missing field"
        assert error.error_code == "CFG001"

    def test_configuration_error_with_context(self):
        """Test ConfigurationError with context."""
        context = {"field": "api_key", "provider": "gemini"}
        error = ConfigurationError("Missing API key", context=context)
        assert error.message == "Missing API key"
        assert error.context == context

    def test_configuration_error_to_dict(self):
        """Test ConfigurationError to_dict method."""
        context = {"field": "model_name"}
        error = ConfigurationError(
            "Invalid model", error_code="CFG002", context=context
        )
        error_dict = error.to_dict()

        expected_dict = {
            "error_type": "ConfigurationError",
            "message": "Invalid model",
            "error_code": "CFG002",
            "context": {"field": "model_name"},
        }
        assert error_dict == expected_dict

    def test_configuration_error_can_be_raised(self):
        """Test that ConfigurationError can be raised and caught."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Config error", error_code="CFG001")

        assert exc_info.value.message == "Config error"
        assert exc_info.value.error_code == "CFG001"


class TestProviderError:
    """Test suite for ProviderError class."""

    def test_provider_error_import(self):
        """Test that ProviderError can be imported."""
        assert ProviderError is not None

    def test_provider_error_inheritance(self):
        """Test ProviderError inherits from CheapLLMError."""
        error = ProviderError("Provider error", provider="gemini")
        assert isinstance(error, CheapLLMError)
        assert isinstance(error, ProviderError)

    def test_provider_error_basic_instantiation(self):
        """Test basic ProviderError instantiation."""
        error = ProviderError("API error", provider="gemini")
        assert error.message == "API error"
        assert error.provider == "gemini"
        assert error.error_code is None
        assert error.context["provider"] == "gemini"

    def test_provider_error_with_error_code(self):
        """Test ProviderError with error code."""
        error = ProviderError("Rate limit", provider="openai", error_code="PRV001")
        assert error.message == "Rate limit"
        assert error.provider == "openai"
        assert error.error_code == "PRV001"

    def test_provider_error_with_context(self):
        """Test ProviderError with additional context."""
        context = {"status_code": 429, "retry_after": 60}
        error = ProviderError("Rate limited", provider="gemini", context=context)
        assert error.message == "Rate limited"
        assert error.provider == "gemini"
        assert error.context["provider"] == "gemini"
        assert error.context["status_code"] == 429
        assert error.context["retry_after"] == 60

    def test_provider_error_provider_in_context(self):
        """Test that provider is automatically added to context."""
        error = ProviderError("Error", provider="codex")
        assert error.context["provider"] == "codex"

    def test_provider_error_to_dict(self):
        """Test ProviderError to_dict method."""
        context = {"status_code": 500}
        error = ProviderError(
            "Server error", provider="gemini", error_code="PRV002", context=context
        )
        error_dict = error.to_dict()

        expected_dict = {
            "error_type": "ProviderError",
            "message": "Server error",
            "error_code": "PRV002",
            "context": {"provider": "gemini", "status_code": 500},
        }
        assert error_dict == expected_dict

    def test_provider_error_can_be_raised(self):
        """Test that ProviderError can be raised and caught."""
        with pytest.raises(ProviderError) as exc_info:
            raise ProviderError(
                "Provider error", provider="gemini", error_code="PRV001"
            )

        assert exc_info.value.message == "Provider error"
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.error_code == "PRV001"


class TestRateLimitError:
    """Test suite for RateLimitError class."""

    def test_rate_limit_error_import(self):
        """Test that RateLimitError can be imported."""
        assert RateLimitError is not None

    def test_rate_limit_error_inheritance(self):
        """Test RateLimitError inherits from ProviderError."""
        error = RateLimitError("Rate limit", provider="gemini", retry_after=60)
        assert isinstance(error, ProviderError)
        assert isinstance(error, RateLimitError)

    def test_rate_limit_error_basic_instantiation(self):
        """Test basic RateLimitError instantiation."""
        error = RateLimitError("Rate limit exceeded", provider="gemini", retry_after=60)
        assert error.message == "Rate limit exceeded"
        assert error.provider == "gemini"
        assert error.retry_after == 60
        assert error.context["provider"] == "gemini"
        assert error.context["retry_after"] == 60

    def test_rate_limit_error_with_error_code(self):
        """Test RateLimitError with error code."""
        error = RateLimitError(
            "Rate limit", provider="openai", retry_after=120, error_code="RTE001"
        )
        assert error.message == "Rate limit"
        assert error.provider == "openai"
        assert error.retry_after == 120
        assert error.error_code == "RTE001"

    def test_rate_limit_error_with_context(self):
        """Test RateLimitError with additional context."""
        context = {"quota_remaining": 0, "reset_time": "2024-01-01T12:00:00Z"}
        error = RateLimitError(
            "Quota exceeded", provider="gemini", retry_after=300, context=context
        )
        assert error.message == "Quota exceeded"
        assert error.provider == "gemini"
        assert error.retry_after == 300
        assert error.context["provider"] == "gemini"
        assert error.context["retry_after"] == 300
        assert error.context["quota_remaining"] == 0
        assert error.context["reset_time"] == "2024-01-01T12:00:00Z"

    def test_rate_limit_error_to_dict(self):
        """Test RateLimitError to_dict method."""
        error = RateLimitError(
            "Rate limit", provider="gemini", retry_after=60, error_code="RTE001"
        )
        error_dict = error.to_dict()

        expected_dict = {
            "error_type": "RateLimitError",
            "message": "Rate limit",
            "error_code": "RTE001",
            "context": {"provider": "gemini", "retry_after": 60},
        }
        assert error_dict == expected_dict

    def test_rate_limit_error_can_be_raised(self):
        """Test that RateLimitError can be raised and caught."""
        with pytest.raises(RateLimitError) as exc_info:
            raise RateLimitError(
                "Rate limit", provider="gemini", retry_after=60, error_code="RTE001"
            )

        assert exc_info.value.message == "Rate limit"
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.retry_after == 60
        assert exc_info.value.error_code == "RTE001"


class TestValidationError:
    """Test suite for ValidationError class."""

    def test_validation_error_import(self):
        """Test that ValidationError can be imported."""
        assert ValidationError is not None

    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from CheapLLMError."""
        error = ValidationError("Validation error")
        assert isinstance(error, CheapLLMError)
        assert isinstance(error, ValidationError)

    def test_validation_error_basic_instantiation(self):
        """Test basic ValidationError instantiation."""
        error = ValidationError("Invalid input")
        assert error.message == "Invalid input"
        assert error.error_code is None
        assert error.context == {}

    def test_validation_error_with_error_code(self):
        """Test ValidationError with error code."""
        error = ValidationError("Prompt too long", error_code="VAL001")
        assert error.message == "Prompt too long"
        assert error.error_code == "VAL001"

    def test_validation_error_with_context(self):
        """Test ValidationError with context."""
        context = {"max_length": 10000, "actual_length": 15000}
        error = ValidationError("Prompt too long", context=context)
        assert error.message == "Prompt too long"
        assert error.context == context

    def test_validation_error_to_dict(self):
        """Test ValidationError to_dict method."""
        context = {"field": "temperature", "value": 2.0, "max": 1.0}
        error = ValidationError(
            "Invalid temperature", error_code="VAL002", context=context
        )
        error_dict = error.to_dict()

        expected_dict = {
            "error_type": "ValidationError",
            "message": "Invalid temperature",
            "error_code": "VAL002",
            "context": {"field": "temperature", "value": 2.0, "max": 1.0},
        }
        assert error_dict == expected_dict

    def test_validation_error_can_be_raised(self):
        """Test that ValidationError can be raised and caught."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Validation error", error_code="VAL001")

        assert exc_info.value.message == "Validation error"
        assert exc_info.value.error_code == "VAL001"


class TestSecurityError:
    """Test suite for SecurityError class."""

    def test_security_error_import(self):
        """Test that SecurityError can be imported."""
        assert SecurityError is not None

    def test_security_error_inheritance(self):
        """Test SecurityError inherits from CheapLLMError."""
        error = SecurityError("Security violation")
        assert isinstance(error, CheapLLMError)
        assert isinstance(error, SecurityError)

    def test_security_error_basic_instantiation(self):
        """Test basic SecurityError instantiation."""
        error = SecurityError("Unsafe command detected")
        assert error.message == "Unsafe command detected"
        assert error.error_code is None
        assert error.context == {}

    def test_security_error_with_error_code(self):
        """Test SecurityError with error code."""
        error = SecurityError("Command injection", error_code="SEC001")
        assert error.message == "Command injection"
        assert error.error_code == "SEC001"

    def test_security_error_with_context(self):
        """Test SecurityError with context."""
        context = {"command": "rm -rf /", "source": "user_input"}
        error = SecurityError("Unsafe command", context=context)
        assert error.message == "Unsafe command"
        assert error.context == context

    def test_security_error_to_dict(self):
        """Test SecurityError to_dict method."""
        context = {"threat_level": "high", "blocked": True}
        error = SecurityError(
            "Security violation", error_code="SEC002", context=context
        )
        error_dict = error.to_dict()

        expected_dict = {
            "error_type": "SecurityError",
            "message": "Security violation",
            "error_code": "SEC002",
            "context": {"threat_level": "high", "blocked": True},
        }
        assert error_dict == expected_dict

    def test_security_error_can_be_raised(self):
        """Test that SecurityError can be raised and caught."""
        with pytest.raises(SecurityError) as exc_info:
            raise SecurityError("Security error", error_code="SEC001")

        assert exc_info.value.message == "Security error"
        assert exc_info.value.error_code == "SEC001"


class TestExceptionHierarchy:
    """Test suite for exception hierarchy and relationships."""

    def test_exception_hierarchy_correctness(self):
        """Test that exception hierarchy is correct."""
        # All exceptions should inherit from CheapLLMError
        assert issubclass(ConfigurationError, CheapLLMError)
        assert issubclass(ProviderError, CheapLLMError)
        assert issubclass(ValidationError, CheapLLMError)
        assert issubclass(SecurityError, CheapLLMError)

        # RateLimitError should inherit from ProviderError
        assert issubclass(RateLimitError, ProviderError)
        assert issubclass(RateLimitError, CheapLLMError)

        # All should inherit from Exception
        assert issubclass(CheapLLMError, Exception)
        assert issubclass(ConfigurationError, Exception)
        assert issubclass(ProviderError, Exception)
        assert issubclass(RateLimitError, Exception)
        assert issubclass(ValidationError, Exception)
        assert issubclass(SecurityError, Exception)

    def test_exception_hierarchy_catching(self):
        """Test that exceptions can be caught by their parent classes."""
        # ConfigurationError can be caught as CheapLLMError
        with pytest.raises(CheapLLMError):
            raise ConfigurationError("Config error")

        # ProviderError can be caught as CheapLLMError
        with pytest.raises(CheapLLMError):
            raise ProviderError("Provider error", provider="gemini")

        # RateLimitError can be caught as ProviderError or CheapLLMError
        with pytest.raises(ProviderError):
            raise RateLimitError("Rate limit", provider="gemini", retry_after=60)

        with pytest.raises(CheapLLMError):
            raise RateLimitError("Rate limit", provider="gemini", retry_after=60)

    def test_error_code_uniqueness_validation(self):
        """Test error code uniqueness validation."""
        # This test validates that error codes are unique within the system
        # We'll test some common error codes that should be unique

        config_error = ConfigurationError("Config error", error_code="CFG001")
        provider_error = ProviderError(
            "Provider error", provider="gemini", error_code="PRV001"
        )
        validation_error = ValidationError("Validation error", error_code="VAL001")
        security_error = SecurityError("Security error", error_code="SEC001")
        rate_limit_error = RateLimitError(
            "Rate limit", provider="gemini", retry_after=60, error_code="RTE001"
        )

        # Error codes should be different
        error_codes = [
            config_error.error_code,
            provider_error.error_code,
            validation_error.error_code,
            security_error.error_code,
            rate_limit_error.error_code,
        ]

        # All error codes should be unique
        assert len(error_codes) == len(set(error_codes))

        # Error codes should follow expected patterns
        assert config_error.error_code.startswith("CFG")
        assert provider_error.error_code.startswith("PRV")
        assert validation_error.error_code.startswith("VAL")
        assert security_error.error_code.startswith("SEC")
        assert rate_limit_error.error_code.startswith("RTE")

    def test_structured_error_data_validation(self):
        """Test structured error data validation."""
        # Test that error data is properly structured
        context = {"field": "api_key", "provider": "gemini", "status": "missing"}
        error = ConfigurationError(
            "Missing API key", error_code="CFG001", context=context
        )

        error_dict = error.to_dict()

        # Should have all required fields
        assert "error_type" in error_dict
        assert "message" in error_dict
        assert "error_code" in error_dict
        assert "context" in error_dict

        # Should have correct types
        assert isinstance(error_dict["error_type"], str)
        assert isinstance(error_dict["message"], str)
        assert isinstance(error_dict["error_code"], str)
        assert isinstance(error_dict["context"], dict)

        # Should preserve all context data
        assert error_dict["context"]["field"] == "api_key"
        assert error_dict["context"]["provider"] == "gemini"
        assert error_dict["context"]["status"] == "missing"

    def test_provider_error_mapping_accuracy(self):
        """Test provider error mapping accuracy."""
        # Test mapping of different provider errors
        providers = ["gemini", "openai", "anthropic", "codex", "llama"]

        for provider in providers:
            error = ProviderError(
                f"{provider} error", provider=provider, error_code="PRV001"
            )
            assert error.provider == provider
            assert error.context["provider"] == provider

            # Test RateLimitError for each provider
            rate_error = RateLimitError(
                f"{provider} rate limit", provider=provider, retry_after=60
            )
            assert rate_error.provider == provider
            assert rate_error.context["provider"] == provider
            assert rate_error.retry_after == 60
            assert rate_error.context["retry_after"] == 60

    def test_error_data_serialization(self):
        """Test error data serialization capabilities."""
        # Test that error data can be serialized to JSON
        import json

        context = {
            "provider": "gemini",
            "status_code": 429,
            "retry_after": 60,
            "request_id": "req_123",
            "nested": {"key": "value", "number": 42},
        }

        error = RateLimitError(
            "Rate limit",
            provider="gemini",
            retry_after=60,
            error_code="RTE001",
            context=context,
        )
        error_dict = error.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(error_dict)
        assert isinstance(json_str, str)

        # Should be deserializable
        deserialized = json.loads(json_str)
        assert deserialized == error_dict

        # Should preserve all data
        assert deserialized["error_type"] == "RateLimitError"
        assert deserialized["message"] == "Rate limit"
        assert deserialized["error_code"] == "RTE001"
        assert deserialized["context"]["provider"] == "gemini"
        assert deserialized["context"]["retry_after"] == 60
        assert deserialized["context"]["nested"]["key"] == "value"
