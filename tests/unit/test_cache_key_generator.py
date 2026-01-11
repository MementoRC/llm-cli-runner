"""Unit tests for cache key generation system.

Tests the SHA256-based cache key generation with different request
configurations to ensure deterministic and collision-resistant keys.
"""

import pytest

from mcp_server_llm_cli_runner.cache.key_generator import CacheKeyGenerator
from mcp_server_llm_cli_runner.core.models import LLMRequest, ProviderType


class TestCacheKeyGenerator:
    """Test cache key generation functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.generator = CacheKeyGenerator()

    def test_basic_key_generation(self):
        """Test basic cache key generation."""
        request = LLMRequest(
            prompt="Hello world",
            provider=ProviderType.GEMINI,
            max_tokens=100,
            temperature=0.7,
        )

        key = self.generator.generate_request_key(request)

        # Should have correct format
        assert key.startswith("req:")
        assert ":" in key  # Should contain hash

        # Should be deterministic
        key2 = self.generator.generate_request_key(request)
        assert key == key2

    def test_different_prompts_different_keys(self):
        """Test that different prompts generate different keys."""
        request1 = LLMRequest(prompt="Hello", provider=ProviderType.GEMINI)
        request2 = LLMRequest(prompt="Hi", provider=ProviderType.GEMINI)

        key1 = self.generator.generate_request_key(request1)
        key2 = self.generator.generate_request_key(request2)

        assert key1 != key2

    def test_different_providers_different_keys(self):
        """Test that different providers generate different keys."""
        request1 = LLMRequest(prompt="Hello", provider=ProviderType.GEMINI)
        request2 = LLMRequest(prompt="Hello", provider=ProviderType.OPENAI)

        key1 = self.generator.generate_request_key(request1)
        key2 = self.generator.generate_request_key(request2)

        assert key1 != key2

    def test_different_parameters_different_keys(self):
        """Test that different parameters generate different keys."""
        request1 = LLMRequest(
            prompt="Hello",
            provider=ProviderType.GEMINI,
            max_tokens=100,
            temperature=0.7,
        )
        request2 = LLMRequest(
            prompt="Hello",
            provider=ProviderType.GEMINI,
            max_tokens=200,
            temperature=0.7,
        )

        key1 = self.generator.generate_request_key(request1)
        key2 = self.generator.generate_request_key(request2)

        assert key1 != key2

    def test_metadata_affects_key(self):
        """Test that metadata affects key generation."""
        request1 = LLMRequest(
            prompt="Hello",
            provider=ProviderType.GEMINI,
            metadata={"custom": "value1"},
        )
        request2 = LLMRequest(
            prompt="Hello",
            provider=ProviderType.GEMINI,
            metadata={"custom": "value2"},
        )

        key1 = self.generator.generate_request_key(request1)
        key2 = self.generator.generate_request_key(request2)

        assert key1 != key2

    def test_metadata_order_doesnt_affect_key(self):
        """Test that metadata key order doesn't affect key generation."""
        request1 = LLMRequest(
            prompt="Hello",
            provider=ProviderType.GEMINI,
            metadata={"a": 1, "b": 2},
        )
        request2 = LLMRequest(
            prompt="Hello",
            provider=ProviderType.GEMINI,
            metadata={"b": 2, "a": 1},
        )

        key1 = self.generator.generate_request_key(request1)
        key2 = self.generator.generate_request_key(request2)

        assert key1 == key2

    def test_none_provider_handling(self):
        """Test handling of None provider."""
        request = LLMRequest(prompt="Hello", provider=None)

        key = self.generator.generate_request_key(request)
        assert key.startswith("req:")
        assert ":" in key  # Should contain hash

    def test_system_prompt_affects_key(self):
        """Test that system prompt affects key generation."""
        request1 = LLMRequest(
            prompt="Hello",
            provider=ProviderType.GEMINI,
            system_prompt="You are helpful",
        )
        request2 = LLMRequest(
            prompt="Hello",
            provider=ProviderType.GEMINI,
            system_prompt="You are creative",
        )

        key1 = self.generator.generate_request_key(request1)
        key2 = self.generator.generate_request_key(request2)

        assert key1 != key2

    def test_provider_key_generation(self):
        """Test provider-specific key generation."""
        key1 = self.generator.generate_provider_key("gemini")
        key2 = self.generator.generate_provider_key("openai")

        assert key1 != key2
        assert "provider:gemini" in key1
        assert "provider:openai" in key2

    def test_provider_key_with_model(self):
        """Test provider key generation with model."""
        key1 = self.generator.generate_provider_key("gemini", "gemini-pro")
        key2 = self.generator.generate_provider_key("gemini", "gemini-ultra")

        assert key1 != key2
        assert "provider:gemini:" in key1
        assert "provider:gemini:" in key2

    def test_session_key_generation(self):
        """Test session key generation."""
        session_id = "test-session-123"

        session_key = self.generator.generate_session_key(session_id)

        assert session_key.startswith("req:session:")
        assert session_id in session_key

        # Should be deterministic
        session_key2 = self.generator.generate_session_key(session_id)
        assert session_key == session_key2

    def test_key_validation(self):
        """Test cache key validation."""
        # Valid key
        request = LLMRequest(prompt="Hello", provider=ProviderType.GEMINI)
        valid_key = self.generator.generate_request_key(request)
        assert self.generator.validate_key_format(valid_key)

        # Invalid keys
        assert not self.generator.validate_key_format("")
        assert not self.generator.validate_key_format("invalid")
        assert not self.generator.validate_key_format("no_colon")
        assert self.generator.validate_key_format("valid:format")  # This should pass

    def test_hash_extraction(self):
        """Test hash extraction from cache key."""
        request = LLMRequest(prompt="Hello", provider=ProviderType.GEMINI)
        key = self.generator.generate_request_key(request)

        hash_value = self.generator.extract_hash_from_key(key)
        assert hash_value is not None
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

        # Invalid key should return None
        invalid_hash = self.generator.extract_hash_from_key("invalid_key")
        assert invalid_hash is None

    def test_metadata_normalization(self):
        """Test metadata normalization for consistent hashing."""
        # Nested metadata should be normalized
        metadata1 = {"nested": {"b": 2, "a": 1}, "simple": "value"}
        metadata2 = {"simple": "value", "nested": {"a": 1, "b": 2}}

        normalized1 = self.generator._normalize_metadata(metadata1)
        normalized2 = self.generator._normalize_metadata(metadata2)

        assert normalized1 == normalized2

    def test_metadata_list_handling(self):
        """Test handling of lists in metadata."""
        # Sortable lists should be sorted
        request1 = LLMRequest(
            prompt="Hello",
            provider=ProviderType.GEMINI,
            metadata={"tags": ["b", "a", "c"]},
        )
        request2 = LLMRequest(
            prompt="Hello",
            provider=ProviderType.GEMINI,
            metadata={"tags": ["a", "b", "c"]},
        )

        key1 = self.generator.generate_request_key(request1)
        key2 = self.generator.generate_request_key(request2)

        assert key1 == key2

    def test_custom_prefix(self):
        """Test custom key prefix."""
        custom_generator = CacheKeyGenerator(prefix="custom")

        request = LLMRequest(prompt="Hello", provider=ProviderType.GEMINI)
        key = custom_generator.generate_request_key(request)

        assert key.startswith("custom:")
        assert custom_generator.validate_key_format(key)

        # Both generators validate the same format structure
        default_generator = CacheKeyGenerator()
        assert default_generator.validate_key_format(key)  # Structure is valid

        # But prefixes should differ
        assert custom_generator.get_key_prefix(key) == "custom"

        # Default generator key should have different prefix
        default_key = default_generator.generate_request_key(request)
        assert default_generator.get_key_prefix(default_key) == "req"

    def test_include_metadata_flag(self):
        """Test include_metadata configuration."""
        request = LLMRequest(
            prompt="Hello",
            provider=ProviderType.GEMINI,
            metadata={"important": "data"},
        )

        # Generator with metadata
        gen_with_metadata = CacheKeyGenerator(include_metadata=True)
        key_with = gen_with_metadata.generate_request_key(request)

        # Generator without metadata
        gen_without_metadata = CacheKeyGenerator(include_metadata=False)
        key_without = gen_without_metadata.generate_request_key(request)

        # Keys should be different
        assert key_with != key_without

        # Request without metadata should generate same key for both generators
        request_no_metadata = LLMRequest(prompt="Hello", provider=ProviderType.GEMINI)
        key_with_no_meta = gen_with_metadata.generate_request_key(request_no_metadata)
        key_without_no_meta = gen_without_metadata.generate_request_key(
            request_no_metadata
        )

        assert key_with_no_meta == key_without_no_meta
