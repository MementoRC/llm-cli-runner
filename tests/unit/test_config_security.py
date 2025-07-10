"""Unit tests for API key management and encryption."""

import os
import tempfile
from unittest.mock import patch

import pytest

from mcp_server_cheap_llm.utils.config import APIKeyManager
from mcp_server_cheap_llm.utils.errors import ConfigurationError


class TestAPIKeyManager:
    """Test suite for APIKeyManager functionality."""

    def test_generate_encryption_key(self):
        """Test encryption key generation."""
        manager = APIKeyManager()
        key = manager.generate_encryption_key()
        assert isinstance(key, bytes)
        assert len(key) == 44  # Base64 encoded Fernet key length

    def test_encrypt_decrypt_api_key(self):
        """Test API key encryption and decryption."""
        manager = APIKeyManager()
        original_key = "sk-test123456789abcdef"

        # Encrypt the key
        encrypted_key = manager.encrypt_key(original_key)
        assert encrypted_key != original_key
        assert isinstance(encrypted_key, str)

        # Decrypt the key
        decrypted_key = manager.decrypt_key(encrypted_key)
        assert decrypted_key == original_key

    def test_encrypt_decrypt_with_custom_key(self):
        """Test encryption/decryption with custom encryption key."""
        encryption_key = APIKeyManager.generate_encryption_key()
        manager = APIKeyManager(encryption_key=encryption_key)

        original_key = "test-api-key-12345"
        encrypted_key = manager.encrypt_key(original_key)
        decrypted_key = manager.decrypt_key(encrypted_key)

        assert decrypted_key == original_key

    def test_validate_openai_api_key_valid(self):
        """Test OpenAI API key validation with valid keys."""
        manager = APIKeyManager()

        valid_keys = [
            "sk-1234567890abcdef1234567890abcdef12345678",
            "sk-proj-1234567890abcdef1234567890abcdef12345678",
        ]

        for key in valid_keys:
            assert manager.validate_api_key(key, "openai") is True

    def test_validate_openai_api_key_invalid(self):
        """Test OpenAI API key validation with invalid keys."""
        manager = APIKeyManager()

        invalid_keys = [
            "invalid-key",
            "sk-",
            "sk-short",
            "not-sk-prefix",
            "",
        ]

        for key in invalid_keys:
            assert manager.validate_api_key(key, "openai") is False

    def test_validate_google_api_key_valid(self):
        """Test Google API key validation with valid keys."""
        manager = APIKeyManager()

        valid_keys = [
            "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567",
            "AIzaSyAnotherValidGoogleAPIKey123456789",
        ]

        for key in valid_keys:
            assert manager.validate_api_key(key, "google") is True

    def test_validate_google_api_key_invalid(self):
        """Test Google API key validation with invalid keys."""
        manager = APIKeyManager()

        invalid_keys = [
            "invalid-key",
            "AIza",
            "NotAIza-prefix",
            "",
            "AIzaSy",  # Too short
        ]

        for key in invalid_keys:
            assert manager.validate_api_key(key, "google") is False

    def test_validate_anthropic_api_key_valid(self):
        """Test Anthropic API key validation with valid keys."""
        manager = APIKeyManager()

        valid_keys = [
            "sk-ant-api03-1234567890abcdef1234567890abcdef",
            "sk-ant-api03-anothervalidkey1234567890abcdef",
        ]

        for key in valid_keys:
            assert manager.validate_api_key(key, "anthropic") is True

    def test_validate_anthropic_api_key_invalid(self):
        """Test Anthropic API key validation with invalid keys."""
        manager = APIKeyManager()

        invalid_keys = [
            "invalid-key",
            "sk-ant-",
            "sk-ant-api03-",
            "not-anthropic-key",
            "",
        ]

        for key in invalid_keys:
            assert manager.validate_api_key(key, "anthropic") is False

    def test_validate_unsupported_provider(self):
        """Test API key validation for unsupported provider."""
        manager = APIKeyManager()

        with pytest.raises(ValueError, match="Unsupported provider"):
            manager.validate_api_key("any-key", "unsupported")

    def test_store_and_retrieve_encrypted_key(self):
        """Test storing and retrieving encrypted API keys."""
        manager = APIKeyManager()

        original_key = "sk-test123456789abcdef"
        provider = "openai"

        # Store encrypted key
        manager.store_encrypted_key(provider, original_key)

        # Retrieve and decrypt key
        retrieved_key = manager.get_decrypted_key(provider)
        assert retrieved_key == original_key

    def test_store_invalid_key_raises_error(self):
        """Test that storing an invalid key raises an error."""
        manager = APIKeyManager()

        with pytest.raises(ConfigurationError, match="Invalid API key"):
            manager.store_encrypted_key("openai", "invalid-key")

    def test_get_nonexistent_key_returns_none(self):
        """Test retrieving a non-existent key returns None."""
        manager = APIKeyManager()

        result = manager.get_decrypted_key("nonexistent")
        assert result is None

    def test_rotate_encryption_key(self):
        """Test encryption key rotation."""
        manager = APIKeyManager()

        # Store a key with original encryption
        original_api_key = "sk-test123456789abcdef"
        manager.store_encrypted_key("openai", original_api_key)

        # Rotate the encryption key
        old_encrypted = manager._encrypted_keys["openai"]
        manager.rotate_encryption_key()

        # Verify key can still be retrieved
        retrieved_key = manager.get_decrypted_key("openai")
        assert retrieved_key == original_api_key

        # Verify the encrypted value changed
        new_encrypted = manager._encrypted_keys["openai"]
        assert old_encrypted != new_encrypted

    def test_list_stored_providers(self):
        """Test listing providers with stored keys."""
        manager = APIKeyManager()

        # Initially empty
        assert manager.list_stored_providers() == []

        # Store some keys
        manager.store_encrypted_key("openai", "sk-test123456789abcdef")
        manager.store_encrypted_key("google", "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567")

        providers = manager.list_stored_providers()
        assert set(providers) == {"openai", "google"}

    def test_remove_stored_key(self):
        """Test removing a stored encrypted key."""
        manager = APIKeyManager()

        # Store a key
        manager.store_encrypted_key("openai", "sk-test123456789abcdef")
        assert "openai" in manager.list_stored_providers()

        # Remove the key
        removed = manager.remove_stored_key("openai")
        assert removed is True
        assert "openai" not in manager.list_stored_providers()

        # Try to remove non-existent key
        removed = manager.remove_stored_key("nonexistent")
        assert removed is False

    def test_clear_all_keys(self):
        """Test clearing all stored encrypted keys."""
        manager = APIKeyManager()

        # Store multiple keys
        manager.store_encrypted_key("openai", "sk-test123456789abcdef")
        manager.store_encrypted_key("google", "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567")

        assert len(manager.list_stored_providers()) == 2

        # Clear all keys
        manager.clear_all_keys()
        assert len(manager.list_stored_providers()) == 0

    def test_key_exists(self):
        """Test checking if a key exists for a provider."""
        manager = APIKeyManager()

        assert manager.key_exists("openai") is False

        manager.store_encrypted_key("openai", "sk-test123456789abcdef")
        assert manager.key_exists("openai") is True

    def test_encryption_key_persistence(self):
        """Test that encryption key can be persisted and loaded."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            key_file = f.name

        try:
            # Create manager and save key
            manager1 = APIKeyManager()
            manager1.save_encryption_key(key_file)

            # Create new manager and load key
            manager2 = APIKeyManager()
            manager2.load_encryption_key(key_file)

            # Test that they can encrypt/decrypt each other's data
            original_key = "sk-test123456789abcdef"
            encrypted = manager1.encrypt_key(original_key)
            decrypted = manager2.decrypt_key(encrypted)

            assert decrypted == original_key

        finally:
            os.unlink(key_file)

    def test_encryption_error_handling(self):
        """Test error handling in encryption operations."""
        manager = APIKeyManager()

        # Test decryption with invalid data
        with pytest.raises(ConfigurationError, match="Failed to decrypt"):
            manager.decrypt_key("invalid-encrypted-data")

    def test_secure_memory_operations(self):
        """Test that sensitive data is handled securely."""
        manager = APIKeyManager()

        # This is a basic test - in real implementation,
        # we'd test memory zeroing and secure cleanup
        original_key = "sk-test123456789abcdef"
        encrypted = manager.encrypt_key(original_key)

        # Verify the original key isn't stored in plaintext anywhere
        assert original_key not in str(manager.__dict__.values())
        assert original_key not in encrypted

    def test_environment_integration(self):
        """Test integration with environment variable loading."""
        manager = APIKeyManager()

        # Mock environment with encrypted keys
        encrypted_openai = manager.encrypt_key("sk-test123456789abcdef")

        env_vars = {
            "OPENAI_API_KEY_ENCRYPTED": encrypted_openai,
            "MCP_ENCRYPTION_KEY": manager._encryption_key.decode(),
        }

        with patch.dict(os.environ, env_vars):
            # Test loading encrypted key from environment
            loaded_key = manager.load_from_environment("openai")
            assert loaded_key == "sk-test123456789abcdef"

    def test_multiple_provider_key_management(self):
        """Test managing keys for multiple providers simultaneously."""
        manager = APIKeyManager()

        providers_and_keys = {
            "openai": "sk-test123456789abcdef",
            "google": "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567",
            "anthropic": "sk-ant-api03-1234567890abcdef1234567890abcdef",
        }

        # Store all keys
        for provider, key in providers_and_keys.items():
            manager.store_encrypted_key(provider, key)

        # Verify all keys can be retrieved correctly
        for provider, expected_key in providers_and_keys.items():
            retrieved_key = manager.get_decrypted_key(provider)
            assert retrieved_key == expected_key

        # Test batch operations
        all_providers = manager.list_stored_providers()
        assert set(all_providers) == set(providers_and_keys.keys())
