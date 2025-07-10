"""Configuration management for MCP Server Cheap LLM.

This module handles configuration loading, validation, and provider management.
Follows atomic design with configuration-driven approach (200-300 lines).

Key classes:
    ConfigManager: Main configuration management
    ConfigValidator: Validates configuration data
    EnvironmentLoader: Loads environment variables

Example:
    >>> manager = ConfigManager("/path/to/config.toml")
    >>> providers = manager.get_enabled_providers()
    >>> config = manager.get_provider_config("gemini")
"""

import base64
import os
import re
from pathlib import Path
from typing import Any, Literal

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef,import-not-found]

import structlog  # type: ignore[import-not-found]
from cryptography.fernet import Fernet  # type: ignore[import-not-found]
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

from mcp_server_cheap_llm.core.models import ProviderConfig as CoreProviderConfig
from mcp_server_cheap_llm.utils.errors import ConfigurationError

logger = structlog.get_logger(__name__)


# Define allowed log levels and providers
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
ProviderType = Literal["openai", "google", "anthropic", "llama", "codex"]


class ConfigModel(BaseModel):
    """Basic configuration model for the MCP server.

    This model represents the core configuration for the MCP server,
    including server identification, logging settings, and provider configuration.

    Attributes:
        server_name: Unique identifier for this server instance
        log_level: Logging verbosity level
        enabled_providers: List of provider names that are enabled
        default_provider: The default provider to use for requests
        max_retries: Maximum number of retry attempts for failed requests
        timeout: Global timeout for requests in seconds

    Example:
        >>> config = ConfigModel(
        ...     server_name="mcp-cheap-llm",
        ...     enabled_providers=["openai", "google"],
        ...     default_provider="openai"
        ... )
    """

    server_name: str = Field(
        ..., description="Unique identifier for this server instance", min_length=1
    )
    log_level: LogLevel = Field(default="INFO", description="Logging verbosity level")
    enabled_providers: list[str] = Field(
        ..., description="List of provider names that are enabled", min_length=1
    )
    default_provider: str = Field(
        ..., description="The default provider to use for requests"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for failed requests",
        ge=0,
        le=10,
    )
    timeout: int = Field(
        default=30, description="Global timeout for requests in seconds", ge=1, le=300
    )

    @model_validator(mode="after")
    def validate_default_provider(self) -> "ConfigModel":
        """Ensure default_provider is in enabled_providers."""
        if self.default_provider not in self.enabled_providers:
            raise ValueError("default_provider must be one of enabled_providers")
        return self


class APIKeyConfig(BaseModel):
    """Configuration for API keys with encryption support.

    This model manages API key storage and validation for different
    LLM providers, with support for encryption at rest.

    Attributes:
        provider: The LLM provider this key is for
        api_key: The actual API key (may be encrypted)
        is_encrypted: Whether the API key is encrypted

    Example:
        >>> key_config = APIKeyConfig(
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     is_encrypted=False
        ... )
    """

    provider: ProviderType = Field(..., description="The LLM provider this key is for")
    api_key: str = Field(
        ..., description="The actual API key (may be encrypted)", min_length=1
    )
    is_encrypted: bool = Field(
        default=False, description="Whether the API key is encrypted"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Ensure API key is not empty and strip whitespace."""
        if not v or not v.strip():
            raise ValueError("api_key cannot be empty")
        return v.strip()


class ProviderConfig(BaseModel):
    """Provider-specific configuration with quotas and rate limits.

    This model defines configuration for individual LLM providers,
    including rate limiting, quotas, and model-specific settings.

    Attributes:
        name: Unique name for this provider configuration
        endpoint: Optional custom API endpoint URL
        rate_limit: Maximum requests per minute
        quota_limit: Maximum tokens per month
        enabled: Whether this provider is enabled
        timeout: Request timeout in seconds
        model_settings: Model-specific configuration overrides

    Example:
        >>> provider = ProviderConfig(
        ...     name="openai",
        ...     endpoint="https://api.openai.com/v1",
        ...     rate_limit=100,
        ...     model_settings={
        ...         "gpt-4": {"max_tokens": 8192}
        ...     }
        ... )
    """

    name: str = Field(
        ..., description="Unique name for this provider configuration", min_length=1
    )
    endpoint: str | None = Field(
        default=None, description="Optional custom API endpoint URL"
    )
    rate_limit: int = Field(
        default=60, description="Maximum requests per minute", gt=0, le=1000
    )
    quota_limit: int = Field(
        default=1000000, description="Maximum tokens per month", gt=0
    )
    enabled: bool = Field(default=True, description="Whether this provider is enabled")
    timeout: int = Field(
        default=30, description="Request timeout in seconds", ge=1, le=300
    )
    model_settings: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Model-specific configuration overrides"
    )

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str | None) -> str | None:
        """Validate endpoint is a proper URL if provided."""
        if v is not None:
            if not v.startswith(("http://", "https://")):
                raise ValueError(
                    "endpoint must be a valid URL starting with http:// or https://"
                )
            # Basic URL validation
            if len(v) < 10 or "." not in v[8:]:  # After https://
                raise ValueError("endpoint must be a valid URL")
        return v

    @field_validator("rate_limit")
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        """Ensure rate limit is positive."""
        if v <= 0:
            raise ValueError("rate_limit must be positive")
        return v


class ServerConfig(BaseModel):
    """Main server configuration model.

    Attributes:
        default_provider: Default provider to use
        max_concurrent_requests: Maximum concurrent requests
        request_timeout_seconds: Global request timeout
        enable_metrics: Whether to collect metrics
        log_level: Logging level
        providers: List of provider configurations

    Example:
        >>> config = ServerConfig(
        ...     default_provider="gemini",
        ...     providers=[provider_config1, provider_config2]
        ... )
    """

    default_provider: str = "gemini"
    max_concurrent_requests: int = Field(default=10, ge=1, le=100)
    request_timeout_seconds: int = Field(default=30, ge=1, le=300)
    enable_metrics: bool = True
    log_level: str = Field(
        default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    providers: list[CoreProviderConfig] = Field(default_factory=list)

    def get_debug_state(self) -> dict[str, Any]:
        """Return configuration state for debugging.

        Returns:
            Dictionary with configuration information (no sensitive data)
        """
        return {
            "default_provider": self.default_provider,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout_seconds": self.request_timeout_seconds,
            "enable_metrics": self.enable_metrics,
            "log_level": self.log_level,
            "provider_count": len(self.providers),
            "enabled_providers": [p.name for p in self.providers if p.enabled],
            "provider_types": list({p.provider_type for p in self.providers}),
        }


class APIKeyManager:
    """Manages API key encryption, validation, and secure storage.

    This class provides secure handling of API keys with encryption at rest,
    validation for different providers, and key rotation capabilities.

    Attributes:
        _encryption_key: The Fernet encryption key
        _cipher: The Fernet cipher instance
        _encrypted_keys: Dictionary storing encrypted API keys by provider

    Example:
        >>> manager = APIKeyManager()
        >>> manager.store_encrypted_key("openai", "sk-...")
        >>> key = manager.get_decrypted_key("openai")
    """

    def __init__(self, encryption_key: bytes | None = None):
        """Initialize API key manager with optional encryption key.

        Args:
            encryption_key: Optional encryption key. If None, generates new key.
        """
        if encryption_key is None:
            self._encryption_key = Fernet.generate_key()
        else:
            self._encryption_key = encryption_key

        self._cipher = Fernet(self._encryption_key)
        self._encrypted_keys: dict[str, str] = {}

        logger.debug("APIKeyManager initialized with encryption support")

    @staticmethod
    def generate_encryption_key() -> bytes:
        """Generate a new Fernet encryption key.

        Returns:
            Base64-encoded Fernet encryption key

        Example:
            >>> key = APIKeyManager.generate_encryption_key()
            >>> manager = APIKeyManager(encryption_key=key)
        """
        return Fernet.generate_key()

    def encrypt_key(self, api_key: str) -> str:
        """Encrypt an API key for secure storage.

        Args:
            api_key: The plaintext API key to encrypt

        Returns:
            Base64-encoded encrypted API key

        Example:
            >>> encrypted = manager.encrypt_key("sk-...")
        """
        try:
            key_bytes = api_key.encode("utf-8")
            encrypted_bytes = self._cipher.encrypt(key_bytes)
            return base64.b64encode(encrypted_bytes).decode("utf-8")
        except Exception as e:
            logger.error("Failed to encrypt API key", error=str(e))
            raise ConfigurationError(f"Failed to encrypt API key: {e}") from e

    def decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt an encrypted API key.

        Args:
            encrypted_key: Base64-encoded encrypted API key

        Returns:
            The plaintext API key

        Raises:
            ConfigurationError: If decryption fails

        Example:
            >>> plaintext = manager.decrypt_key(encrypted_key)
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_key.encode("utf-8"))
            decrypted_bytes = self._cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode("utf-8")
        except Exception as e:
            logger.error("Failed to decrypt API key", error=str(e))
            raise ConfigurationError(f"Failed to decrypt API key: {e}") from e

    def validate_api_key(self, api_key: str, provider: str) -> bool:
        """Validate API key format for specific provider.

        Args:
            api_key: The API key to validate
            provider: The provider name (openai, google, anthropic)

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If provider is not supported

        Example:
            >>> is_valid = manager.validate_api_key("sk-...", "openai")
        """
        if not api_key or not api_key.strip():
            return False

        api_key = api_key.strip()

        if provider.lower() == "openai":
            # OpenAI keys start with "sk-" and are typically 51+ characters
            return api_key.startswith(("sk-", "sk-proj-")) and len(api_key) >= 20

        elif provider.lower() == "google":
            # Google API keys start with "AIza" and are typically 39 characters
            return (
                api_key.startswith("AIzaSy")
                and len(api_key) >= 20
                and bool(re.match(r"^AIzaSy[A-Za-z0-9_-]+$", api_key))
            )

        elif provider.lower() == "anthropic":
            # Anthropic keys start with "sk-ant-api03-"
            return api_key.startswith("sk-ant-api03-") and len(api_key) >= 30

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def store_encrypted_key(self, provider: str, api_key: str) -> None:
        """Store an API key in encrypted form.

        Args:
            provider: The provider name
            api_key: The plaintext API key

        Raises:
            ConfigurationError: If the API key is invalid

        Example:
            >>> manager.store_encrypted_key("openai", "sk-...")
        """
        if not self.validate_api_key(api_key, provider):
            raise ConfigurationError(f"Invalid API key format for provider: {provider}")

        encrypted_key = self.encrypt_key(api_key)
        self._encrypted_keys[provider] = encrypted_key

        logger.info("API key stored securely", provider=provider)

    def get_decrypted_key(self, provider: str) -> str | None:
        """Retrieve and decrypt an API key.

        Args:
            provider: The provider name

        Returns:
            The decrypted API key or None if not found

        Example:
            >>> key = manager.get_decrypted_key("openai")
        """
        encrypted_key = self._encrypted_keys.get(provider)
        if encrypted_key is None:
            return None

        try:
            return self.decrypt_key(encrypted_key)
        except ConfigurationError:
            logger.error("Failed to decrypt stored key", provider=provider)
            return None

    def rotate_encryption_key(self) -> None:
        """Rotate the encryption key, re-encrypting all stored keys.

        This method creates a new encryption key and re-encrypts all
        stored API keys with the new key.

        Example:
            >>> manager.rotate_encryption_key()
        """
        # First, decrypt all existing keys
        decrypted_keys = {}
        for provider, encrypted_key in self._encrypted_keys.items():
            try:
                decrypted_keys[provider] = self.decrypt_key(encrypted_key)
            except ConfigurationError:
                logger.error("Failed to decrypt key during rotation", provider=provider)
                continue

        # Generate new encryption key and cipher
        self._encryption_key = Fernet.generate_key()
        self._cipher = Fernet(self._encryption_key)

        # Re-encrypt all keys with new encryption key
        self._encrypted_keys.clear()
        for provider, api_key in decrypted_keys.items():
            encrypted_key = self.encrypt_key(api_key)
            self._encrypted_keys[provider] = encrypted_key

        logger.info(
            "Encryption key rotated successfully",
            re_encrypted_count=len(decrypted_keys),
        )

    def list_stored_providers(self) -> list[str]:
        """Get list of providers with stored keys.

        Returns:
            List of provider names that have stored keys

        Example:
            >>> providers = manager.list_stored_providers()
        """
        return list(self._encrypted_keys.keys())

    def remove_stored_key(self, provider: str) -> bool:
        """Remove a stored encrypted key.

        Args:
            provider: The provider name

        Returns:
            True if key was removed, False if not found

        Example:
            >>> removed = manager.remove_stored_key("openai")
        """
        if provider in self._encrypted_keys:
            del self._encrypted_keys[provider]
            logger.info("API key removed", provider=provider)
            return True
        return False

    def clear_all_keys(self) -> None:
        """Clear all stored encrypted keys.

        Example:
            >>> manager.clear_all_keys()
        """
        count = len(self._encrypted_keys)
        self._encrypted_keys.clear()
        logger.info("All API keys cleared", cleared_count=count)

    def key_exists(self, provider: str) -> bool:
        """Check if a key exists for the provider.

        Args:
            provider: The provider name

        Returns:
            True if key exists, False otherwise

        Example:
            >>> exists = manager.key_exists("openai")
        """
        return provider in self._encrypted_keys

    def save_encryption_key(self, file_path: str) -> None:
        """Save the encryption key to a file.

        Args:
            file_path: Path to save the encryption key

        Example:
            >>> manager.save_encryption_key("/secure/path/key.bin")
        """
        try:
            with open(file_path, "wb") as f:
                f.write(self._encryption_key)
            logger.info("Encryption key saved", path=file_path)
        except Exception as e:
            raise ConfigurationError(f"Failed to save encryption key: {e}") from e

    def load_encryption_key(self, file_path: str) -> None:
        """Load encryption key from a file.

        Args:
            file_path: Path to load the encryption key from

        Example:
            >>> manager.load_encryption_key("/secure/path/key.bin")
        """
        try:
            with open(file_path, "rb") as f:
                self._encryption_key = f.read()
            self._cipher = Fernet(self._encryption_key)
            logger.info("Encryption key loaded", path=file_path)
        except Exception as e:
            raise ConfigurationError(f"Failed to load encryption key: {e}") from e

    def load_from_environment(self, provider: str) -> str | None:
        """Load encrypted API key from environment variables.

        Args:
            provider: The provider name

        Returns:
            Decrypted API key or None if not found

        Example:
            >>> key = manager.load_from_environment("openai")
        """
        env_var_name = f"{provider.upper()}_API_KEY_ENCRYPTED"
        encrypted_key = os.getenv(env_var_name)

        if encrypted_key:
            try:
                return self.decrypt_key(encrypted_key)
            except ConfigurationError:
                logger.error(
                    "Failed to decrypt environment key",
                    provider=provider,
                    env_var=env_var_name,
                )
                return None

        return None


class EnvironmentLoader:
    """Loads configuration from environment variables.

    This utility class provides methods to safely load configuration
    from environment variables with proper validation.
    """

    @staticmethod
    def get_api_key(provider_name: str) -> str | None:
        """Get API key for provider from environment.

        Args:
            provider_name: Name of the provider

        Returns:
            API key if found, None otherwise

        Example:
            >>> key = EnvironmentLoader.get_api_key("gemini")
            >>> # Looks for GEMINI_API_KEY, GEMINI_KEY, etc.
        """
        possible_keys = [
            f"{provider_name.upper()}_API_KEY",
            f"{provider_name.upper()}_KEY",
            f"MCP_{provider_name.upper()}_API_KEY",
        ]

        for key in possible_keys:
            value = os.getenv(key)
            if value:
                logger.debug(
                    "Found API key for provider", provider=provider_name, env_var=key
                )
                return value

        logger.warning(
            "No API key found for provider",
            provider=provider_name,
            checked_vars=possible_keys,
        )
        return None

    @staticmethod
    def get_server_config() -> dict[str, Any]:
        """Load server configuration from environment variables.

        Returns:
            Dictionary with server configuration values

        Example:
            >>> config = EnvironmentLoader.get_server_config()
            >>> print(config['log_level'])
        """
        return {
            "default_provider": os.getenv("MCP_DEFAULT_PROVIDER", "gemini"),
            "max_concurrent_requests": int(os.getenv("MCP_MAX_CONCURRENT", "10")),
            "request_timeout_seconds": int(os.getenv("MCP_REQUEST_TIMEOUT", "30")),
            "enable_metrics": os.getenv("MCP_ENABLE_METRICS", "true").lower() == "true",
            "log_level": os.getenv("MCP_LOG_LEVEL", "INFO").upper(),
        }


class ConfigManager:
    """Manages configuration loading and provider access.

    This class handles loading configuration from files and environment,
    validates settings, and provides access to provider configurations.

    Attributes:
        config: The loaded and validated server configuration

    Example:
        >>> manager = ConfigManager("/path/to/config.toml")
        >>> providers = manager.get_enabled_providers()
        >>> config = manager.get_provider_config("gemini")
    """

    def __init__(self, config_path: str | None = None):
        """Initialize configuration manager.

        Args:
            config_path: Optional path to configuration file

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config_path = config_path
        self.config = self._load_configuration()
        logger.info(
            "Configuration loaded successfully",
            provider_count=len(self.config.providers),
            enabled_count=len(self.get_enabled_providers()),
        )

    def _load_configuration(self) -> ServerConfig:
        """Load and validate configuration from file and environment.

        Returns:
            Validated ServerConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Start with environment configuration
            config_data = EnvironmentLoader.get_server_config()

            # Load from file if provided
            if self.config_path:
                file_config = self._load_config_file()
                config_data.update(file_config)

            # Create default providers if none specified
            if "providers" not in config_data:
                config_data["providers"] = self._create_default_providers()

            # Validate and create configuration
            return ServerConfig(**config_data)

        except ValidationError as e:
            raise ConfigurationError(f"Invalid configuration: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}") from e

    def _load_config_file(self) -> dict[str, Any]:
        """Load configuration from TOML file.

        Returns:
            Dictionary with configuration data

        Raises:
            ConfigurationError: If file cannot be loaded
        """
        if self.config_path is None:
            raise ConfigurationError("No configuration file path provided")

        config_path = Path(self.config_path)

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse configuration file: {e}") from e

    def _create_default_providers(self) -> list[dict[str, Any]]:
        """Create default provider configurations.

        Returns:
            List of default provider configurations
        """
        defaults = []

        # Gemini provider
        gemini_key = EnvironmentLoader.get_api_key("gemini")
        if gemini_key:
            defaults.append(
                {
                    "name": "gemini",
                    "provider_type": "gemini",
                    "enabled": True,
                    "api_key": gemini_key,
                    "model_name": "gemini-pro",
                }
            )

        # Codex provider
        openai_key = EnvironmentLoader.get_api_key("openai")
        if openai_key:
            defaults.append(
                {
                    "name": "codex",
                    "provider_type": "codex",
                    "enabled": True,
                    "api_key": openai_key,
                    "model_name": "gpt-3.5-turbo-instruct",
                }
            )

        # LLaMA provider (no API key needed for local)
        defaults.append(
            {
                "name": "llama",
                "provider_type": "llama",
                "enabled": True,
                "model_name": "llama-2-7b-chat",
            }
        )

        logger.info("Created default providers", count=len(defaults))
        return defaults

    def get_enabled_providers(self) -> list[str]:
        """Get list of enabled provider names.

        Returns:
            List of enabled provider names
        """
        return [p.name for p in self.config.providers if p.enabled]

    def get_provider_config(self, provider_name: str) -> CoreProviderConfig | None:
        """Get configuration for specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderConfig if found, None otherwise
        """
        for provider in self.config.providers:
            if provider.name == provider_name:
                return provider
        return None

    def get_default_provider(self) -> str:
        """Get the default provider name.

        Returns:
            Default provider name
        """
        return self.config.default_provider

    def get_debug_state(self) -> dict[str, Any]:
        """Get complete configuration state for debugging.

        Returns:
            Dictionary with configuration state information
        """
        return {
            "config_path": self.config_path,
            "server_config": self.config.get_debug_state(),
            "provider_details": [
                {
                    "name": p.name,
                    "type": p.provider_type,
                    "enabled": p.enabled,
                    "model": p.model_name,
                    "has_api_key": p.api_key is not None,
                }
                for p in self.config.providers
            ],
        }
