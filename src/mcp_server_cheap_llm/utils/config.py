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

import os
from pathlib import Path
from typing import Any, Literal

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef,import-not-found]

import structlog  # type: ignore[import-not-found]
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

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
