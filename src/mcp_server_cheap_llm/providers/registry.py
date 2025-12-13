"""Provider registration and factory system.

This module implements a registry pattern for managing LLM providers,
allowing dynamic registration and instantiation of provider implementations.

Key classes:
    ProviderRegistry: Central registry for provider management

Example:
    >>> registry = ProviderRegistry()
    >>> registry.register_provider(GeminiProvider)
    >>> provider = registry.create_provider(config)

"""

from typing import Any

from mcp_server_cheap_llm.core.errors import ConfigurationError, ProviderError
from mcp_server_cheap_llm.core.models import ProviderConfig, ProviderType
from mcp_server_cheap_llm.utils.logging import StructuredLogger

from .base import LLMProvider


class ProviderRegistry:
    """Central registry for LLM provider management.

    This class implements the registry pattern for provider management,
    allowing dynamic registration, instantiation, and lookup of providers.

    Attributes:
        _providers: Dictionary mapping provider types to classes
        _instances: Cache of provider instances
        logger: Structured logger instance

    Example:
        >>> registry = ProviderRegistry()
        >>>
        >>> # Register a provider class
        >>> registry.register_provider(GeminiProvider)
        >>>
        >>> # Create provider instance
        >>> config = ProviderConfig(name="my_gemini", provider_type=ProviderType.GEMINI)
        >>> provider = registry.create_provider(config)
        >>>
        >>> # Get existing provider
        >>> same_provider = registry.get_provider("my_gemini")

    """

    def __init__(self) -> None:
        """Initialize provider registry."""
        self._providers: dict[ProviderType, type[LLMProvider]] = {}
        self._instances: dict[str, LLMProvider] = {}
        self.logger = StructuredLogger(__name__)

    def register_provider(self, provider_class: type[LLMProvider]) -> None:
        """Register a provider class.

        Args:
            provider_class: Provider class to register

        Raises:
            ProviderError: If provider class is invalid or already registered

        """
        if not issubclass(provider_class, LLMProvider):
            msg = f"Provider class {provider_class.__name__} must inherit from LLMProvider"
            raise ProviderError(
                msg,
                provider=provider_class.__name__,
                error_code="INVALID_PROVIDER_CLASS",
                context={"class_name": provider_class.__name__},
            )

        # Extract provider type from class (should be set as class attribute)
        if not hasattr(provider_class, "PROVIDER_TYPE"):
            msg = f"Provider class {provider_class.__name__} must define PROVIDER_TYPE"
            raise ProviderError(
                msg,
                provider=provider_class.__name__,
                error_code="MISSING_PROVIDER_TYPE",
                context={"class_name": provider_class.__name__},
            )

        provider_type = provider_class.PROVIDER_TYPE  # type: ignore

        if provider_type in self._providers:
            existing_class = self._providers[provider_type]
            if existing_class != provider_class:
                self.logger.warning(
                    "Overriding existing provider registration",
                    extra={
                        "provider_type": provider_type,
                        "existing_class": existing_class.__name__,
                        "new_class": provider_class.__name__,
                    },
                )

        self._providers[provider_type] = provider_class

        self.logger.info(
            "Provider registered successfully",
            extra={
                "provider_type": provider_type,
                "class_name": provider_class.__name__,
            },
        )

    def get_provider_class(self, provider_type: ProviderType) -> type[LLMProvider]:
        """Get provider class by type.

        Args:
            provider_type: Type of provider to get

        Returns:
            Provider class

        Raises:
            ProviderError: If provider type not registered

        """
        if provider_type not in self._providers:
            available_types = list(self._providers.keys())
            msg = f"Provider type {provider_type} not registered. Available: {available_types}"
            raise ProviderError(
                msg,
                provider=str(provider_type),
                error_code="PROVIDER_NOT_REGISTERED",
                context={
                    "requested_type": provider_type,
                    "available_types": available_types,
                },
            )

        return self._providers[provider_type]

    def create_provider(self, config: ProviderConfig) -> LLMProvider:
        """Create provider instance from configuration.

        This is the main factory method that creates provider instances.
        It handles caching to ensure only one instance per provider name.

        Args:
            config: Provider configuration

        Returns:
            Provider instance

        Raises:
            ProviderError: If provider creation fails
            ConfigurationError: If configuration is invalid

        """
        # Check if instance already exists
        if config.name in self._instances:
            existing_instance = self._instances[config.name]

            # Validate that existing instance matches requested type
            if existing_instance.provider_type != config.provider_type:
                msg = (
                    f"Provider {config.name} already exists with different type "
                    f"(existing: {existing_instance.provider_type}, requested: {config.provider_type})"
                )
                raise ProviderError(
                    msg,
                    provider=config.name,
                    error_code="PROVIDER_TYPE_MISMATCH",
                    context={
                        "provider_name": config.name,
                        "existing_type": existing_instance.provider_type,
                        "requested_type": config.provider_type,
                    },
                )

            self.logger.debug(
                "Returning existing provider instance",
                extra={
                    "provider_name": config.name,
                    "provider_type": config.provider_type,
                },
            )
            return existing_instance

        # Get provider class
        provider_class = self.get_provider_class(config.provider_type)

        # Create new instance
        try:
            instance = provider_class(config)
            self._instances[config.name] = instance

            self.logger.info(
                "Provider instance created successfully",
                extra={
                    "provider_name": config.name,
                    "provider_type": config.provider_type,
                    "class_name": provider_class.__name__,
                },
            )

            return instance

        except Exception as e:
            self.logger.exception(
                "Failed to create provider instance",
                extra={
                    "provider_name": config.name,
                    "provider_type": config.provider_type,
                    "error": str(e),
                },
            )

            if isinstance(e, ProviderError | ConfigurationError):
                raise

            msg = f"Failed to create provider {config.name}: {e!s}"
            raise ProviderError(
                msg,
                provider=config.name,
                error_code="PROVIDER_CREATION_FAILED",
                context={
                    "provider_name": config.name,
                    "provider_type": config.provider_type,
                    "error": str(e),
                },
            ) from e

    def get_provider(self, name: str) -> LLMProvider | None:
        """Get existing provider instance by name.

        Args:
            name: Provider name to lookup

        Returns:
            Provider instance or None if not found

        """
        return self._instances.get(name)

    def list_providers(self) -> list[str]:
        """List names of all registered provider instances.

        Returns:
            List of provider names

        """
        return list(self._instances.keys())

    def list_available_types(self) -> list[ProviderType]:
        """List all registered provider types.

        Returns:
            List of available provider types

        """
        return list(self._providers.keys())

    def remove_provider(self, name: str) -> bool:
        """Remove provider instance from registry.

        Args:
            name: Name of provider to remove

        Returns:
            True if provider was removed, False if not found

        """
        if name in self._instances:
            removed_provider = self._instances.pop(name)

            self.logger.info(
                "Provider instance removed from registry",
                extra={
                    "provider_name": name,
                    "provider_type": removed_provider.provider_type,
                },
            )
            return True

        return False

    def get_registry_status(self) -> dict[str, Any]:
        """Get comprehensive registry status.

        Returns:
            Dictionary with registry status information

        """
        instance_info = []
        for name, instance in self._instances.items():
            instance_info.append(
                {
                    "name": name,
                    "type": instance.provider_type,
                    "class": instance.__class__.__name__,
                    "health": instance.get_health_status(),
                },
            )

        return {
            "registered_types": list(self._providers.keys()),
            "active_instances": len(self._instances),
            "instance_details": instance_info,
            "total_registered_classes": len(self._providers),
        }

    def clear_instances(self) -> None:
        """Clear all provider instances.

        This method removes all cached provider instances but keeps
        registered provider classes. Useful for testing or reset scenarios.
        """
        instance_count = len(self._instances)
        self._instances.clear()

        self.logger.info(
            "All provider instances cleared from registry",
            extra={"cleared_count": instance_count},
        )
