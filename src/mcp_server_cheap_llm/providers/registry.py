"""Provider registry and factory for dynamic provider loading.

This module implements the provider registration system and factory pattern
for dynamically loading and managing LLM providers.
"""

from typing import Any

from mcp_server_cheap_llm.core.errors import ProviderError, ValidationError

from .base import LLMProvider


class ProviderRegistry:
    """Registry for LLM providers with factory pattern.

    Manages provider registration, validation, and instantiation.
    Supports dynamic loading and configuration validation.
    """

    _providers: dict[str, type[LLMProvider]] = {}
    _instances: dict[str, LLMProvider] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[LLMProvider]) -> None:
        """Register a provider class.

        Args:
            name: Provider name (e.g., "gemini", "openai")
            provider_class: Provider class implementing LLMProvider

        Raises:
            ValidationError: If provider class is invalid
        """
        if not issubclass(provider_class, LLMProvider):
            raise ValidationError(
                f"Provider class {provider_class.__name__} must inherit from LLMProvider"
            )

        cls._providers[name] = provider_class

    @classmethod
    def get_provider_class(cls, name: str) -> type[LLMProvider]:
        """Get registered provider class.

        Args:
            name: Provider name

        Returns:
            Type[LLMProvider]: Provider class

        Raises:
            ProviderError: If provider not found
        """
        if name not in cls._providers:
            available = list(cls._providers.keys())
            raise ProviderError(
                f"Provider '{name}' not registered. Available: {available}",
                provider=name,
            )

        return cls._providers[name]

    @classmethod
    def create_provider(cls, name: str, config: dict[str, Any]) -> LLMProvider:
        """Create provider instance.

        Args:
            name: Provider name
            config: Provider configuration

        Returns:
            LLMProvider: Configured provider instance

        Raises:
            ProviderError: If provider creation fails
            ValidationError: If configuration is invalid
        """
        provider_class = cls.get_provider_class(name)

        try:
            instance = provider_class(config)
            cls._instances[name] = instance
            return instance
        except Exception as e:
            raise ProviderError(
                f"Failed to create provider '{name}': {e}", provider=name
            ) from e

    @classmethod
    def get_instance(cls, name: str) -> LLMProvider | None:
        """Get existing provider instance.

        Args:
            name: Provider name

        Returns:
            Optional[LLMProvider]: Provider instance if exists
        """
        return cls._instances.get(name)

    @classmethod
    def list_providers(cls) -> dict[str, type[LLMProvider]]:
        """List all registered providers.

        Returns:
            Dict[str, Type[LLMProvider]]: Registered providers
        """
        return cls._providers.copy()

    @classmethod
    def list_instances(cls) -> dict[str, LLMProvider]:
        """List all provider instances.

        Returns:
            Dict[str, LLMProvider]: Active provider instances
        """
        return cls._instances.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers and instances.

        Useful for testing and cleanup.
        """
        cls._providers.clear()
        cls._instances.clear()


# Convenience functions for global registry
def register_provider(name: str, provider_class: type[LLMProvider]) -> None:
    """Register a provider class globally.

    Args:
        name: Provider name
        provider_class: Provider class
    """
    ProviderRegistry.register(name, provider_class)


def get_provider(name: str, config: dict[str, Any] | None = None) -> LLMProvider:
    """Get or create a provider instance.

    Args:
        name: Provider name
        config: Provider configuration (required for new instances)

    Returns:
        LLMProvider: Provider instance

    Raises:
        ProviderError: If provider not found or creation fails
        ValidationError: If configuration is required but not provided
    """
    # Try to get existing instance first
    instance = ProviderRegistry.get_instance(name)
    if instance is not None:
        return instance

    # Create new instance if config provided
    if config is None:
        raise ValidationError(
            f"Configuration required to create new instance of provider '{name}'"
        )

    return ProviderRegistry.create_provider(name, config)


def list_available_providers() -> dict[str, type[LLMProvider]]:
    """List all available providers.

    Returns:
        Dict[str, Type[LLMProvider]]: Available providers
    """
    return ProviderRegistry.list_providers()
