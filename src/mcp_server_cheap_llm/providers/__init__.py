"""Provider architecture for MCP Server Cheap LLM.

This module implements the abstract provider interface and registration system.
Follows atomic design patterns with clear separation of concerns.

Key components:
    LLMProvider: Abstract base class for all providers
    ProviderRegistry: Provider registration and factory system
    CircuitBreakerMixin: Circuit breaker integration

Example:
    >>> from mcp_server_cheap_llm.providers import ProviderRegistry, LLMProvider
    >>> registry = ProviderRegistry()
    >>> provider = registry.get_provider("gemini")
"""

from .base import LLMProvider, ProviderCapabilities
from .registry import ProviderRegistry

__all__ = ["LLMProvider", "ProviderCapabilities", "ProviderRegistry"]
