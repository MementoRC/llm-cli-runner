"""Provider system for LLM integrations.

This module provides the abstract provider interface and registration system
for integrating different LLM providers (Gemini, OpenAI, LLaMA, etc.).

Key components:
    LLMProvider: Abstract base class for all providers
    ProviderRegistry: Factory for dynamic provider loading
    CircuitBreaker: Failure handling and recovery
"""

from .base import LLMProvider, ProviderCapabilities, ProviderMetadata
from .registry import ProviderRegistry, get_provider, register_provider

__all__ = [
    "LLMProvider",
    "ProviderCapabilities",
    "ProviderMetadata",
    "ProviderRegistry",
    "get_provider",
    "register_provider",
]
