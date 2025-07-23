"""Provider architecture for MCP Server Cheap LLM.

This module implements the abstract provider interface, registration system,
and intelligent routing capabilities.

Key components:
    LLMProvider: Abstract base class for all providers
    ProviderRegistry: Provider registration and factory system
    ProviderRouter: Intelligent routing based on complexity analysis
    ComplexityAnalyzer: Prompt complexity analysis system

Example:
    >>> from mcp_server_cheap_llm.providers import ProviderRegistry, ProviderRouter
    >>> registry = ProviderRegistry()
    >>> router = ProviderRouter(registry)
    >>> decision = router.route_request(request)
"""

from .base import LLMProvider, ProviderCapabilities
from .registry import ProviderRegistry
from .routing import (
    ComplexityAnalyzer,
    ComplexityFeatures,
    ComplexityLevel,
    ProviderRouter,
    RoutingDecision,
)

__all__ = [
    "LLMProvider",
    "ProviderCapabilities",
    "ProviderRegistry",
    "ComplexityAnalyzer",
    "ComplexityLevel",
    "ComplexityFeatures",
    "ProviderRouter",
    "RoutingDecision",
]
