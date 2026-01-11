"""Core data models and validation for MCP Server LLM CLI Runner.

This package contains the fundamental data structures and validation logic
following atomic design principles.

Modules:
    models: Pydantic models for requests, responses, and configuration
    request_processor: Request processing and routing engine
    validators: Input validation and sanitization functions

Example:
    >>> from mcp_server_llm_cli_runner.core.models import LLMRequest, LLMResponse
    >>> request = LLMRequest(prompt="Hello", provider="gemini")

"""

from mcp_server_llm_cli_runner.core.models import (
    CostEstimate,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderStatus,
    ProviderStatusInfo,
    ProviderType,
)
from mcp_server_llm_cli_runner.core.request_processor import (
    ContextManager,
    RequestProcessor,
)

__all__ = [
    "ContextManager",
    "CostEstimate",
    "LLMRequest",
    "LLMResponse",
    "ProviderConfig",
    "ProviderStatus",
    "ProviderStatusInfo",
    "ProviderType",
    "RequestProcessor",
]
