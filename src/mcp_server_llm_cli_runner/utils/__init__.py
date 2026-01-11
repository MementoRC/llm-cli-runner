"""Utility modules for MCP Server LLM CLI Runner.

This package provides core utilities for configuration, logging, error handling,
performance monitoring, connection pooling, and resource management.
All utilities follow atomic design principles with clear responsibilities.

Modules:
    config: Configuration management and validation
    logging: Structured logging setup and utilities
    errors: Custom exception classes
    performance: Performance metrics and timing utilities
    connection_pool: Async connection pooling
    resource_monitor: System resource monitoring

Example:
    >>> from mcp_server_llm_cli_runner.utils import get_logger, ConfigManager, SecurityConfig
    >>> from mcp_server_llm_cli_runner.utils import PerformanceMetrics, ResourceMonitor
    >>> logger = get_logger(__name__)
    >>> config = ConfigManager()
    >>> metrics = PerformanceMetrics()
"""

from mcp_server_llm_cli_runner.utils.config import (
    APIKeyManager,
    CacheConfig,
    ConfigManager,
    SecurityConfig,
)
from mcp_server_llm_cli_runner.utils.connection_pool import (
    AsyncConnectionPool,
    ConnectionPoolConfig,
    ConnectionPoolManager,
    PooledConnection,
    PoolStatistics,
)
from mcp_server_llm_cli_runner.utils.errors import (
    ConfigurationError,
    ErrorSerializer,
    LLMCliRunnerError,
    ProviderError,
    RateLimitError,
    SecurityError,
    ValidationError,
)
from mcp_server_llm_cli_runner.utils.logging import get_logger, setup_logging
from mcp_server_llm_cli_runner.utils.performance import (
    LatencyStats,
    LatencyTracker,
    MetricsAggregator,
    MetricsReporter,
    PerformanceMetrics,
    PerformanceSnapshot,
    ThroughputStats,
    sync_timing_decorator,
    timing_decorator,
)
from mcp_server_llm_cli_runner.utils.resource_monitor import (
    GCOptimizer,
    ResourceAlert,
    ResourceLevel,
    ResourceMonitor,
    ResourceSnapshot,
    ResourceThresholds,
)

__all__ = [
    # Config
    "ConfigManager",
    "SecurityConfig",
    "APIKeyManager",
    "CacheConfig",
    # Errors
    "LLMCliRunnerError",
    "ConfigManager",
    "ConfigurationError",
    "ErrorSerializer",
    "ProviderError",
    "RateLimitError",
    "SecurityError",
    "ValidationError",
    # Logging
    "get_logger",
    "setup_logging",
    # Performance
    "PerformanceMetrics",
    "PerformanceSnapshot",
    "LatencyStats",
    "LatencyTracker",
    "ThroughputStats",
    "MetricsAggregator",
    "MetricsReporter",
    "timing_decorator",
    "sync_timing_decorator",
    # Connection Pool
    "AsyncConnectionPool",
    "ConnectionPoolConfig",
    "ConnectionPoolManager",
    "PooledConnection",
    "PoolStatistics",
    # Resource Monitor
    "ResourceMonitor",
    "ResourceSnapshot",
    "ResourceThresholds",
    "ResourceAlert",
    "ResourceLevel",
    "GCOptimizer",
]
