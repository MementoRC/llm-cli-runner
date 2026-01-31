"""Provider Manager for centralized provider coordination and management.

This module implements the core ProviderManager class that coordinates all LLM providers
with comprehensive health monitoring, usage tracking, and intelligent failover capabilities.
Follows atomic design principles and integrates with existing cache system and provider registry.

Key classes:
    ProviderManager: Central coordinator for all provider operations
    ProviderHealthMonitor: Health monitoring and status tracking
    UsageTracker: Real-time usage aggregation and tracking
    QuotaManager: Quota monitoring and threshold management

Example:
    >>> manager = ProviderManager()
    >>> await manager.initialize()
    >>>
    >>> # Get a provider with health checking
    >>> provider = await manager.get_healthy_provider("gemini")
    >>>
    >>> # Route request with smart selection
    >>> response = await manager.route_request(request)

"""

import asyncio
import contextlib
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

from src.mcp_server_llm_cli_runner.cache.service import CacheService, MemoryCache
from src.mcp_server_llm_cli_runner.core.errors import ProviderError
from src.mcp_server_llm_cli_runner.core.models import (
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderStatus,
    ProviderType,
    QuotaStatus,
)
from src.mcp_server_llm_cli_runner.utils.config import ConfigManager
from src.mcp_server_llm_cli_runner.utils.logging import StructuredLogger

from .base import LLMProvider
from .registry import ProviderRegistry
from .routing import ProviderRouter


class ProviderHealthStatus:
    """Atomic health status component for individual providers."""

    def __init__(self, provider_name: str) -> None:
        """Initialize health status tracking.

        Args:
            provider_name: Name of the provider to track

        """
        self.provider_name = provider_name
        self.status = ProviderStatus.UNKNOWN
        self.last_check = None
        self.consecutive_failures = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.average_response_time = 0.0
        self.last_error: str | None = None
        self.uptime_start = datetime.now(UTC)

    def update_status(
        self,
        status: ProviderStatus,
        response_time: float | None = None,
        error: str | None = None,
    ) -> None:
        """Update provider health status.

        Args:
            status: New provider status
            response_time: Response time in milliseconds
            error: Error message if status indicates failure

        """
        self.last_check = datetime.now(UTC)
        previous_status = self.status
        self.status = status
        self.total_requests += 1

        if status == ProviderStatus.HEALTHY:
            self.successful_requests += 1
            self.consecutive_failures = 0
            if response_time:
                # Simple moving average update
                self.average_response_time = (
                    self.average_response_time * (self.successful_requests - 1)
                    + response_time
                ) / self.successful_requests
        else:
            self.consecutive_failures += 1
            if error:
                self.last_error = error

        # Reset uptime if recovering from unhealthy state
        if (
            previous_status != ProviderStatus.HEALTHY
            and status == ProviderStatus.HEALTHY
        ):
            self.uptime_start = datetime.now(UTC)

    def get_success_rate(self) -> float:
        """Calculate success rate percentage.

        Returns:
            Success rate as percentage (0.0-100.0)

        """
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds since last recovery.

        Returns:
            Uptime in seconds

        """
        return (datetime.now(UTC) - self.uptime_start).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert health status to dictionary.

        Returns:
            Dictionary representation of health status

        """
        return {
            "provider_name": self.provider_name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "consecutive_failures": self.consecutive_failures,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.get_success_rate(),
            "average_response_time": self.average_response_time,
            "uptime_seconds": self.get_uptime_seconds(),
            "last_error": self.last_error,
        }


class UsageMetrics:
    """Atomic usage metrics component for tracking provider usage."""

    def __init__(self, provider_name: str, window_size: int = 100) -> None:
        """Initialize usage metrics.

        Args:
            provider_name: Name of the provider
            window_size: Size of the rolling window for metrics

        """
        self.provider_name = provider_name
        self.window_size = window_size

        # Rolling windows for metrics
        self.response_times: deque[float] = deque(maxlen=window_size)
        self.request_timestamps: deque[datetime] = deque(maxlen=window_size)
        self.token_counts: deque[int] = deque(maxlen=window_size)

        # Cumulative counters
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def record_request(
        self,
        response_time: float,
        token_count: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Record a request for metrics tracking.

        Args:
            response_time: Response time in seconds
            token_count: Number of tokens processed
            cost: Cost of the request in dollars

        """
        now = datetime.now(UTC)

        self.response_times.append(response_time)
        self.request_timestamps.append(now)
        self.token_counts.append(token_count)

        self.total_requests += 1
        self.total_tokens += token_count
        self.total_cost += cost

    def get_requests_per_minute(self) -> float:
        """Calculate requests per minute based on recent activity.

        Returns:
            Requests per minute

        """
        if len(self.request_timestamps) < 2:
            return 0.0

        now = datetime.now(UTC)
        one_minute_ago = now - timedelta(minutes=1)

        recent_requests = sum(
            1 for timestamp in self.request_timestamps if timestamp > one_minute_ago
        )

        return float(recent_requests)

    def get_average_response_time(self) -> float:
        """Get average response time from recent requests.

        Returns:
            Average response time in seconds

        """
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def get_tokens_per_second(self) -> float:
        """Calculate tokens per second throughput.

        Returns:
            Tokens per second

        """
        if len(self.request_timestamps) < 2 or not self.token_counts:
            return 0.0

        time_span = (
            self.request_timestamps[-1] - self.request_timestamps[0]
        ).total_seconds()
        if time_span <= 0:
            return 0.0

        return sum(self.token_counts) / time_span

    def to_dict(self) -> dict[str, Any]:
        """Convert usage metrics to dictionary.

        Returns:
            Dictionary representation of usage metrics

        """
        return {
            "provider_name": self.provider_name,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "requests_per_minute": self.get_requests_per_minute(),
            "average_response_time": self.get_average_response_time(),
            "tokens_per_second": self.get_tokens_per_second(),
            "window_size": len(self.response_times),
        }


class QuotaTracker:
    """Atomic quota tracking component for provider limits."""

    def __init__(self, provider_name: str) -> None:
        """Initialize quota tracker.

        Args:
            provider_name: Name of the provider

        """
        self.provider_name = provider_name
        self.daily_requests = 0
        self.daily_tokens = 0
        self.daily_cost = 0.0
        self.last_reset = datetime.now(UTC).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        # Limits (would be loaded from configuration)
        self.max_daily_requests: int | None = None
        self.max_daily_tokens: int | None = None
        self.max_daily_cost: float | None = None

    def record_usage(self, tokens: int = 0, cost: float = 0.0) -> None:
        """Record usage against quotas.

        Args:
            tokens: Number of tokens used
            cost: Cost incurred

        """
        # Check if we need to reset daily counters
        self._reset_if_new_day()

        self.daily_requests += 1
        self.daily_tokens += tokens
        self.daily_cost += cost

    def _reset_if_new_day(self) -> None:
        """Reset daily counters if it's a new day."""
        now = datetime.now(UTC)
        current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if current_day > self.last_reset:
            self.daily_requests = 0
            self.daily_tokens = 0
            self.daily_cost = 0.0
            self.last_reset = current_day

    def check_quota_status(self) -> QuotaStatus:
        """Check current quota status.

        Returns:
            Current quota status

        """
        # Check request limits
        if self.max_daily_requests and self.daily_requests >= self.max_daily_requests:
            return QuotaStatus.EXCEEDED

        # Check token limits
        if self.max_daily_tokens and self.daily_tokens >= self.max_daily_tokens:
            return QuotaStatus.EXCEEDED

        # Check cost limits
        if self.max_daily_cost and self.daily_cost >= self.max_daily_cost:
            return QuotaStatus.EXCEEDED

        # Check if approaching limits (80% threshold)
        approaching_request_limit = (
            self.max_daily_requests
            and self.daily_requests >= self.max_daily_requests * 0.8
        )
        approaching_token_limit = (
            self.max_daily_tokens and self.daily_tokens >= self.max_daily_tokens * 0.8
        )
        approaching_cost_limit = (
            self.max_daily_cost and self.daily_cost >= self.max_daily_cost * 0.8
        )

        if (
            approaching_request_limit
            or approaching_token_limit
            or approaching_cost_limit
        ):
            return QuotaStatus.WARNING

        return QuotaStatus.HEALTHY

    def get_remaining_quota(self) -> dict[str, float]:
        """Get remaining quota percentages.

        Returns:
            Dictionary with remaining quota percentages

        """
        result = {}

        if self.max_daily_requests:
            remaining_requests = max(0, self.max_daily_requests - self.daily_requests)
            result["requests"] = (remaining_requests / self.max_daily_requests) * 100

        if self.max_daily_tokens:
            remaining_tokens = max(0, self.max_daily_tokens - self.daily_tokens)
            result["tokens"] = (remaining_tokens / self.max_daily_tokens) * 100

        if self.max_daily_cost:
            remaining_cost = max(0, self.max_daily_cost - self.daily_cost)
            result["cost"] = (remaining_cost / self.max_daily_cost) * 100

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert quota tracker to dictionary.

        Returns:
            Dictionary representation of quota tracker

        """
        return {
            "provider_name": self.provider_name,
            "daily_requests": self.daily_requests,
            "daily_tokens": self.daily_tokens,
            "daily_cost": self.daily_cost,
            "max_daily_requests": self.max_daily_requests,
            "max_daily_tokens": self.max_daily_tokens,
            "max_daily_cost": self.max_daily_cost,
            "quota_status": self.check_quota_status().value,
            "remaining_quota": self.get_remaining_quota(),
        }


class ProviderHealthMonitor:
    """Comprehensive health monitoring for all providers."""

    def __init__(self, registry: ProviderRegistry) -> None:
        """Initialize health monitor.

        Args:
            registry: Provider registry instance

        """
        self.registry = registry
        self.health_statuses: dict[str, ProviderHealthStatus] = {}
        self.logger = StructuredLogger(__name__)
        self.monitoring_task: asyncio.Task[None] | None = None
        self.check_interval = 30  # seconds

    async def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Started provider health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task
            self.monitoring_task = None
            self.logger.info("Stopped provider health monitoring")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await self._check_all_providers()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_all_providers(self) -> None:
        """Check health of all registered providers."""
        for provider_name in self.registry.list_providers():
            try:
                await self._check_provider_health(provider_name)
            except Exception as e:
                self.logger.exception(f"Health check failed for {provider_name}: {e}")

    async def _check_provider_health(self, provider_name: str) -> None:
        """Check health of a specific provider.

        Args:
            provider_name: Name of provider to check

        """
        start_time = time.time()

        try:
            provider = self.registry.get_provider(provider_name)
            if not provider:
                self._update_provider_status(
                    provider_name,
                    ProviderStatus.UNAVAILABLE,
                    error="Provider not found",
                )
                return

            # Perform health check (this would call provider-specific health check)
            health_ok = await self._perform_health_check(provider)
            response_time = (time.time() - start_time) * 1000  # Convert to ms

            if health_ok:
                self._update_provider_status(
                    provider_name,
                    ProviderStatus.HEALTHY,
                    response_time=response_time,
                )
            else:
                self._update_provider_status(
                    provider_name,
                    ProviderStatus.UNHEALTHY,
                    error="Health check failed",
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_provider_status(
                provider_name,
                ProviderStatus.ERROR,
                response_time=response_time,
                error=str(e),
            )

    async def _perform_health_check(self, provider: LLMProvider) -> bool:
        """Perform actual health check on provider.

        Args:
            provider: Provider instance

        Returns:
            True if healthy, False otherwise

        """
        # This would implement provider-specific health checks
        # For now, return True as placeholder
        return True

    def _update_provider_status(
        self,
        provider_name: str,
        status: ProviderStatus,
        response_time: float | None = None,
        error: str | None = None,
    ) -> None:
        """Update provider health status.

        Args:
            provider_name: Name of provider
            status: New status
            response_time: Response time in milliseconds
            error: Error message if any

        """
        if provider_name not in self.health_statuses:
            self.health_statuses[provider_name] = ProviderHealthStatus(provider_name)

        self.health_statuses[provider_name].update_status(status, response_time, error)

    def get_provider_health(self, provider_name: str) -> ProviderHealthStatus | None:
        """Get health status for a provider.

        Args:
            provider_name: Name of provider

        Returns:
            Health status or None if not tracked

        """
        return self.health_statuses.get(provider_name)

    def get_healthy_providers(self) -> list[str]:
        """Get list of currently healthy provider names.

        Returns:
            List of healthy provider names

        """
        healthy = []
        for name, status in self.health_statuses.items():
            if status.status == ProviderStatus.HEALTHY:
                healthy.append(name)
        return healthy

    def get_all_health_statuses(self) -> dict[str, dict[str, Any]]:
        """Get health statuses for all providers.

        Returns:
            Dictionary mapping provider names to health status dictionaries

        """
        return {name: status.to_dict() for name, status in self.health_statuses.items()}


class ProviderManager:
    """Central manager for provider coordination and health monitoring.

    Provides unified interface for provider operations with intelligent
    routing, health monitoring, and usage tracking.
    """

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        """Initialize Provider Manager.

        Args:
            config_manager: Configuration manager instance

        """
        self.config_manager = config_manager
        self.logger = StructuredLogger(__name__)

        # Core components
        self.registry = ProviderRegistry()
        self.router: ProviderRouter | None = None
        self.health_monitor = ProviderHealthMonitor(self.registry)
        self.cache_service: CacheService | None = None

        # Tracking components
        self.usage_metrics: dict[str, UsageMetrics] = {}
        self.quota_trackers: dict[str, QuotaTracker] = {}

        # State
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize the Provider Manager and all its components."""
        if self.initialized:
            return

        try:
            self.logger.info("Initializing Provider Manager")

            # Initialize cache service integration with memory backend
            primary_backend = MemoryCache(max_size=1000)
            self.cache_service = CacheService(primary_backend=primary_backend)

            # Initialize router with registry
            self.router = ProviderRouter(self.registry)

            # Load provider configurations and register providers
            await self._load_and_register_providers()

            # Start health monitoring
            await self.health_monitor.start_monitoring()

            self.initialized = True
            self.logger.info("Provider Manager initialized successfully")

        except Exception as e:
            self.logger.exception(f"Failed to initialize Provider Manager: {e}")
            msg = f"Provider Manager initialization failed: {e}"
            raise ProviderError(
                msg,
                provider="manager",
                error_code="MANAGER_INIT_FAILED",
                context={"error": str(e)},
            ) from e

    async def _load_and_register_providers(self) -> None:
        """Load provider configurations and register providers."""
        # This would load from configuration
        # For now, we'll implement basic provider registration
        provider_configs = self._get_provider_configs()

        for config in provider_configs:
            try:
                # Register provider with registry
                # Note: Actual provider classes would be imported and registered
                # provider = self.registry.create_provider(config)

                # Initialize tracking components
                self.usage_metrics[config.name] = UsageMetrics(config.name)
                self.quota_trackers[config.name] = QuotaTracker(config.name)

                self.logger.info(f"Registered provider: {config.name}")

            except Exception as e:
                self.logger.exception(f"Failed to register provider {config.name}: {e}")

    def _get_provider_configs(self) -> list[ProviderConfig]:
        """Get provider configurations from config manager.

        Returns:
            List of provider configurations

        """
        # Placeholder implementation
        # In real implementation, this would load from config_manager
        return [
            ProviderConfig(
                name="gemini",
                provider_type=ProviderType.GEMINI,
                enabled=True,
                api_key="placeholder",
                base_url="https://api.gemini.com",
                models=["gemini-2.5-flash-lite", "gemini-2.5-flash"],
                max_tokens=8192,
                rate_limit={"requests_per_minute": 60},
            ),
            ProviderConfig(
                name="openai",
                provider_type=ProviderType.OPENAI,
                enabled=True,
                api_key="placeholder",
                base_url="https://api.openai.com/v1",
                models=["gpt-4", "gpt-3.5-turbo"],
                max_tokens=4096,
                rate_limit={"requests_per_minute": 60},
            ),
        ]

    async def get_healthy_provider(
        self,
        provider_type: str | None = None,
    ) -> LLMProvider | None:
        """Get a healthy provider instance.

        Args:
            provider_type: Optional provider type filter

        Returns:
            Healthy provider instance or None

        """
        if not self.initialized:
            await self.initialize()

        healthy_providers = self.health_monitor.get_healthy_providers()

        if provider_type:
            # Filter by provider type
            filtered_providers = [
                name
                for name in healthy_providers
                if name.lower() == provider_type.lower()
            ]
            if filtered_providers:
                return self.registry.get_provider(filtered_providers[0])
        # Return any healthy provider
        elif healthy_providers:
            return self.registry.get_provider(healthy_providers[0])

        return None

    async def route_request(self, request: LLMRequest) -> LLMResponse:
        """Route request through intelligent provider selection.

        Args:
            request: LLM request to route

        Returns:
            LLM response from selected provider

        Raises:
            ProviderError: If no providers available or all fail

        """
        if not self.initialized:
            await self.initialize()

        if not self.router:
            msg = "Router not initialized"
            raise ProviderError(
                msg,
                provider="manager",
                error_code="ROUTER_NOT_INITIALIZED",
            )

        try:
            # Record request start time for metrics
            start_time = time.time()

            # Route request through router to get decision
            routing_decision = self.router.route_request(request)

            # Get the selected provider and make the actual request
            provider_name = routing_decision.selected_provider.value
            provider = self.get_provider(provider_name)

            if not provider:
                msg = f"Selected provider {provider_name} not available"
                raise ProviderError(
                    msg,
                    provider=provider_name,
                    error_code="PROVIDER_UNAVAILABLE",
                )

            # Make the actual request to the provider
            response = await provider.generate(request)

            # Record metrics
            response_time = time.time() - start_time

            if provider_name in self.usage_metrics:
                token_count = getattr(response, "tokens_used", 0)
                cost = getattr(response, "cost", 0.0)

                self.usage_metrics[provider_name].record_request(
                    response_time,
                    token_count,
                    cost,
                )

                if provider_name in self.quota_trackers:
                    self.quota_trackers[provider_name].record_usage(token_count, cost)

            return response

        except Exception as e:
            self.logger.exception(f"Request routing failed: {e}")
            msg = f"Request routing failed: {e}"
            raise ProviderError(
                msg,
                provider="manager",
                error_code="ROUTING_FAILED",
                context={"error": str(e)},
            ) from e

    def get_provider_stats(self, provider_name: str | None = None) -> dict[str, Any]:
        """Get comprehensive provider statistics.

        Args:
            provider_name: Optional specific provider name

        Returns:
            Dictionary with provider statistics

        """
        if provider_name:
            if provider_name not in self.usage_metrics:
                return {}

            provider_health = self.health_monitor.get_provider_health(provider_name)
            return {
                "health": (
                    provider_health.to_dict() if provider_health is not None else None
                ),
                "usage": self.usage_metrics[provider_name].to_dict(),
                "quota": (
                    self.quota_trackers[provider_name].to_dict()
                    if provider_name in self.quota_trackers
                    else None
                ),
            }
        # Return stats for all providers
        all_stats = {}
        for name in self.usage_metrics:
            all_stats[name] = self.get_provider_stats(name)
        return all_stats

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health status.

        Returns:
            Dictionary with system health information

        """
        healthy_count = len(self.health_monitor.get_healthy_providers())
        total_count = len(self.health_monitor.health_statuses)

        total_requests = sum(
            metrics.total_requests for metrics in self.usage_metrics.values()
        )
        total_cost = sum(metrics.total_cost for metrics in self.usage_metrics.values())

        return {
            "providers": {
                "healthy": healthy_count,
                "total": total_count,
                "health_percentage": (healthy_count / max(total_count, 1)) * 100,
            },
            "usage": {
                "total_requests": total_requests,
                "total_cost": total_cost,
            },
            "cache": (self.cache_service.get_metrics() if self.cache_service else {}),
            "initialized": self.initialized,
        }

    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """Register a provider instance with the manager.

        Args:
            name: Name to register the provider under
            provider: Provider instance to register

        Raises:
            ProviderError: If registration fails

        """
        try:
            # Store the provider instance directly in the registry
            self.registry._instances[name] = provider

            # Initialize tracking components for the provider
            self.usage_metrics[name] = UsageMetrics(name)
            self.quota_trackers[name] = QuotaTracker(name)

            # Initialize health status
            if name not in self.health_monitor.health_statuses:
                self.health_monitor.health_statuses[name] = ProviderHealthStatus(name)

            self.logger.info(f"Provider registered: {name}")

        except Exception as e:
            self.logger.exception(f"Failed to register provider {name}: {e}")
            msg = f"Failed to register provider {name}: {e}"
            raise ProviderError(
                msg,
                provider=name,
                error_code="PROVIDER_REGISTRATION_FAILED",
                context={"provider_name": name, "error": str(e)},
            ) from e

    def get_provider(self, name: str) -> LLMProvider | None:
        """Get a registered provider by name.

        Args:
            name: Name of the provider to get

        Returns:
            Provider instance or None if not found

        """
        return self.registry.get_provider(name)

    def list_providers(self) -> list[LLMProvider]:
        """Get list of all registered provider instances.

        Returns:
            List of registered provider instances

        """
        return [
            self.registry._instances[name] for name in self.registry.list_providers()
        ]

    async def generate(
        self,
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate content using a specific provider.

        Args:
            prompt: Text prompt for generation
            provider: Name of provider to use
            model: Model name to use
            **kwargs: Additional generation parameters

        Returns:
            Generation response dictionary

        Raises:
            ProviderError: If provider not found or generation fails

        """
        if not provider:
            msg = "Provider name is required for generation"
            raise ProviderError(
                msg,
                provider="unknown",
                error_code="PROVIDER_NAME_REQUIRED",
            )

        provider_instance = self.get_provider(provider)
        if not provider_instance:
            msg = f"Provider '{provider}' not found"
            raise ProviderError(
                msg,
                provider=provider,
                error_code="PROVIDER_NOT_FOUND",
                context={"provider_name": provider},
            )

        try:
            # Record request start time for metrics
            start_time = time.time()

            # Create LLMRequest object from parameters
            llm_request = LLMRequest(
                prompt=prompt, provider=provider, model=model, **kwargs
            )

            # Call the provider's generate method
            result = await provider_instance.generate(llm_request)

            # Record metrics if successful
            response_time = time.time() - start_time
            if provider in self.usage_metrics:
                token_count = result.tokens_used
                cost = result.cost

                self.usage_metrics[provider].record_request(
                    response_time,
                    token_count,
                    cost,
                )

                if provider in self.quota_trackers:
                    self.quota_trackers[provider].record_usage(token_count, cost)

            # Convert LLMResponse to dict for backward compatibility
            return {
                "content": result.content,
                "provider": result.provider,
                "model": result.model,
                "success": result.success,
                "tokens_used": result.tokens_used,
                "cost": result.cost,
                "response_time_ms": result.response_time_ms,
                "error_message": result.error_message,
                "metadata": result.metadata,
            }

        except Exception as e:
            self.logger.exception(f"Generation failed for provider {provider}: {e}")
            msg = f"Generation failed: {e}"
            raise ProviderError(
                msg,
                provider=provider,
                error_code="GENERATION_FAILED",
                context={"provider_name": provider, "error": str(e)},
            ) from e

    async def shutdown(self) -> None:
        """Shutdown provider manager and cleanup resources."""
        try:
            # Stop health monitoring
            await self.health_monitor.stop_monitoring()

            # Close cache service
            if self.cache_service:
                await self.cache_service.close()

            self.initialized = False
            self.logger.info("Provider Manager shut down successfully")

        except Exception as e:
            self.logger.exception(f"Error during shutdown: {e}")
            raise
