"""Circuit breaker implementation for provider reliability.

This module provides circuit breaker functionality specifically designed for
LLM providers to handle failures gracefully and prevent cascading failures.

Key classes:
    ProviderCircuitBreaker: Circuit breaker for provider operations

Example:
    >>> breaker = ProviderCircuitBreaker(provider_name="gemini")
    >>> result = await breaker.call(some_async_function)

"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mcp_server_cheap_llm.core.errors import ProviderError
from mcp_server_cheap_llm.utils.logging import StructuredLogger


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening
    timeout_duration: float = 60.0  # Seconds to wait in OPEN state
    success_threshold: int = 3  # Successes needed to close from HALF_OPEN


class ProviderCircuitBreaker:
    """Circuit breaker for LLM provider operations.

    Implements the circuit breaker pattern to prevent cascading failures
    when a provider becomes unreliable or unavailable.

    Attributes:
        provider_name: Name of the provider this protects
        config: Circuit breaker configuration
        state: Current circuit breaker state
        failure_count: Current consecutive failure count
        success_count: Current consecutive success count (in HALF_OPEN)
        last_failure_time: Timestamp of last failure
        logger: Structured logger instance

    Example:
        >>> breaker = ProviderCircuitBreaker("gemini")
        >>>
        >>> async def risky_operation():
        ...     # Some operation that might fail
        ...     return await api_call()
        >>>
        >>> try:
        ...     result = await breaker.call(risky_operation)
        ... except ProviderError as e:
        ...     # Handle circuit breaker or operation failure
        ...     pass

    """

    def __init__(
        self, provider_name: str, config: CircuitBreakerConfig | None = None
    ) -> None:
        """Initialize circuit breaker for a provider.

        Args:
            provider_name: Name of the provider to protect
            config: Optional circuit breaker configuration

        """
        self.provider_name = provider_name
        self.config = config or CircuitBreakerConfig()

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None

        self.logger = StructuredLogger(__name__)
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[[], Awaitable[Any]]) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute

        Returns:
            Result of function execution

        Raises:
            ProviderError: If circuit is open or function fails

        """
        async with self._lock:
            # Check if circuit should transition states
            await self._check_state_transition()

            if self.state == CircuitBreakerState.OPEN:
                msg = f"Circuit breaker is OPEN for provider {self.provider_name}"
                raise ProviderError(
                    msg,
                    provider=self.provider_name,
                    error_code="CIRCUIT_BREAKER_OPEN",
                    context={
                        "provider": self.provider_name,
                        "state": self.state,
                        "failure_count": self.failure_count,
                        "last_failure_time": self.last_failure_time,
                    },
                )

        # Execute the function
        try:
            result = await func()
            await self._on_success()
            return result

        except Exception as e:
            await self._on_failure(e)
            raise

    async def _check_state_transition(self) -> None:
        """Check if circuit breaker should transition states."""
        if self.state == CircuitBreakerState.OPEN and (
            self.last_failure_time
            and time.time() - self.last_failure_time >= self.config.timeout_duration
        ):
            self.logger.info(
                "Circuit breaker transitioning to HALF_OPEN",
                extra={
                    "provider": self.provider_name,
                    "previous_state": self.state,
                    "timeout_duration": self.config.timeout_duration,
                },
            )
            self.state = CircuitBreakerState.HALF_OPEN
            self.success_count = 0

    async def _on_success(self) -> None:
        """Handle successful function execution."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1

                if self.success_count >= self.config.success_threshold:
                    self.logger.info(
                        "Circuit breaker transitioning to CLOSED after successful recovery",
                        extra={
                            "provider": self.provider_name,
                            "success_count": self.success_count,
                            "success_threshold": self.config.success_threshold,
                        },
                    )
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0

            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed function execution.

        Args:
            exception: Exception that occurred

        """
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            self.logger.warning(
                "Circuit breaker recorded failure",
                extra={
                    "provider": self.provider_name,
                    "failure_count": self.failure_count,
                    "failure_threshold": self.config.failure_threshold,
                    "error": str(exception),
                    "current_state": self.state,
                },
            )

            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.logger.error(
                        "Circuit breaker opening due to failure threshold",
                        extra={
                            "provider": self.provider_name,
                            "failure_count": self.failure_count,
                            "failure_threshold": self.config.failure_threshold,
                        },
                    )
                    self.state = CircuitBreakerState.OPEN

            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.logger.warning(
                    "Circuit breaker reopening after failed recovery attempt",
                    extra={"provider": self.provider_name},
                )
                self.state = CircuitBreakerState.OPEN

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state.

        Returns:
            Current state of circuit breaker

        """
        return self.state

    def get_health_status(self) -> dict[str, Any]:
        """Get health status information.

        Returns:
            Dict containing circuit breaker health information

        """
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.config.failure_threshold,
            "success_threshold": self.config.success_threshold,
            "timeout_duration": self.config.timeout_duration,
            "last_failure_time": self.last_failure_time,
            "is_healthy": self.state != CircuitBreakerState.OPEN,
        }

    async def reset(self) -> None:
        """Reset circuit breaker to CLOSED state.

        This method allows manual reset of the circuit breaker,
        useful for administrative operations.
        """
        async with self._lock:
            old_state = self.state
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None

            self.logger.info(
                "Circuit breaker manually reset",
                extra={
                    "provider": self.provider_name,
                    "previous_state": old_state,
                    "new_state": self.state,
                },
            )

    async def force_open(self) -> None:
        """Force circuit breaker to OPEN state.

        Useful for maintenance or when provider is known to be down.
        """
        async with self._lock:
            old_state = self.state
            self.state = CircuitBreakerState.OPEN
            self.last_failure_time = time.time()

            self.logger.warning(
                "Circuit breaker forced to OPEN state",
                extra={"provider": self.provider_name, "previous_state": old_state},
            )
