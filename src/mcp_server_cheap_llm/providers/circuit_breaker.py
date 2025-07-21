"""Circuit breaker pattern for provider failure handling.

This module implements the circuit breaker pattern to handle provider failures
gracefully and provide automatic recovery mechanisms.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mcp_server_cheap_llm.core.errors import ProviderError


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes to close circuit from half-open
        timeout: Time to wait before trying half-open (seconds)
        reset_timeout: Time to reset failure count (seconds)
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    reset_timeout: float = 300.0


@dataclass
class CircuitBreakerState:
    """Current state of circuit breaker.

    Attributes:
        state: Current circuit state
        failure_count: Number of consecutive failures
        success_count: Number of consecutive successes in half-open
        last_failure_time: Timestamp of last failure
        next_attempt_time: When next attempt is allowed
    """

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    next_attempt_time: float = 0.0


class CircuitBreaker:
    """Circuit breaker for provider calls.

    Implements the circuit breaker pattern to prevent cascading failures
    and provide automatic recovery for provider operations.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Any: Function result

        Raises:
            ProviderError: If circuit is open or function fails
        """
        async with self._lock:
            current_time = time.time()

            # Check if circuit should transition states
            await self._check_state_transition(current_time)

            # Block calls if circuit is open
            if self.state.state == CircuitState.OPEN:
                if current_time < self.state.next_attempt_time:
                    raise ProviderError(
                        f"Circuit breaker is OPEN. Next attempt allowed at "
                        f"{time.ctime(self.state.next_attempt_time)}",
                        provider="circuit_breaker",
                    )
                else:
                    # Transition to half-open
                    self.state.state = CircuitState.HALF_OPEN
                    self.state.success_count = 0

            # Execute function
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                await self._record_success(current_time)
                return result

            except Exception as e:
                await self._record_failure(current_time)
                raise ProviderError(
                    f"Function call failed: {e}", provider="circuit_breaker"
                ) from e

    async def _check_state_transition(self, current_time: float) -> None:
        """Check if circuit breaker should transition states.

        Args:
            current_time: Current timestamp
        """
        # Reset failure count if enough time has passed
        if (
            current_time - self.state.last_failure_time > self.config.reset_timeout
            and self.state.failure_count > 0
        ):
            self.state.failure_count = 0

    async def _record_success(self, current_time: float) -> None:
        """Record successful operation.

        Args:
            current_time: Current timestamp
        """
        if self.state.state == CircuitState.HALF_OPEN:
            self.state.success_count += 1

            # Close circuit if enough successes
            if self.state.success_count >= self.config.success_threshold:
                self.state.state = CircuitState.CLOSED
                self.state.failure_count = 0
                self.state.success_count = 0

        elif self.state.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.state.failure_count = 0

    async def _record_failure(self, current_time: float) -> None:
        """Record failed operation.

        Args:
            current_time: Current timestamp
        """
        self.state.failure_count += 1
        self.state.last_failure_time = current_time

        # Open circuit if failure threshold exceeded
        if self.state.failure_count >= self.config.failure_threshold:
            self.state.state = CircuitState.OPEN
            self.state.next_attempt_time = current_time + self.config.timeout
            self.state.success_count = 0

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state.

        Returns:
            CircuitBreakerState: Current state information
        """
        return self.state

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitBreakerState()

    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        self.state.state = CircuitState.OPEN
        self.state.next_attempt_time = time.time() + self.config.timeout

    def force_close(self) -> None:
        """Force circuit breaker to closed state."""
        self.state.state = CircuitState.CLOSED
        self.state.failure_count = 0
        self.state.success_count = 0


class ProviderCircuitBreaker:
    """Circuit breaker specifically for LLM providers.

    Provides provider-specific circuit breaker functionality with
    additional features for LLM provider patterns.
    """

    def __init__(self, provider_name: str, config: CircuitBreakerConfig | None = None):
        """Initialize provider circuit breaker.

        Args:
            provider_name: Name of the provider
            config: Circuit breaker configuration
        """
        self.provider_name = provider_name
        self.circuit_breaker = CircuitBreaker(config)

    async def generate(self, provider_func: Callable, *args, **kwargs) -> Any:
        """Execute provider generate function with circuit breaker.

        Args:
            provider_func: Provider's generate method
            *args: Arguments for generate method
            **kwargs: Keyword arguments for generate method

        Returns:
            Any: Generate method result

        Raises:
            ProviderError: If circuit is open or generation fails
        """
        try:
            return await self.circuit_breaker.call(provider_func, *args, **kwargs)
        except ProviderError as e:
            # Add provider context to error
            raise ProviderError(
                f"Provider '{self.provider_name}' generation failed: {e}",
                provider=self.provider_name,
            ) from e

    def get_health_status(self) -> dict:
        """Get provider health status.

        Returns:
            dict: Health status information
        """
        state = self.circuit_breaker.get_state()

        return {
            "provider": self.provider_name,
            "circuit_state": state.state.value,
            "failure_count": state.failure_count,
            "success_count": state.success_count,
            "healthy": state.state == CircuitState.CLOSED,
            "last_failure": time.ctime(state.last_failure_time)
            if state.last_failure_time
            else None,
            "next_attempt": time.ctime(state.next_attempt_time)
            if state.next_attempt_time
            else None,
        }
