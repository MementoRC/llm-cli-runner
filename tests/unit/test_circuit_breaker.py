"""Unit tests for circuit breaker pattern."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from mcp_server_cheap_llm.core.errors import ProviderError
from mcp_server_cheap_llm.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitState,
    ProviderCircuitBreaker,
)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout == 60.0
        assert config.reset_timeout == 300.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            timeout=120.0,
            reset_timeout=600.0,
        )

        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.timeout == 120.0
        assert config.reset_timeout == 600.0


class TestCircuitBreakerState:
    """Test circuit breaker state."""

    def test_default_state(self):
        """Test default state values."""
        state = CircuitBreakerState()

        assert state.state == CircuitState.CLOSED
        assert state.failure_count == 0
        assert state.success_count == 0
        assert state.last_failure_time == 0.0
        assert state.next_attempt_time == 0.0

    def test_custom_state(self):
        """Test custom state values."""
        state = CircuitBreakerState(
            state=CircuitState.OPEN,
            failure_count=5,
            success_count=2,
            last_failure_time=1234567890.0,
            next_attempt_time=1234567950.0,
        )

        assert state.state == CircuitState.OPEN
        assert state.failure_count == 5
        assert state.success_count == 2
        assert state.last_failure_time == 1234567890.0
        assert state.next_attempt_time == 1234567950.0


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker()

        assert isinstance(breaker.config, CircuitBreakerConfig)
        assert isinstance(breaker.state, CircuitBreakerState)
        assert breaker.state.state == CircuitState.CLOSED

    def test_circuit_breaker_custom_config(self):
        """Test circuit breaker with custom config."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        assert breaker.config.failure_threshold == 3

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test successful function call."""
        breaker = CircuitBreaker()

        async def success_func():
            return "success"

        result = await breaker.call(success_func)

        assert result == "success"
        assert breaker.state.state == CircuitState.CLOSED
        assert breaker.state.failure_count == 0

    @pytest.mark.asyncio
    async def test_failed_call(self):
        """Test failed function call."""
        breaker = CircuitBreaker()

        async def fail_func():
            raise Exception("Test failure")

        with pytest.raises(ProviderError, match="Function call failed"):
            await breaker.call(fail_func)

        assert breaker.state.failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout=1.0)
        breaker = CircuitBreaker(config)

        async def fail_func():
            raise Exception("Test failure")

        # Trigger failures to open circuit
        for _ in range(3):
            with pytest.raises(ProviderError):
                await breaker.call(fail_func)

        assert breaker.state.state == CircuitState.OPEN
        assert breaker.state.failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_blocks_calls_when_open(self):
        """Test circuit blocks calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=10.0)
        breaker = CircuitBreaker(config)

        async def fail_func():
            raise Exception("Test failure")

        # Open the circuit
        with pytest.raises(ProviderError):
            await breaker.call(fail_func)

        # Subsequent calls should be blocked
        with pytest.raises(ProviderError, match="Circuit breaker is OPEN"):
            await breaker.call(fail_func)

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        breaker = CircuitBreaker(config)

        async def fail_func():
            raise Exception("Test failure")

        async def success_func():
            return "success"

        # Open the circuit
        with pytest.raises(ProviderError):
            await breaker.call(fail_func)

        assert breaker.state.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Next call should transition to half-open
        result = await breaker.call(success_func)

        assert result == "success"
        assert breaker.state.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_after_successes(self):
        """Test circuit closes after success threshold in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1, success_threshold=2, timeout=0.1
        )
        breaker = CircuitBreaker(config)

        async def fail_func():
            raise Exception("Test failure")

        async def success_func():
            return "success"

        # Open the circuit
        with pytest.raises(ProviderError):
            await breaker.call(fail_func)

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Make successful calls to close circuit
        await breaker.call(success_func)  # Half-open
        await breaker.call(success_func)  # Should close circuit

        assert breaker.state.state == CircuitState.CLOSED
        assert breaker.state.success_count == 0
        assert breaker.state.failure_count == 0

    @pytest.mark.asyncio
    async def test_failure_count_resets_on_success(self):
        """Test failure count resets on successful call."""
        breaker = CircuitBreaker()

        async def fail_func():
            raise Exception("Test failure")

        async def success_func():
            return "success"

        # Make some failures
        for _ in range(2):
            with pytest.raises(ProviderError):
                await breaker.call(fail_func)

        assert breaker.state.failure_count == 2

        # Make successful call
        await breaker.call(success_func)

        assert breaker.state.failure_count == 0

    @pytest.mark.asyncio
    async def test_synchronous_function_call(self):
        """Test calling synchronous functions."""
        breaker = CircuitBreaker()

        def sync_func():
            return "sync_result"

        result = await breaker.call(sync_func)

        assert result == "sync_result"

    def test_get_state(self):
        """Test getting circuit breaker state."""
        breaker = CircuitBreaker()

        state = breaker.get_state()

        assert isinstance(state, CircuitBreakerState)
        assert state.state == CircuitState.CLOSED

    def test_reset(self):
        """Test resetting circuit breaker."""
        breaker = CircuitBreaker()

        # Modify state
        breaker.state.failure_count = 5
        breaker.state.state = CircuitState.OPEN

        # Reset
        breaker.reset()

        assert breaker.state.state == CircuitState.CLOSED
        assert breaker.state.failure_count == 0

    def test_force_open(self):
        """Test forcing circuit breaker open."""
        breaker = CircuitBreaker()

        breaker.force_open()

        assert breaker.state.state == CircuitState.OPEN
        assert breaker.state.next_attempt_time > time.time()

    def test_force_close(self):
        """Test forcing circuit breaker closed."""
        breaker = CircuitBreaker()

        # Set to open state
        breaker.state.state = CircuitState.OPEN
        breaker.state.failure_count = 5

        # Force close
        breaker.force_close()

        assert breaker.state.state == CircuitState.CLOSED
        assert breaker.state.failure_count == 0


class TestProviderCircuitBreaker:
    """Test provider-specific circuit breaker."""

    def test_provider_circuit_breaker_initialization(self):
        """Test provider circuit breaker initialization."""
        breaker = ProviderCircuitBreaker("test_provider")

        assert breaker.provider_name == "test_provider"
        assert isinstance(breaker.circuit_breaker, CircuitBreaker)

    def test_provider_circuit_breaker_custom_config(self):
        """Test provider circuit breaker with custom config."""
        config = CircuitBreakerConfig(failure_threshold=10)
        breaker = ProviderCircuitBreaker("test_provider", config)

        assert breaker.circuit_breaker.config.failure_threshold == 10

    @pytest.mark.asyncio
    async def test_provider_generate_success(self):
        """Test successful provider generate call."""
        breaker = ProviderCircuitBreaker("test_provider")

        async def mock_generate():
            return "response"

        result = await breaker.generate(mock_generate)

        assert result == "response"

    @pytest.mark.asyncio
    async def test_provider_generate_failure(self):
        """Test failed provider generate call."""
        breaker = ProviderCircuitBreaker("test_provider")

        async def mock_generate():
            raise Exception("Generate failed")

        with pytest.raises(
            ProviderError, match="Provider 'test_provider' generation failed"
        ):
            await breaker.generate(mock_generate)

    def test_get_health_status(self):
        """Test getting provider health status."""
        breaker = ProviderCircuitBreaker("test_provider")

        status = breaker.get_health_status()

        assert status["provider"] == "test_provider"
        assert status["circuit_state"] == "closed"
        assert status["failure_count"] == 0
        assert status["success_count"] == 0
        assert status["healthy"] is True
        assert status["last_failure"] is None
        assert status["next_attempt"] is None

    def test_get_health_status_with_failures(self):
        """Test health status with failures."""
        breaker = ProviderCircuitBreaker("test_provider")

        # Simulate failures
        breaker.circuit_breaker.state.failure_count = 3
        breaker.circuit_breaker.state.last_failure_time = time.time()

        status = breaker.get_health_status()

        assert status["failure_count"] == 3
        assert status["last_failure"] is not None

    def test_get_health_status_circuit_open(self):
        """Test health status when circuit is open."""
        breaker = ProviderCircuitBreaker("test_provider")

        # Simulate open circuit
        breaker.circuit_breaker.state.state = CircuitState.OPEN
        breaker.circuit_breaker.state.next_attempt_time = time.time() + 60

        status = breaker.get_health_status()

        assert status["circuit_state"] == "open"
        assert status["healthy"] is False
        assert status["next_attempt"] is not None
