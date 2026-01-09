"""Async connection pooling for MCP Server LLM CLI Runner.

This module provides connection pooling for provider API connections with
lifecycle management, health checking, and pool size configuration.
Follows atomic design patterns with clear data structures.

Key classes:
    AsyncConnectionPool: Main connection pooling implementation
    PooledConnection: Wrapper for pooled connection objects
    ConnectionPoolConfig: Configuration for pool behavior

Example:
    >>> config = ConnectionPoolConfig(max_size=10, min_size=2)
    >>> pool = AsyncConnectionPool(config, connection_factory)
    >>> async with pool.acquire() as conn:
    ...     await conn.request(...)

"""

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, Field

from mcp_server_llm_cli_runner.utils.logging import get_logger

logger = get_logger(__name__)


class ConnectionState(str, Enum):
    """State of a pooled connection."""

    IDLE = "idle"
    IN_USE = "in_use"
    CLOSED = "closed"
    UNHEALTHY = "unhealthy"


class ConnectionPoolConfig(BaseModel):
    """Configuration for connection pool behavior.

    Attributes:
        max_size: Maximum number of connections in the pool
        min_size: Minimum number of idle connections to maintain
        max_idle_time_seconds: Maximum time a connection can be idle
        connection_timeout_seconds: Timeout for acquiring a connection
        health_check_interval_seconds: Interval between health checks
        max_lifetime_seconds: Maximum lifetime of a connection
        enable_health_checks: Whether to perform periodic health checks
        retry_attempts: Number of retry attempts for failed connections

    """

    max_size: int = Field(default=10, ge=1, le=100)
    min_size: int = Field(default=2, ge=0, le=50)
    max_idle_time_seconds: int = Field(default=300, ge=10, le=3600)
    connection_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    health_check_interval_seconds: int = Field(default=60, ge=10, le=600)
    max_lifetime_seconds: int = Field(default=3600, ge=60, le=86400)
    enable_health_checks: bool = Field(default=True)
    retry_attempts: int = Field(default=3, ge=1, le=10)


@dataclass
class PooledConnection[T]:
    """Wrapper for a pooled connection with metadata.

    Attributes:
        connection: The actual connection object
        created_at: Timestamp when connection was created
        last_used_at: Timestamp when connection was last used
        use_count: Number of times connection has been used
        state: Current state of the connection
        pool_id: Unique identifier within the pool

    """

    connection: T
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0
    state: ConnectionState = ConnectionState.IDLE
    pool_id: int = 0

    def __hash__(self) -> int:
        """Make PooledConnection hashable using pool_id."""
        return hash(self.pool_id)

    def __eq__(self, other: object) -> bool:
        """Compare PooledConnections by pool_id."""
        if not isinstance(other, PooledConnection):
            return NotImplemented
        return self.pool_id == other.pool_id

    def mark_used(self) -> None:
        """Mark the connection as used."""
        self.last_used_at = time.time()
        self.use_count += 1
        self.state = ConnectionState.IN_USE

    def mark_idle(self) -> None:
        """Mark the connection as idle."""
        self.state = ConnectionState.IDLE

    def mark_unhealthy(self) -> None:
        """Mark the connection as unhealthy."""
        self.state = ConnectionState.UNHEALTHY

    def mark_closed(self) -> None:
        """Mark the connection as closed."""
        self.state = ConnectionState.CLOSED

    @property
    def age_seconds(self) -> float:
        """Get the age of the connection in seconds."""
        return time.time() - self.created_at

    @property
    def idle_time_seconds(self) -> float:
        """Get how long the connection has been idle."""
        return time.time() - self.last_used_at

    def is_expired(self, max_lifetime: int, max_idle: int) -> bool:
        """Check if the connection has expired.

        Args:
            max_lifetime: Maximum lifetime in seconds
            max_idle: Maximum idle time in seconds

        Returns:
            True if the connection should be closed

        """
        return self.age_seconds > max_lifetime or self.idle_time_seconds > max_idle


class PoolStatistics(BaseModel):
    """Statistics for the connection pool.

    Attributes:
        total_connections: Total connections created
        active_connections: Currently in-use connections
        idle_connections: Currently idle connections
        pending_requests: Requests waiting for connections
        total_acquisitions: Total connection acquisitions
        total_releases: Total connection releases
        total_timeouts: Total acquisition timeouts
        average_wait_time_ms: Average wait time for connections
        health_check_failures: Number of health check failures

    """

    total_connections: int = Field(default=0, ge=0)
    active_connections: int = Field(default=0, ge=0)
    idle_connections: int = Field(default=0, ge=0)
    pending_requests: int = Field(default=0, ge=0)
    total_acquisitions: int = Field(default=0, ge=0)
    total_releases: int = Field(default=0, ge=0)
    total_timeouts: int = Field(default=0, ge=0)
    average_wait_time_ms: float = Field(default=0.0, ge=0.0)
    health_check_failures: int = Field(default=0, ge=0)


class AsyncConnectionPool[T]:
    """Async connection pool with lifecycle management.

    Provides connection pooling for async I/O operations with features
    including automatic connection creation, health checking, and
    connection expiration.

    Attributes:
        config: Pool configuration
        name: Pool identifier for logging

    Example:
        >>> async def create_connection():
        ...     return await httpx.AsyncClient()
        ...
        >>> pool = AsyncConnectionPool(
        ...     ConnectionPoolConfig(max_size=5),
        ...     create_connection
        ... )
        >>> await pool.start()
        >>> async with pool.acquire() as conn:
        ...     response = await conn.get("https://api.example.com")
        >>> await pool.stop()

    """

    def __init__(
        self,
        config: ConnectionPoolConfig,
        connection_factory: Callable[[], T | Any],
        close_connection: Callable[[T], Any] | None = None,
        health_check: Callable[[T], bool | Any] | None = None,
        name: str = "default",
    ) -> None:
        """Initialize connection pool.

        Args:
            config: Pool configuration
            connection_factory: Async or sync callable that creates connections
            close_connection: Optional callable to close a connection
            health_check: Optional callable to check connection health
            name: Pool identifier for logging

        """
        self.config = config
        self.name = name
        self._connection_factory = connection_factory
        self._close_connection = close_connection
        self._health_check = health_check

        # Connection storage
        self._idle_connections: deque[PooledConnection[T]] = deque()
        self._active_connections: set[PooledConnection[T]] = set()
        self._all_connections: set[PooledConnection[T]] = set()

        # Synchronization
        self._lock = asyncio.Lock()
        self._available = asyncio.Semaphore(config.max_size)
        self._connection_available = asyncio.Condition()

        # State tracking
        self._next_pool_id = 0
        self._is_running = False
        self._health_check_task: asyncio.Task[None] | None = None
        self._maintenance_task: asyncio.Task[None] | None = None

        # Statistics
        self._stats = PoolStatistics()
        self._wait_times: deque[float] = deque(maxlen=100)

        logger.info(
            f"Connection pool '{name}' initialized",
            max_size=config.max_size,
            min_size=config.min_size,
        )

    async def start(self) -> None:
        """Start the connection pool and background tasks."""
        if self._is_running:
            return

        self._is_running = True

        # Pre-create minimum connections
        for _ in range(self.config.min_size):
            try:
                await self._create_connection()
            except Exception as e:
                logger.warning(f"Failed to pre-create connection: {e}")

        # Start background tasks
        if self.config.enable_health_checks:
            self._health_check_task = asyncio.create_task(self._health_check_loop())

        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

        logger.info(f"Connection pool '{self.name}' started")

    async def stop(self) -> None:
        """Stop the connection pool and close all connections."""
        if not self._is_running:
            return

        self._is_running = False

        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        if self._maintenance_task:
            self._maintenance_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._maintenance_task

        # Close all connections
        async with self._lock:
            for pooled_conn in list(self._all_connections):
                await self._close_pooled_connection(pooled_conn)

            self._idle_connections.clear()
            self._active_connections.clear()
            self._all_connections.clear()

        logger.info(f"Connection pool '{self.name}' stopped")

    async def _create_connection(self) -> PooledConnection[T]:
        """Create a new pooled connection.

        Returns:
            Newly created PooledConnection

        """
        self._next_pool_id += 1
        pool_id = self._next_pool_id

        # Call factory (may be async or sync)
        result = self._connection_factory()
        if asyncio.iscoroutine(result):
            connection = cast(T, await result)
        else:
            connection = cast(T, result)

        pooled_conn: PooledConnection[T] = PooledConnection(
            connection=connection,
            pool_id=pool_id,
        )

        async with self._lock:
            self._all_connections.add(pooled_conn)
            self._idle_connections.append(pooled_conn)
            self._stats.total_connections += 1

        logger.debug(f"Created connection {pool_id} in pool '{self.name}'")
        return pooled_conn

    async def _close_pooled_connection(self, pooled_conn: PooledConnection[T]) -> None:
        """Close a pooled connection.

        Args:
            pooled_conn: Connection to close

        """
        pooled_conn.mark_closed()

        if self._close_connection:
            try:
                result = self._close_connection(pooled_conn.connection)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Error closing connection {pooled_conn.pool_id}: {e}")

        logger.debug(f"Closed connection {pooled_conn.pool_id} in pool '{self.name}'")

    async def acquire(self, timeout: float | None = None) -> T:
        """Acquire a connection from the pool.

        Args:
            timeout: Maximum time to wait for a connection (uses config default if None)

        Returns:
            A connection object

        Raises:
            TimeoutError: If no connection becomes available within timeout

        """
        if timeout is None:
            timeout = self.config.connection_timeout_seconds

        start_time = time.time()

        try:
            # Wait for a connection slot
            acquired = await asyncio.wait_for(
                self._available.acquire(),
                timeout=timeout,
            )
            if not acquired:
                self._stats.total_timeouts += 1
                raise TimeoutError(
                    f"Timeout acquiring connection from pool '{self.name}'"
                )
        except TimeoutError:
            self._stats.total_timeouts += 1
            raise TimeoutError(
                f"Timeout acquiring connection from pool '{self.name}'"
            ) from None

        async with self._connection_available:
            # Try to get an idle connection
            async with self._lock:
                while self._idle_connections:
                    pooled_conn = self._idle_connections.popleft()

                    # Check if connection is still valid
                    if pooled_conn.state == ConnectionState.CLOSED:
                        continue

                    if pooled_conn.is_expired(
                        self.config.max_lifetime_seconds,
                        self.config.max_idle_time_seconds,
                    ):
                        await self._close_pooled_connection(pooled_conn)
                        self._all_connections.discard(pooled_conn)
                        continue

                    # Use this connection
                    pooled_conn.mark_used()
                    self._active_connections.add(pooled_conn)

                    wait_time = (time.time() - start_time) * 1000
                    self._wait_times.append(wait_time)
                    self._stats.total_acquisitions += 1
                    self._update_stats()

                    return pooled_conn.connection

            # No idle connections, create new one if under limit
            if len(self._all_connections) < self.config.max_size:
                try:
                    pooled_conn = await self._create_connection()
                    async with self._lock:
                        self._idle_connections.remove(pooled_conn)
                        pooled_conn.mark_used()
                        self._active_connections.add(pooled_conn)

                    wait_time = (time.time() - start_time) * 1000
                    self._wait_times.append(wait_time)
                    self._stats.total_acquisitions += 1
                    self._update_stats()

                    return pooled_conn.connection
                except Exception as e:
                    self._available.release()
                    raise RuntimeError(f"Failed to create connection: {e}") from e

        # Should not reach here, but release semaphore just in case
        self._available.release()
        raise RuntimeError(f"Failed to acquire connection from pool '{self.name}'")

    async def release(self, connection: T) -> None:
        """Release a connection back to the pool.

        Args:
            connection: The connection to release

        """
        async with self._lock:
            # Find the pooled connection
            pooled_conn = None
            for pc in self._active_connections:
                if pc.connection is connection:
                    pooled_conn = pc
                    break

            if pooled_conn is None:
                logger.warning(f"Released unknown connection to pool '{self.name}'")
                return

            self._active_connections.discard(pooled_conn)

            # Check if connection should be closed
            if pooled_conn.is_expired(
                self.config.max_lifetime_seconds,
                self.config.max_idle_time_seconds,
            ):
                await self._close_pooled_connection(pooled_conn)
                self._all_connections.discard(pooled_conn)
            elif pooled_conn.state != ConnectionState.UNHEALTHY:
                pooled_conn.mark_idle()
                self._idle_connections.append(pooled_conn)
            else:
                # Connection is unhealthy, close it
                await self._close_pooled_connection(pooled_conn)
                self._all_connections.discard(pooled_conn)

            self._stats.total_releases += 1
            self._update_stats()

        # Notify waiting acquirers
        async with self._connection_available:
            self._connection_available.notify()

        self._available.release()

    @contextlib.asynccontextmanager
    async def connection(self, timeout: float | None = None):
        """Context manager for connection acquisition and release.

        Args:
            timeout: Maximum time to wait for a connection

        Yields:
            A connection object

        Example:
            >>> async with pool.connection() as conn:
            ...     await conn.request(...)

        """
        conn = await self.acquire(timeout)
        try:
            yield conn
        finally:
            await self.release(conn)

    def _update_stats(self) -> None:
        """Update pool statistics."""
        self._stats.active_connections = len(self._active_connections)
        self._stats.idle_connections = len(self._idle_connections)

        if self._wait_times:
            self._stats.average_wait_time_ms = sum(self._wait_times) / len(
                self._wait_times
            )

    def get_statistics(self) -> PoolStatistics:
        """Get current pool statistics.

        Returns:
            PoolStatistics with current values

        """
        self._update_stats()
        return self._stats.model_copy()

    async def _health_check_loop(self) -> None:
        """Background task for connection health checking."""
        while self._is_running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                if not self._is_running:
                    break

                # Check idle connections
                async with self._lock:
                    connections_to_check = list(self._idle_connections)

                for pooled_conn in connections_to_check:
                    if not self._is_running:
                        break

                    if self._health_check:
                        try:
                            result = self._health_check(pooled_conn.connection)
                            if asyncio.iscoroutine(result):
                                is_healthy = await result
                            else:
                                is_healthy = result

                            if not is_healthy:
                                pooled_conn.mark_unhealthy()
                                self._stats.health_check_failures += 1
                                logger.warning(
                                    f"Connection {pooled_conn.pool_id} failed health check"
                                )
                        except Exception as e:
                            pooled_conn.mark_unhealthy()
                            self._stats.health_check_failures += 1
                            logger.warning(
                                f"Health check error for connection {pooled_conn.pool_id}: {e}"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Health check loop error: {e}")

    async def _maintenance_loop(self) -> None:
        """Background task for connection pool maintenance."""
        while self._is_running:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds

                if not self._is_running:
                    break

                # Remove expired and unhealthy connections
                async with self._lock:
                    connections_to_remove = []

                    for pooled_conn in list(self._idle_connections):
                        if pooled_conn.state == ConnectionState.UNHEALTHY:
                            connections_to_remove.append(pooled_conn)
                        elif pooled_conn.is_expired(
                            self.config.max_lifetime_seconds,
                            self.config.max_idle_time_seconds,
                        ):
                            connections_to_remove.append(pooled_conn)

                    for pooled_conn in connections_to_remove:
                        if pooled_conn in self._idle_connections:
                            self._idle_connections.remove(pooled_conn)
                        await self._close_pooled_connection(pooled_conn)
                        self._all_connections.discard(pooled_conn)

                    # Ensure minimum connections
                    while len(self._all_connections) < self.config.min_size:
                        try:
                            await self._create_connection()
                        except Exception as e:
                            logger.warning(
                                f"Failed to maintain minimum connections: {e}"
                            )
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Maintenance loop error: {e}")


class ConnectionPoolManager:
    """Manages multiple connection pools for different providers.

    Provides a central registry for connection pools with automatic
    lifecycle management.

    Example:
        >>> manager = ConnectionPoolManager()
        >>> manager.register_pool("gemini", pool_config, factory)
        >>> async with manager.get_connection("gemini") as conn:
        ...     await conn.request(...)

    """

    def __init__(self) -> None:
        """Initialize connection pool manager."""
        self._pools: dict[str, AsyncConnectionPool[Any]] = {}
        self._lock = asyncio.Lock()

    async def register_pool(
        self,
        name: str,
        config: ConnectionPoolConfig,
        connection_factory: Callable[[], Any],
        close_connection: Callable[[Any], Any] | None = None,
        health_check: Callable[[Any], bool | Any] | None = None,
    ) -> AsyncConnectionPool[Any]:
        """Register and start a new connection pool.

        Args:
            name: Pool name (usually provider name)
            config: Pool configuration
            connection_factory: Factory for creating connections
            close_connection: Optional close handler
            health_check: Optional health check function

        Returns:
            The created and started pool

        """
        async with self._lock:
            if name in self._pools:
                raise ValueError(f"Pool '{name}' already registered")

            pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
                config=config,
                connection_factory=connection_factory,
                close_connection=close_connection,
                health_check=health_check,
                name=name,
            )
            await pool.start()
            self._pools[name] = pool

            logger.info(f"Registered connection pool '{name}'")
            return pool

    async def get_pool(self, name: str) -> AsyncConnectionPool[Any]:
        """Get a registered pool by name.

        Args:
            name: Pool name

        Returns:
            The connection pool

        Raises:
            KeyError: If pool not found

        """
        async with self._lock:
            if name not in self._pools:
                raise KeyError(f"Pool '{name}' not found")
            return self._pools[name]

    @contextlib.asynccontextmanager
    async def get_connection(self, name: str, timeout: float | None = None):
        """Context manager to get a connection from a named pool.

        Args:
            name: Pool name
            timeout: Acquisition timeout

        Yields:
            A connection object

        """
        pool = await self.get_pool(name)
        async with pool.connection(timeout) as conn:
            yield conn

    async def close_all(self) -> None:
        """Stop all registered pools."""
        async with self._lock:
            for name, pool in self._pools.items():
                try:
                    await pool.stop()
                    logger.info(f"Closed pool '{name}'")
                except Exception as e:
                    logger.exception(f"Error closing pool '{name}': {e}")

            self._pools.clear()

    def get_all_statistics(self) -> dict[str, PoolStatistics]:
        """Get statistics for all pools.

        Returns:
            Dictionary of pool name to statistics

        """
        return {name: pool.get_statistics() for name, pool in self._pools.items()}
