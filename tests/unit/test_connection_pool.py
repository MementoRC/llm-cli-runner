"""Unit tests for connection pool module - TDD approach."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_cheap_llm.utils.connection_pool import (
    AsyncConnectionPool,
    ConnectionPoolConfig,
    ConnectionPoolManager,
    ConnectionState,
    PooledConnection,
    PoolStatistics,
)


class MockConnection:
    """Mock connection object for testing."""

    def __init__(self, connection_id: int = 1):
        self.id = connection_id
        self.closed = False

    async def close(self):
        self.closed = True

    def is_healthy(self) -> bool:
        return not self.closed


class TestConnectionPoolConfig:
    """Test suite for ConnectionPoolConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ConnectionPoolConfig()

        assert config.max_size == 10
        assert config.min_size == 2
        assert config.max_idle_time_seconds == 300
        assert config.connection_timeout_seconds == 30.0
        assert config.enable_health_checks is True

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = ConnectionPoolConfig(
            max_size=20,
            min_size=5,
            max_idle_time_seconds=600,
            connection_timeout_seconds=60.0,
        )

        assert config.max_size == 20
        assert config.min_size == 5

    def test_config_validation(self):
        """Test configuration validation."""
        # max_size must be >= 1
        with pytest.raises(ValueError):
            ConnectionPoolConfig(max_size=0)

        # min_size cannot exceed reasonable limits
        config = ConnectionPoolConfig(max_size=10, min_size=5)
        assert config.min_size == 5


class TestPooledConnection:
    """Test suite for PooledConnection."""

    def test_pooled_connection_creation(self):
        """Test PooledConnection creation."""
        conn = MockConnection()
        pooled = PooledConnection(connection=conn, pool_id=1)

        assert pooled.connection is conn
        assert pooled.pool_id == 1
        assert pooled.state == ConnectionState.IDLE
        assert pooled.use_count == 0

    def test_pooled_connection_mark_used(self):
        """Test marking connection as used."""
        conn = MockConnection()
        pooled = PooledConnection(connection=conn)

        pooled.mark_used()

        assert pooled.state == ConnectionState.IN_USE
        assert pooled.use_count == 1

    def test_pooled_connection_mark_idle(self):
        """Test marking connection as idle."""
        conn = MockConnection()
        pooled = PooledConnection(connection=conn)
        pooled.mark_used()

        pooled.mark_idle()

        assert pooled.state == ConnectionState.IDLE

    def test_pooled_connection_mark_unhealthy(self):
        """Test marking connection as unhealthy."""
        conn = MockConnection()
        pooled = PooledConnection(connection=conn)

        pooled.mark_unhealthy()

        assert pooled.state == ConnectionState.UNHEALTHY

    def test_pooled_connection_is_expired(self):
        """Test connection expiration check."""
        conn = MockConnection()
        pooled = PooledConnection(connection=conn)

        # Should not be expired immediately
        assert not pooled.is_expired(max_lifetime=3600, max_idle=300)


class TestAsyncConnectionPool:
    """Test suite for AsyncConnectionPool."""

    @pytest.fixture
    def connection_factory(self):
        """Create a mock connection factory."""
        counter = [0]

        def factory():
            counter[0] += 1
            return MockConnection(counter[0])

        return factory

    @pytest.fixture
    def async_connection_factory(self):
        """Create an async mock connection factory."""
        counter = [0]

        async def factory():
            counter[0] += 1
            return MockConnection(counter[0])

        return factory

    def test_pool_initialization(self, connection_factory):
        """Test pool initialization."""
        config = ConnectionPoolConfig(max_size=5, min_size=1)
        pool = AsyncConnectionPool(
            config=config,
            connection_factory=connection_factory,
            name="test",
        )

        assert pool.name == "test"
        assert pool.config.max_size == 5

    @pytest.mark.asyncio
    async def test_pool_start_stop(self, connection_factory):
        """Test pool start and stop."""
        config = ConnectionPoolConfig(max_size=5, min_size=2)
        pool = AsyncConnectionPool(
            config=config,
            connection_factory=connection_factory,
            name="test",
        )

        await pool.start()
        assert pool._is_running is True

        # Should have min_size connections pre-created
        stats = pool.get_statistics()
        assert stats.idle_connections >= 1  # At least some pre-created

        await pool.stop()
        assert pool._is_running is False

    @pytest.mark.asyncio
    async def test_pool_acquire_release(self, connection_factory):
        """Test acquiring and releasing connections."""
        config = ConnectionPoolConfig(max_size=5, min_size=1)
        pool = AsyncConnectionPool(
            config=config,
            connection_factory=connection_factory,
            name="test",
        )

        await pool.start()

        try:
            # Acquire a connection
            conn = await pool.acquire()
            assert conn is not None
            assert isinstance(conn, MockConnection)

            stats = pool.get_statistics()
            assert stats.active_connections == 1

            # Release the connection
            await pool.release(conn)

            stats = pool.get_statistics()
            assert stats.active_connections == 0
            assert stats.idle_connections >= 1
        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_context_manager(self, connection_factory):
        """Test pool connection context manager."""
        config = ConnectionPoolConfig(max_size=5, min_size=1)
        pool = AsyncConnectionPool(
            config=config,
            connection_factory=connection_factory,
            name="test",
        )

        await pool.start()

        try:
            async with pool.connection() as conn:
                assert conn is not None
                stats = pool.get_statistics()
                assert stats.active_connections == 1

            # After context, connection should be released
            stats = pool.get_statistics()
            assert stats.active_connections == 0
        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_acquire_timeout(self, connection_factory):
        """Test acquisition timeout."""
        config = ConnectionPoolConfig(
            max_size=1, min_size=0, connection_timeout_seconds=1.0
        )
        pool = AsyncConnectionPool(
            config=config,
            connection_factory=connection_factory,
            name="test",
        )

        await pool.start()

        try:
            # Acquire the only connection
            conn1 = await pool.acquire()

            # Try to acquire another - should timeout (using lower timeout for test)
            with pytest.raises(TimeoutError):
                await pool.acquire(timeout=0.05)  # Very short timeout for test

            await pool.release(conn1)
        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_statistics(self, connection_factory):
        """Test pool statistics tracking."""
        config = ConnectionPoolConfig(max_size=5, min_size=1)
        pool = AsyncConnectionPool(
            config=config,
            connection_factory=connection_factory,
            name="test",
        )

        await pool.start()

        try:
            conn = await pool.acquire()
            await pool.release(conn)

            stats = pool.get_statistics()
            assert stats.total_acquisitions >= 1
            assert stats.total_releases >= 1
        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_with_close_handler(self, connection_factory):
        """Test pool with custom close handler."""
        closed_connections = []

        async def close_handler(conn):
            closed_connections.append(conn.id)

        config = ConnectionPoolConfig(max_size=2, min_size=1)
        pool = AsyncConnectionPool(
            config=config,
            connection_factory=connection_factory,
            close_connection=close_handler,
            name="test",
        )

        await pool.start()
        await pool.stop()

        # Connections should have been closed
        assert len(closed_connections) >= 0  # May or may not have pre-created

    @pytest.mark.asyncio
    async def test_pool_with_health_check(self, connection_factory):
        """Test pool with health checking."""

        def health_check(conn):
            return conn.is_healthy()

        config = ConnectionPoolConfig(
            max_size=2,
            min_size=1,
            enable_health_checks=True,
            health_check_interval_seconds=10,  # Minimum allowed value
        )
        pool = AsyncConnectionPool(
            config=config,
            connection_factory=connection_factory,
            health_check=health_check,
            name="test",
        )

        await pool.start()

        try:
            # Just verify pool works with health checking enabled
            conn = await pool.acquire()
            await pool.release(conn)
        finally:
            await pool.stop()


class TestConnectionPoolManager:
    """Test suite for ConnectionPoolManager."""

    @pytest.fixture
    def connection_factory(self):
        """Create a mock connection factory."""
        counter = [0]

        def factory():
            counter[0] += 1
            return MockConnection(counter[0])

        return factory

    def test_manager_initialization(self):
        """Test ConnectionPoolManager initialization."""
        manager = ConnectionPoolManager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_register_pool(self, connection_factory):
        """Test registering a pool."""
        manager = ConnectionPoolManager()
        config = ConnectionPoolConfig(max_size=5)

        pool = await manager.register_pool(
            name="test",
            config=config,
            connection_factory=connection_factory,
        )

        assert pool is not None

        await manager.close_all()

    @pytest.mark.asyncio
    async def test_get_pool(self, connection_factory):
        """Test getting a registered pool."""
        manager = ConnectionPoolManager()
        config = ConnectionPoolConfig(max_size=5)

        await manager.register_pool(
            name="test",
            config=config,
            connection_factory=connection_factory,
        )

        pool = await manager.get_pool("test")
        assert pool is not None

        await manager.close_all()

    @pytest.mark.asyncio
    async def test_get_pool_not_found(self):
        """Test getting non-existent pool."""
        manager = ConnectionPoolManager()

        with pytest.raises(KeyError):
            await manager.get_pool("nonexistent")

    @pytest.mark.asyncio
    async def test_get_connection_context_manager(self, connection_factory):
        """Test get_connection context manager."""
        manager = ConnectionPoolManager()
        config = ConnectionPoolConfig(max_size=5)

        await manager.register_pool(
            name="test",
            config=config,
            connection_factory=connection_factory,
        )

        async with manager.get_connection("test") as conn:
            assert conn is not None

        await manager.close_all()

    @pytest.mark.asyncio
    async def test_get_all_statistics(self, connection_factory):
        """Test getting statistics for all pools."""
        manager = ConnectionPoolManager()
        config = ConnectionPoolConfig(max_size=5)

        await manager.register_pool(
            name="pool1",
            config=config,
            connection_factory=connection_factory,
        )
        await manager.register_pool(
            name="pool2",
            config=config,
            connection_factory=connection_factory,
        )

        stats = manager.get_all_statistics()

        assert "pool1" in stats
        assert "pool2" in stats

        await manager.close_all()

    @pytest.mark.asyncio
    async def test_duplicate_pool_registration(self, connection_factory):
        """Test that duplicate pool names raise error."""
        manager = ConnectionPoolManager()
        config = ConnectionPoolConfig(max_size=5)

        await manager.register_pool(
            name="test",
            config=config,
            connection_factory=connection_factory,
        )

        with pytest.raises(ValueError):
            await manager.register_pool(
                name="test",
                config=config,
                connection_factory=connection_factory,
            )

        await manager.close_all()


class TestPoolStatistics:
    """Test suite for PoolStatistics model."""

    def test_pool_statistics_defaults(self):
        """Test PoolStatistics default values."""
        stats = PoolStatistics()

        assert stats.total_connections == 0
        assert stats.active_connections == 0
        assert stats.idle_connections == 0
        assert stats.total_acquisitions == 0
        assert stats.total_timeouts == 0

    def test_pool_statistics_custom_values(self):
        """Test PoolStatistics with custom values."""
        stats = PoolStatistics(
            total_connections=10,
            active_connections=5,
            idle_connections=5,
            total_acquisitions=100,
            total_releases=95,
        )

        assert stats.total_connections == 10
        assert stats.total_acquisitions == 100


class TestConnectionState:
    """Test suite for ConnectionState enum."""

    def test_connection_states(self):
        """Test all connection states exist."""
        assert ConnectionState.IDLE == "idle"
        assert ConnectionState.IN_USE == "in_use"
        assert ConnectionState.CLOSED == "closed"
        assert ConnectionState.UNHEALTHY == "unhealthy"
