"""Tests for PR #25 review follow-ups: probe cleanup and config precedence."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_server_llm_cli_runner.core.models import ProviderConfig, ProviderType
from mcp_server_llm_cli_runner.providers.agy import AgyProvider


def _hanging_process() -> Mock:
    """Mock subprocess whose communicate() and wait() never complete."""

    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    process = Mock()
    process.communicate = AsyncMock(side_effect=hang)
    process.wait = AsyncMock(side_effect=hang)
    process.kill = Mock()
    process.returncode = None
    return process


def _reaping_process() -> Mock:
    """Hanging process whose wait() completes once kill() has been called."""
    killed = asyncio.Event()

    async def communicate():
        await asyncio.Event().wait()

    async def wait():
        await killed.wait()
        return -9

    process = Mock()
    process.communicate = AsyncMock(side_effect=communicate)
    process.wait = AsyncMock(side_effect=wait)
    process.kill = Mock(side_effect=killed.set)
    process.returncode = None
    return process


class TestProbeCleanup:
    @pytest.mark.asyncio
    async def test_health_check_timeout_kills_and_reaps(self):
        provider = AgyProvider()
        provider.PROBE_TIMEOUT = 0.05
        process = _reaping_process()
        with patch("asyncio.create_subprocess_exec", return_value=process):
            result = await provider.health_check()

        assert result["status"] == "error"
        process.kill.assert_called_once()
        process.wait.assert_awaited()

    @pytest.mark.asyncio
    async def test_health_check_cancellation_kills_and_reaps(self):
        provider = AgyProvider()
        process = _reaping_process()
        with patch("asyncio.create_subprocess_exec", return_value=process):
            task = asyncio.ensure_future(provider.health_check())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        process.kill.assert_called_once()
        process.wait.assert_awaited()

    @pytest.mark.asyncio
    async def test_is_available_timeout_kills_and_is_unavailable(self):
        provider = AgyProvider()
        provider.PROBE_TIMEOUT = 0.05
        process = _reaping_process()
        with patch("asyncio.create_subprocess_exec", return_value=process):
            assert await provider.is_available() is False

        process.kill.assert_called_once()
        process.wait.assert_awaited()

    @pytest.mark.asyncio
    async def test_is_available_nonzero_exit_unavailable(self):
        provider = AgyProvider()
        process = Mock()
        process.wait = AsyncMock(return_value=1)
        process.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=process):
            assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_nonzero_exit(self):
        provider = AgyProvider()
        process = Mock()
        process.communicate = AsyncMock(return_value=(b"", b"boom"))
        process.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=process):
            result = await provider.health_check()

        assert result["status"] == "unhealthy"
        assert result["error"] == "boom"


def _config(**kwargs) -> ProviderConfig:
    return ProviderConfig(name="agy", provider_type=ProviderType.AGY, **kwargs)


class TestConfigPrecedence:
    def test_config_only(self):
        provider = AgyProvider(
            _config(model_name="m-cfg", provider_specific={"default_temperature": 0.2})
        )

        assert provider.model == "m-cfg"
        assert provider.temperature == 0.2

    def test_explicit_overrides_config(self):
        provider = AgyProvider(
            _config(model_name="m-cfg", provider_specific={"default_temperature": 0.2}),
            model="m-arg",
            temperature=0.9,
        )

        assert provider.model == "m-arg"
        assert provider.temperature == 0.9

    def test_neither_uses_defaults(self):
        provider = AgyProvider()
        assert provider.model == ""
        assert provider.temperature == 0.7

        bare = AgyProvider(_config())
        assert bare.model == ""
        assert bare.temperature == 0.7
