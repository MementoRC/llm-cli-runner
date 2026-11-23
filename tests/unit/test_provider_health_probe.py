"""Health probes must reflect real availability and not pollute usage counters."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_llm_cli_runner.core.errors import ProviderError
from mcp_server_llm_cli_runner.core.models import LLMRequest, ProviderStatus
from mcp_server_llm_cli_runner.lean_mcp_interface import LeanMCPInterface
from mcp_server_llm_cli_runner.providers.manager import ProviderManager


class _Stub:
    """Minimal provider double with a controllable is_available()."""

    def __init__(self, available=True, check_error=None, hang=False):
        self.available = available
        self.check_error = check_error
        self.hang = hang
        self.generate = AsyncMock(
            return_value=SimpleNamespace(
                success=True, tokens_used=1, cost=0.0, content="x"
            )
        )

    async def is_available(self) -> bool:
        if self.hang:
            await asyncio.sleep(60)
        if self.check_error:
            raise self.check_error
        return self.available


def _manager(**providers):
    manager = ProviderManager()
    for name, stub in providers.items():
        manager.register_provider(name, stub)
    return manager


def _interface(manager):
    with patch("mcp_server_llm_cli_runner.lean_mcp_interface.ProviderManager"):
        interface = LeanMCPInterface(MagicMock())
    interface.manager = manager
    interface.config_manager.get_enabled_providers.return_value = list(
        manager.registry.list_providers()
    )
    return interface


async def test_unavailable_provider_reported_unavailable_in_both_tools():
    interface = _interface(_manager(agy=_Stub(available=False), llama=_Stub()))

    listed = (await interface._list_providers())["health_status"]
    info = await interface._provider_info("agy")

    assert listed["agy"]["available"] is False
    assert listed["agy"]["status"] == ProviderStatus.UNAVAILABLE.value
    assert info["available"] is False
    assert info["health"]["status"] == ProviderStatus.UNAVAILABLE.value
    assert listed["llama"]["available"] is True
    assert listed["llama"]["status"] == ProviderStatus.HEALTHY.value


async def test_probes_do_not_change_usage_counters():
    manager = _manager(agy=_Stub())
    monitor = manager.health_monitor

    for _ in range(3):
        status = await monitor.probe("agy")

    assert status.probe_count == 3
    assert status.last_probe_at is not None
    assert status.total_requests == 0
    assert status.successful_requests == 0
    assert status.get_success_rate() == 0.0


async def test_probe_failures_counted_separately():
    manager = _manager(agy=_Stub(available=False))

    status = await manager.health_monitor.probe("agy")

    assert status.probe_failures == 1
    assert status.total_requests == 0


async def test_real_completion_success_and_failure_update_usage():
    stub = _Stub()
    manager = _manager(agy=stub)
    manager.initialized = True
    manager.router = MagicMock()
    manager.router.route_request.return_value = SimpleNamespace(
        selected_provider=SimpleNamespace(value="agy")
    )
    request = LLMRequest(prompt="hi", provider="agy")

    await manager.route_request(request)
    health = manager.health_monitor.get_provider_health("agy")
    assert (health.total_requests, health.successful_requests) == (1, 1)
    assert health.get_success_rate() == 100.0

    stub.generate.side_effect = RuntimeError("boom")
    with pytest.raises(ProviderError):
        await manager.route_request(request)
    assert (health.total_requests, health.successful_requests) == (2, 1)
    assert health.get_success_rate() == 50.0
    assert health.last_error == "boom"
    assert health.probe_count == 0


async def test_generate_records_completion():
    manager = _manager(agy=_Stub())
    stub = manager.get_provider("agy")
    stub.generate.return_value = SimpleNamespace(
        success=True,
        tokens_used=1,
        cost=0.0,
        content="x",
        provider="agy",
        model="m",
        response_time_ms=1,
        error_message=None,
        metadata={},
    )

    await manager.generate("hi", provider="agy")

    health = manager.health_monitor.get_provider_health("agy")
    assert health.total_requests == 1


async def test_health_check_exception_yields_unhealthy():
    manager = _manager(agy=_Stub(check_error=RuntimeError("cli exploded")))

    status = await manager.health_monitor.probe("agy")

    assert status.status == ProviderStatus.UNHEALTHY
    assert status.available is False
    assert "cli exploded" in (status.last_error or "")


async def test_health_check_timeout_yields_unhealthy():
    manager = _manager(agy=_Stub(hang=True))
    manager.health_monitor.PROBE_TIMEOUT = 0.05

    status = await manager.health_monitor.probe("agy")

    assert status.status == ProviderStatus.UNHEALTHY
    assert status.available is False


async def test_llama_probe_does_not_load_model(tmp_path):
    from mcp_server_llm_cli_runner.providers import llama as llama_mod
    from mcp_server_llm_cli_runner.providers.llama import LLaMAProvider

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"x")
    provider = LLaMAProvider(model_path=str(model_file))
    provider.model_manager.load_model = AsyncMock()
    manager = _manager(llama=provider)

    with patch.object(llama_mod, "LLAMA_CPP_AVAILABLE", True):
        status = await manager.health_monitor.probe("llama")

    provider.model_manager.load_model.assert_not_called()
    assert status.status == ProviderStatus.HEALTHY


async def test_unsuccessful_response_records_error_message():
    manager = _manager(agy=_Stub())
    stub = manager.get_provider("agy")
    stub.generate.return_value = SimpleNamespace(
        success=False,
        tokens_used=0,
        cost=0.0,
        content="",
        provider="agy",
        model="m",
        response_time_ms=1,
        error_message="quota exceeded",
        metadata={},
    )

    await manager.generate("hi", provider="agy")

    health = manager.health_monitor.get_provider_health("agy")
    assert health.total_requests == 1
    assert health.successful_requests == 0
    assert health.last_error == "quota exceeded"


async def test_unsuccessful_route_response_records_error_message():
    stub = _Stub()
    stub.generate.return_value = SimpleNamespace(
        success=False, tokens_used=0, cost=0.0, error_message="bad output"
    )
    manager = _manager(agy=stub)
    manager.initialized = True
    manager.router = MagicMock()
    manager.router.route_request.return_value = SimpleNamespace(
        selected_provider=SimpleNamespace(value="agy")
    )

    await manager.route_request(LLMRequest(prompt="hi", provider="agy"))

    health = manager.health_monitor.get_provider_health("agy")
    assert health.last_error == "bad output"


async def test_older_probe_finishing_late_does_not_overwrite_newer():
    class _Ordered(_Stub):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def is_available(self) -> bool:
            self.calls += 1
            if self.calls == 1:  # older probe: slow, reports unavailable
                await asyncio.sleep(0.1)
                return False
            return True  # newer probe: fast, reports available

    manager = _manager(agy=_Ordered())
    monitor = manager.health_monitor

    await asyncio.gather(monitor.probe("agy"), monitor.probe("agy"))

    assert monitor.get_provider_health("agy").status == ProviderStatus.HEALTHY


async def test_list_providers_probes_concurrently():
    class _Slow(_Stub):
        async def is_available(self) -> bool:
            await asyncio.sleep(0.2)
            return True

    interface = _interface(_manager(agy=_Slow(), llama=_Slow(), claude=_Slow()))

    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await interface._list_providers()
    elapsed = loop.time() - started

    assert set(result["health_status"]) == {"agy", "llama", "claude"}
    assert elapsed < 0.5  # sequential probing would take >= 0.6s


async def test_llama_model_unreadable_is_not_available(tmp_path):
    import os

    from mcp_server_llm_cli_runner.providers.llama import LLaMAProvider

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"x")
    provider = LLaMAProvider(model_path=str(model_file))

    with patch.object(os, "access", return_value=False):
        assert provider.model_manager.is_model_available() is False
    assert provider.model_manager.is_model_available() is True


async def test_llama_probe_unavailable_when_model_missing(tmp_path):
    from mcp_server_llm_cli_runner.providers import llama as llama_mod
    from mcp_server_llm_cli_runner.providers.llama import LLaMAProvider

    provider = LLaMAProvider(model_path=str(tmp_path / "missing.gguf"))
    manager = _manager(llama=provider)

    with patch.object(llama_mod, "LLAMA_CPP_AVAILABLE", True):
        status = await manager.health_monitor.probe("llama")

    assert status.status == ProviderStatus.UNAVAILABLE
