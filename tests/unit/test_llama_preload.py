"""Unit tests for llama model preloading at server start."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from mcp_server_llm_cli_runner.lean_mcp_interface import LeanMCPInterface
from mcp_server_llm_cli_runner.providers.llama import LLaMAProvider


def _make_interface(provider):
    config_manager = MagicMock()
    config_manager.get_enabled_providers.return_value = ["llama"]
    with patch("mcp_server_llm_cli_runner.lean_mcp_interface.ProviderManager"):
        interface = LeanMCPInterface(config_manager)
    interface.manager.get_provider.return_value = provider
    return interface


def _provider(tmp_path, with_path=True):
    if not with_path:
        return LLaMAProvider(model_path=None)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    return LLaMAProvider(model_path=str(model))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LLAMA_PRELOAD", raising=False)
    monkeypatch.delenv("LLAMA_MODEL_PATH", raising=False)


@pytest.fixture
def llama_cpp():
    fake = MagicMock()
    with (
        patch("mcp_server_llm_cli_runner.providers.llama.LLAMA_CPP_AVAILABLE", True),
        patch("mcp_server_llm_cli_runner.providers.llama.Llama", fake),
    ):
        yield fake


@pytest.fixture
def log():
    with patch("mcp_server_llm_cli_runner.providers.llama.logger") as mock_logger:
        yield mock_logger


async def test_preload_scheduled_and_loads_model(tmp_path, llama_cpp, log):
    provider = _provider(tmp_path)
    interface = _make_interface(provider)
    async with interface._lifespan(None):
        assert interface._preload_task is not None
        assert await interface._preload_task is True
    llama_cpp.assert_called_once()
    assert provider.model_manager.model_loaded
    assert any("preloaded in" in c.args[0] for c in log.info.call_args_list)


@pytest.mark.parametrize("value", ["0", "false", "No"])
async def test_preload_disabled_by_env(tmp_path, llama_cpp, monkeypatch, value):
    monkeypatch.setenv("LLAMA_PRELOAD", value)
    interface = _make_interface(_provider(tmp_path))
    async with interface._lifespan(None):
        assert interface._preload_task is None
    llama_cpp.assert_not_called()


async def test_no_model_path_no_warmup(llama_cpp, log):
    interface = _make_interface(_provider(None, with_path=False))
    async with interface._lifespan(None):
        assert interface._preload_task is None
    llama_cpp.assert_not_called()
    log.warning.assert_not_called()


async def test_llama_not_registered_no_warmup():
    interface = _make_interface(None)
    async with interface._lifespan(None):
        assert interface._preload_task is None


async def test_load_failure_logged_not_raised(tmp_path, llama_cpp, log):
    llama_cpp.side_effect = RuntimeError("boom")
    provider = _provider(tmp_path)
    interface = _make_interface(provider)
    async with interface._lifespan(None):
        assert await interface._preload_task is False
    assert not provider.model_manager.model_loaded
    assert any("preload failed" in c.args[0] for c in log.warning.call_args_list)


async def test_missing_file_warns_not_raises(tmp_path, llama_cpp, log):
    provider = LLaMAProvider(model_path=str(tmp_path / "missing.gguf"))
    assert await provider.preload() is False
    log.warning.assert_called_once()


async def test_startup_does_not_wait_for_load(tmp_path, llama_cpp):
    provider = _provider(tmp_path)
    release = asyncio.Event()
    started = asyncio.Event()

    async def slow_preload() -> bool:
        started.set()
        await release.wait()
        return True

    provider.preload = slow_preload  # type: ignore[method-assign]
    interface = _make_interface(provider)
    async with interface._lifespan(None):
        # entering the lifespan returned while the load is still pending
        assert interface._preload_task is not None
        assert not interface._preload_task.done()
        await started.wait()
    # shutdown cancelled the pending warm-up cleanly
    assert interface._preload_task is None
