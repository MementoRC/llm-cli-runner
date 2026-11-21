"""Tests for the gemini -> agy deprecation alias."""

import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_llm_cli_runner import lean_mcp_interface as lmi
from mcp_server_llm_cli_runner.core.models import (
    AGY_MODELS,
    map_legacy_model,
    resolve_provider_name,
)
from mcp_server_llm_cli_runner.lean_mcp_interface import LeanMCPInterface


def test_resolve_provider_name():
    assert resolve_provider_name("gemini") == ("agy", True)
    assert resolve_provider_name("agy") == ("agy", False)
    assert resolve_provider_name("llama") == ("llama", False)


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ("gemini-2.5-flash-lite", "gemini-3.8-flash-low"),
        ("gemini-2.5-flash", "gemini-3.8-flash-medium"),
        ("gemini-2.5-pro", "gemini-3.8-flash-high"),
    ],
)
def test_map_legacy_model_tiers(legacy, expected):
    assert map_legacy_model(legacy) == (expected, None)


@pytest.mark.parametrize("model", AGY_MODELS)
def test_map_legacy_model_agy_native_passthrough(model):
    assert map_legacy_model(model) == (model, None)


def test_map_legacy_model_unknown_dropped_with_warning():
    new, warning = map_legacy_model("gemini-ultra-9")
    assert new is None
    assert warning is not None
    assert "gemini-ultra-9" in warning


@pytest.mark.parametrize("model", [None, "", "default"])
def test_map_legacy_model_empty_means_default(model):
    assert map_legacy_model(model) == (None, None)


def _response(model: str = "gemini-3.8-flash-medium"):
    return SimpleNamespace(
        provider="agy",
        model=model,
        content="hi",
        success=True,
        tokens_used=1,
        cost=0.0,
        response_time_ms=1,
        metadata={},
    )


def _interface(agy_available: bool = True, registered: bool = True):
    config_manager = MagicMock()
    config_manager.get_enabled_providers.return_value = ["agy"]
    with patch("mcp_server_llm_cli_runner.lean_mcp_interface.ProviderManager"):
        interface = LeanMCPInterface(config_manager)
    manager = interface.manager
    manager.route_request = AsyncMock(return_value=_response())
    agy = MagicMock()
    agy.is_available = AsyncMock(return_value=agy_available)
    manager.get_provider.side_effect = lambda name: (
        agy if (registered and name == "agy") else None
    )
    return interface


@pytest.fixture(autouse=True)
def _reset_warn_flag(monkeypatch):
    monkeypatch.setattr(lmi, "_deprecation_warned", False)


async def test_llm_complete_gemini_routes_to_agy_with_mapping():
    interface = _interface()
    with pytest.warns(DeprecationWarning, match="agy"):
        result = await interface._llm_complete(
            "hello", provider="gemini", model="gemini-2.5-flash-lite"
        )

    request = interface.manager.route_request.call_args.args[0]
    assert request.provider == "agy"
    assert request.model == "gemini-3.8-flash-low"
    assert result["deprecated_alias"] == "gemini->agy"
    assert result["model_mapped"] == "gemini-2.5-flash-lite->gemini-3.8-flash-low"


async def test_llm_complete_unknown_gemini_model_dropped():
    interface = _interface()
    with pytest.warns(DeprecationWarning):
        result = await interface._llm_complete(
            "hello", provider="gemini", model="gemini-ultra-9"
        )

    assert interface.manager.route_request.call_args.args[0].model is None
    assert result["model_mapped"] == "gemini-ultra-9->default"


async def test_llm_complete_warning_emitted_once():
    interface = _interface()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await interface._llm_complete("a", provider="gemini")
        await interface._llm_complete("b", provider="gemini")

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1


async def test_llm_complete_agy_direct_not_mapped():
    interface = _interface()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = await interface._llm_complete(
            "hello", provider="agy", model="gemini-2.5-pro"
        )

    request = interface.manager.route_request.call_args.args[0]
    assert request.model == "gemini-2.5-pro"
    assert "deprecated_alias" not in result
    assert "model_mapped" not in result


@pytest.mark.parametrize("kwargs", [{"registered": False}, {"agy_available": False}])
async def test_llm_complete_gemini_agy_unavailable_clear_error(kwargs):
    interface = _interface(**kwargs)
    with pytest.warns(DeprecationWarning):
        result = await interface._llm_complete("hello", provider="gemini")

    if kwargs.get("registered") is False:
        assert "AGY_CLI_PATH" in result["error"]
        assert "not installed" in result["error"]
    else:
        assert result["error"] == (
            "agy provider unavailable (CLI error or daily quota exhausted)"
        )
    interface.manager.route_request.assert_not_called()


async def test_llm_complete_omitted_provider_sets_default_hint():
    interface = _interface()
    interface.config_manager.get_default_provider.return_value = "llama"

    await interface._llm_complete("hello")

    request = interface.manager.route_request.call_args.args[0]
    assert request.provider is None
    assert request.metadata["preferred_provider"] == "llama"


async def test_llm_complete_omitted_provider_config_default_gemini_hint_no_alias():
    interface = _interface()
    interface.config_manager.get_default_provider.return_value = "gemini"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = await interface._llm_complete("hello")

    request = interface.manager.route_request.call_args.args[0]
    assert request.provider is None
    assert request.metadata["preferred_provider"] == "gemini"
    assert "deprecated_alias" not in result


async def test_llm_complete_explicit_unavailable_returns_error_code():
    interface = _interface()
    manager, _ = _manager(agy_available=False)
    interface.manager.route_request = manager.route_request

    result = await interface._llm_complete("hello", provider="agy")

    assert result["error_code"] == "PROVIDER_UNAVAILABLE"
    assert "error" in result


async def test_llm_complete_notification_labels():
    interface = _interface()
    interface.config_manager.get_default_provider.return_value = "llama"
    interface._send_notification = AsyncMock()

    await interface._llm_complete("hello")

    sent = {c.args[0]: c.args[1] for c in interface._send_notification.call_args_list}
    assert sent["llm/started"]["provider"] == "llama"
    assert sent["llm/completed"]["provider"] == "agy"


def _manager(agy_available: bool, llama_available: bool = True):
    """Real ProviderManager routing over a fake registry."""
    from mcp_server_llm_cli_runner.core.models import ProviderType
    from mcp_server_llm_cli_runner.providers.manager import ProviderManager
    from mcp_server_llm_cli_runner.providers.routing import ProviderRouter

    def _instance(ptype, available):
        inst = MagicMock()
        inst.provider_type = ptype
        inst.is_healthy = available
        inst.generate = AsyncMock(
            return_value=SimpleNamespace(
                provider=ptype.value, model="m", tokens_used=0, cost=0.0
            )
        )
        return inst

    instances = {
        "agy": _instance(ProviderType.AGY, agy_available),
        "llama": _instance(ProviderType.LLAMA, llama_available),
    }
    registry = MagicMock()
    registry.list_available_types.return_value = [
        ProviderType.AGY,
        ProviderType.LLAMA,
    ]
    registry.list_providers.return_value = list(instances)
    registry.get_provider.side_effect = instances.get

    manager = object.__new__(ProviderManager)
    manager.initialized = True
    manager.registry = registry
    manager.router = ProviderRouter(registry)
    manager.logger = MagicMock()
    manager.usage_metrics = {}
    manager.quota_trackers = {}
    return manager, instances


async def test_manager_route_request_gemini_reaches_agy_with_mapped_model():
    from mcp_server_llm_cli_runner.core.models import LLMRequest

    manager, instances = _manager(agy_available=True)
    request = LLMRequest(prompt="hi", provider="gemini", model="gemini-2.5-pro")

    await manager.route_request(request)

    sent = instances["agy"].generate.call_args.args[0]
    assert sent.provider == "agy"
    assert sent.model == "gemini-3.8-flash-high"
    instances["llama"].generate.assert_not_called()
    assert request.provider == "gemini"


async def test_manager_route_request_gemini_agy_unavailable_raises_no_fallback():
    from mcp_server_llm_cli_runner.core.errors import ProviderError
    from mcp_server_llm_cli_runner.core.models import LLMRequest

    manager, instances = _manager(agy_available=False)
    request = LLMRequest(prompt="hi", provider="gemini", model="gemini-2.5-pro")

    with pytest.raises(ProviderError, match="AGY_CLI_PATH") as exc_info:
        await manager.route_request(request)

    assert exc_info.value.error_code == "PROVIDER_UNAVAILABLE"
    instances["llama"].generate.assert_not_called()


async def test_manager_route_request_unknown_provider_raises_invalid_provider():
    from mcp_server_llm_cli_runner.core.errors import ProviderError
    from mcp_server_llm_cli_runner.core.models import LLMRequest

    manager, _ = _manager(agy_available=True)
    request = LLMRequest(prompt="hi", provider="nope")

    with pytest.raises(ProviderError) as exc_info:
        await manager.route_request(request)

    assert exc_info.value.error_code == "INVALID_PROVIDER"
    assert "llama" in exc_info.value.context["valid_providers"]


async def test_manager_route_request_non_aliased_unavailable_raises_provider_unavailable():
    from mcp_server_llm_cli_runner.core.errors import ProviderError
    from mcp_server_llm_cli_runner.core.models import LLMRequest

    manager, instances = _manager(agy_available=False)
    request = LLMRequest(prompt="hi", provider="agy")

    with pytest.raises(ProviderError) as exc_info:
        await manager.route_request(request)

    assert exc_info.value.error_code == "PROVIDER_UNAVAILABLE"
    instances["llama"].generate.assert_not_called()


async def test_get_healthy_provider_gemini_returns_agy():
    from mcp_server_llm_cli_runner.providers.manager import ProviderManager

    manager = object.__new__(ProviderManager)
    manager.initialized = True
    manager.health_monitor = MagicMock()
    manager.health_monitor.get_healthy_providers.return_value = ["llama", "agy"]
    manager.registry = MagicMock()
    manager.registry.get_provider.side_effect = lambda name: f"inst-{name}"

    assert await manager.get_healthy_provider("gemini") == "inst-agy"


def test_list_providers_hides_gemini_and_lists_deprecation():
    interface = _interface()
    interface.manager.health_monitor.get_all_health_statuses.return_value = {}
    interface.manager.registry.list_providers.return_value = ["gemini", "agy"]
    interface.config_manager.get_enabled_providers.return_value = ["gemini", "agy"]

    result = interface._list_providers()

    assert result["providers"] == ["agy"]
    assert result["count"] == 1
    assert "gemini" not in result["configured"]
    assert result["deprecated"] == {"gemini": "agy"}


async def test_provider_info_gemini_returns_agy_info_flagged_deprecated():
    interface = _interface()
    interface.manager.get_provider_stats.return_value = {}
    interface.manager.health_monitor.get_provider_health.return_value = None

    info = await interface._provider_info("gemini")

    assert info["provider"] == "agy"
    assert info["deprecated"] is True
    assert info["redirected_to"] == "agy"


async def test_provider_info_agy_not_flagged():
    interface = _interface()
    interface.manager.get_provider_stats.return_value = {}
    interface.manager.health_monitor.get_provider_health.return_value = None

    info = await interface._provider_info("agy")

    assert "deprecated" not in info
