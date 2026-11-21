"""Unit tests for LeanMCPInterface._init_providers model propagation."""

from unittest.mock import MagicMock, patch

import pytest

from mcp_server_llm_cli_runner.lean_mcp_interface import LeanMCPInterface


def _make_interface(enabled: list[str], models: dict[str, str | None]):
    config_manager = MagicMock()
    config_manager.get_enabled_providers.return_value = enabled
    config_manager.get_provider_config.side_effect = lambda name: (
        {"model_name": models[name]} if name in models else None
    )
    with patch("mcp_server_llm_cli_runner.lean_mcp_interface.ProviderManager"):
        interface = LeanMCPInterface(config_manager)
    return interface


def _assert_only_agy_registered(interface, agy):
    interface.manager.register_provider.assert_called_once_with("agy", agy.return_value)
    interface.manager.registry.register_provider.assert_called_once_with(agy)


def test_init_providers_explicit_agy_model_wins_over_gemini_model():
    interface = _make_interface(
        ["gemini", "agy"], {"gemini": "gemini-2.5-pro", "agy": "agy-model-x"}
    )
    with patch("mcp_server_llm_cli_runner.providers.agy.AgyProvider") as agy:
        interface._init_providers()

    agy.assert_called_once_with(model="agy-model-x")
    _assert_only_agy_registered(interface, agy)


def test_init_providers_gemini_model_mapped_when_agy_model_unset():
    interface = _make_interface(
        ["gemini", "agy"], {"gemini": "gemini-2.5-pro", "agy": "default"}
    )
    with patch("mcp_server_llm_cli_runner.providers.agy.AgyProvider") as agy:
        interface._init_providers()

    agy.assert_called_once_with(model="gemini-3.8-flash-high")
    _assert_only_agy_registered(interface, agy)


def test_init_providers_gemini_only_registers_agy_with_mapped_model():
    interface = _make_interface(["gemini"], {"gemini": "gemini-2.5-flash"})
    with patch("mcp_server_llm_cli_runner.providers.agy.AgyProvider") as agy:
        interface._init_providers()

    agy.assert_called_once_with(model="gemini-3.8-flash-medium")
    _assert_only_agy_registered(interface, agy)


def test_init_providers_gemini_unmappable_model_falls_back_to_default():
    interface = _make_interface(["gemini"], {"gemini": "gemini-ultra-9"})
    with patch("mcp_server_llm_cli_runner.providers.agy.AgyProvider") as agy:
        interface._init_providers()

    agy.assert_called_once_with(model="")
    _assert_only_agy_registered(interface, agy)


@pytest.mark.parametrize("configured", ["default", "", None])
def test_init_providers_default_model_means_no_override(configured):
    interface = _make_interface(["agy"], {"agy": configured})
    with patch("mcp_server_llm_cli_runner.providers.agy.AgyProvider") as agy:
        interface._init_providers()

    agy.assert_called_once_with(model="")
