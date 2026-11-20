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


def test_init_providers_passes_configured_model_to_both():
    interface = _make_interface(
        ["gemini", "agy"], {"gemini": "gemini-2.5-pro", "agy": "agy-model-x"}
    )
    with (
        patch("mcp_server_llm_cli_runner.providers.gemini.GeminiProvider") as gem,
        patch("mcp_server_llm_cli_runner.providers.agy.AgyProvider") as agy,
    ):
        interface._init_providers()

    gem.assert_called_once_with(model="gemini-2.5-pro")
    agy.assert_called_once_with(model="agy-model-x")


@pytest.mark.parametrize("configured", ["default", "", None])
def test_init_providers_default_model_means_no_override(configured):
    interface = _make_interface(
        ["gemini", "agy"], {"gemini": configured, "agy": configured}
    )
    with (
        patch("mcp_server_llm_cli_runner.providers.gemini.GeminiProvider") as gem,
        patch("mcp_server_llm_cli_runner.providers.agy.AgyProvider") as agy,
    ):
        interface._init_providers()

    gem.assert_called_once_with()
    agy.assert_called_once_with(model="")
