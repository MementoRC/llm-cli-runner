"""Unit tests for provider-specific configuration classes."""

import pytest
from pydantic import ValidationError

from mcp_server_llm_cli_runner.core.models import (  # type: ignore[import-not-found]
    ProviderConfig,
    ProviderType,
)


class TestProviderConfig:
    """Test basic provider configuration functionality."""

    def test_provider_config_basic(self):
        """Test basic provider configuration."""
        config = ProviderConfig(
            name="test",
            provider_type=ProviderType.GEMINI,
            models=["gemini-pro"],
            rate_limit={"requests_per_minute": 60},
        )

        assert config.name == "test"
        assert config.provider_type == ProviderType.GEMINI
        assert config.models == ["gemini-pro"]
        assert config.rate_limit == {"requests_per_minute": 60}
        assert config.enabled is True
        assert config.timeout == 120  # CLI LLMs need longer timeout

    def test_provider_config_validation(self):
        """Test provider configuration validation."""
        # Test invalid timeout validation
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                provider_type=ProviderType.GEMINI,
                models=["gemini-pro"],
                timeout=0,  # Invalid: must be >= 1
            )

    def test_provider_config_endpoint_validation(self):
        """Test endpoint URL validation."""
        # Valid endpoints
        valid_endpoints = [
            "https://api.openai.com/v1",
            "https://generativelanguage.googleapis.com/v1",
            "https://api.anthropic.com/v1",
        ]

        for endpoint in valid_endpoints:
            config = ProviderConfig(
                name="test",
                provider_type=ProviderType.OPENAI,
                models=["gpt-3.5-turbo"],
                base_url=endpoint,
            )
            assert config.base_url == endpoint

    def test_provider_config_model_settings(self):
        """Test model settings configuration."""
        provider_specific = {
            "gpt-4": {"max_tokens": 8192},
            "gpt-3.5-turbo": {"max_tokens": 4096},
        }

        config = ProviderConfig(
            name="test",
            provider_type=ProviderType.OPENAI,
            models=["gpt-4"],
            provider_specific=provider_specific,
        )

        assert config.provider_specific == provider_specific
        assert len(config.provider_specific) == 2


class TestGeminiDeprecatedDefaults:
    def test_gemini_default_disabled(self):
        from mcp_server_llm_cli_runner.utils.config import ConfigManager

        defaults = ConfigManager()._create_default_providers()
        gemini = next(p for p in defaults if p["name"] == "gemini")
        assert gemini["enabled"] is False

    def test_default_provider_is_llama(self, monkeypatch):
        from mcp_server_llm_cli_runner.utils.config import ServerConfig

        monkeypatch.delenv("MCP_DEFAULT_PROVIDER", raising=False)
        assert ServerConfig().default_provider == "llama"


class TestAgyDefaultEnabled:
    """agy default enablement depends on the binary being discoverable."""

    @staticmethod
    def _agy_default() -> dict:
        from mcp_server_llm_cli_runner.utils.config import ConfigManager

        defaults = ConfigManager()._create_default_providers()
        return next(p for p in defaults if p["name"] == "agy")

    def test_agy_disabled_when_binary_missing(self, monkeypatch):
        monkeypatch.delenv("AGY_CLI_PATH", raising=False)
        monkeypatch.setattr("shutil.which", lambda _name: None)
        assert self._agy_default()["enabled"] is False

    def test_agy_enabled_when_binary_found(self, monkeypatch):
        monkeypatch.delenv("AGY_CLI_PATH", raising=False)
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/agy")
        assert self._agy_default()["enabled"] is True

    def test_agy_uses_env_override_for_lookup(self, monkeypatch):
        monkeypatch.setenv("AGY_CLI_PATH", "/opt/custom/agy")
        seen: list[str] = []

        def fake_which(name: str) -> str:
            seen.append(name)
            return name

        monkeypatch.setattr("shutil.which", fake_which)
        assert self._agy_default()["enabled"] is True
        # other providers (e.g. claude) also probe their own binaries
        assert [name for name in seen if "agy" in name] == ["/opt/custom/agy"]
