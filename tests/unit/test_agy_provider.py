"""Unit tests for the agy (Antigravity CLI) provider."""

from unittest.mock import AsyncMock, patch

import pytest

from mcp_server_llm_cli_runner.core.errors import (
    ConfigurationError,
    ProviderError,
    RateLimitError,
)
from mcp_server_llm_cli_runner.core.models import LLMRequest
from mcp_server_llm_cli_runner.providers.agy import AgyProvider

ERROR_JSON = (
    '{"conversation_id":"","status":"ERROR","response":"","error":"Eligibility '
    "check failed: Your current account is not eligible for Antigravity. Verify "
    'your account to continue.","duration_seconds":0,"num_turns":0,"usage":'
    '{"input_tokens":0,"output_tokens":0,"thinking_tokens":0,"cache_read_tokens":0,'
    '"total_tokens":0},"finished_reason":"error"}'
)

SUCCESS_JSON = (
    '{"conversation_id":"b65239e3-e1d6-455b-b003-f7ad6cfcc7c4","status":"SUCCESS",'
    '"response":"OK\\n","duration_seconds":11.752439959,"num_turns":1,"usage":'
    '{"input_tokens":9490,"output_tokens":1192,"thinking_tokens":1191,'
    '"cache_read_tokens":0,"total_tokens":10682}}'
)


def _process(stdout: str = "", stderr: str = "", returncode: int = 0) -> AsyncMock:
    process = AsyncMock()
    process.communicate.return_value = (stdout.encode(), stderr.encode())
    process.returncode = returncode
    return process


class TestAgyCommand:
    def test_default_command(self, monkeypatch):
        monkeypatch.delenv("AGY_CLI_PATH", raising=False)
        cmd = AgyProvider()._build_command(prompt="hi", model="")
        assert cmd == [
            "agy",
            "--mode",
            "plan",
            "--output-format",
            "json",
            "-p",
            "hi",
        ]
        assert "--model" not in cmd

    def test_default_model_literal_skipped(self):
        assert "--model" not in AgyProvider()._build_command("hi", model="default")

    def test_model_set(self):
        cmd = AgyProvider()._build_command("hi", model="some-model")
        assert cmd[cmd.index("--model") + 1] == "some-model"

    def test_binary_override(self, monkeypatch):
        monkeypatch.setenv("AGY_CLI_PATH", "/opt/agy")
        assert AgyProvider()._build_command("hi")[0] == "/opt/agy"


class TestAgyExecution:
    @pytest.mark.asyncio
    async def test_generate_success(self):
        provider = AgyProvider()
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.return_value = _process(stdout=SUCCESS_JSON)
            response = await provider.generate(LLMRequest(prompt="hi"))

        assert response.content == "OK"
        assert response.provider == "agy"
        assert response.tokens_used == 10682
        assert response.metadata["usage"] == {
            "prompt": 9490,
            "completion": 1192,
            "total": 10682,
        }

    @pytest.mark.asyncio
    async def test_error_json_on_stderr_raises_auth_error(self):
        provider = AgyProvider(max_retries=1)
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.return_value = _process(stderr=ERROR_JSON, returncode=1)
            with pytest.raises(ConfigurationError, match="not eligible"):
                await provider._execute_with_retries(
                    provider._build_command("hi"), LLMRequest(prompt="hi")
                )

    @pytest.mark.asyncio
    async def test_cli_not_installed(self):
        provider = AgyProvider()
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            with pytest.raises(ConfigurationError, match="not installed"):
                await provider._execute_with_retries(
                    provider._build_command("hi"), LLMRequest(prompt="hi")
                )

    @pytest.mark.asyncio
    async def test_stream_generate_single_chunk(self):
        provider = AgyProvider()
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.return_value = _process(stdout=SUCCESS_JSON)
            chunks = await provider.stream_generate(LLMRequest(prompt="hi"))

        assert len(chunks) == 1
        assert chunks[0].content == "OK"
        assert chunks[0].is_final is True


class TestAgyHealth:
    @pytest.mark.asyncio
    async def test_health_check_version(self):
        provider = AgyProvider()
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.return_value = _process(stdout="1.5.11\n")
            result = await provider.health_check()

        assert result["status"] == "healthy"
        assert result["version"] == "1.5.11"
        assert result["provider"] == "agy"
        assert mock_exec.call_args.args[:2] == ("agy", "--version")

    @pytest.mark.asyncio
    async def test_health_check_not_installed(self):
        provider = AgyProvider()
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await provider.health_check()

        assert result["status"] == "unavailable"
        assert "not installed" in result["error"]


class TestAgyMessageWording:
    @pytest.mark.asyncio
    async def test_quota_exhausted_message_names_agy(self):
        provider = AgyProvider(daily_quota=0)
        with pytest.raises(RateLimitError) as exc_info:
            await provider.generate(LLMRequest(prompt="hi"))

        assert "agy (Antigravity) daily quota exhausted" in str(exc_info.value)
        assert "Gemini" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_error_wrapper_names_agy(self):
        provider = AgyProvider()
        with pytest.raises(ProviderError) as exc_info:
            await provider.generate(LLMRequest(prompt=" "))

        assert "agy (Antigravity) generation failed" in str(exc_info.value)
        assert "Gemini" not in str(exc_info.value)

    def test_display_name(self):
        assert AgyProvider()._display_name() == "agy (Antigravity)"
