"""Unit tests for the claude (Claude Code CLI) provider. No real claude calls."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_server_llm_cli_runner.core.errors import (
    ConfigurationError,
    ProviderError,
    RateLimitError,
)
from mcp_server_llm_cli_runner.core.models import (
    LLMRequest,
    ProviderConfig,
    ProviderType,
    resolve_provider_name,
)
from mcp_server_llm_cli_runner.providers.claude import ClaudeProvider
from mcp_server_llm_cli_runner.providers.routing import ComplexityLevel, ProviderRouter

SUCCESS_JSON = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "OK\n",
        "session_id": "sess-1",
        "total_cost_usd": 0.0123,
        "usage": {"input_tokens": 10, "output_tokens": 5, "future_field": {"x": 1}},
        "unknown_top_level": [1, 2],
    }
)

ISOLATION_FLAGS = [
    "--safe-mode",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--no-session-persistence",
]


@pytest.fixture(autouse=True)
def fake_binary(tmp_path, monkeypatch):
    """Provide an executable stand-in so binary resolution succeeds."""
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("CLAUDE_CLI_PATH", str(binary))
    return str(binary)


def _process(stdout: str = "", stderr: str = "", returncode: int = 0) -> AsyncMock:
    process = AsyncMock()
    process.communicate.return_value = (stdout.encode(), stderr.encode())
    process.returncode = returncode
    process.kill = Mock()
    return process


async def _run(provider: ClaudeProvider, request: LLMRequest, process: AsyncMock):
    with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
        response = await provider.generate(request)
    return response, mock_exec


class TestCommand:
    def test_exact_argv(self, fake_binary):
        cmd = ClaudeProvider()._build_command(model="sonnet")
        assert cmd == [
            fake_binary,
            "-p",
            "--output-format",
            "json",
            "--model=sonnet",
            "--safe-mode",
            "--tools",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--permission-prompts",
            "none",
        ]

    def test_all_six_isolation_flags_present(self):
        cmd = ClaudeProvider()._build_command()
        for flag in ISOLATION_FLAGS:
            assert flag in cmd
        assert cmd[cmd.index("--tools") + 1] == ""
        assert cmd[cmd.index("--permission-prompts") + 1] == "none"

    def test_bare_and_dangerous_flags_absent(self):
        cmd = ClaudeProvider(effort="high")._build_command(system_prompt="be brief")
        joined = " ".join(cmd)
        for forbidden in (
            "--bare",
            "--dangerously-skip-permissions",
            "--allow-dangerously-skip-permissions",
            "bypassPermissions",
            "--add-dir",
            "--mcp-config",
        ):
            assert forbidden not in joined

    def test_effort_only_when_configured(self):
        assert "--effort" not in ClaudeProvider()._build_command()
        cmd = ClaudeProvider(effort="xhigh")._build_command()
        assert cmd[cmd.index("--effort") + 1] == "xhigh"

    def test_invalid_effort_rejected(self):
        with pytest.raises(ConfigurationError, match="effort"):
            ClaudeProvider(effort="turbo")

    def test_system_prompt_only_when_present(self):
        assert not [
            a for a in ClaudeProvider()._build_command() if a.startswith("--system")
        ]
        cmd = ClaudeProvider()._build_command(system_prompt="be brief")
        assert "--system-prompt=be brief" in cmd

    def test_model_override(self):
        cmd = ClaudeProvider()._build_command(model="opus")
        assert "--model=opus" in cmd

    def test_missing_binary_is_provider_unavailable(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CLI_PATH", "/nonexistent/claude")
        with pytest.raises(ProviderError) as exc:
            ClaudeProvider()._build_command()
        assert exc.value.error_code == "PROVIDER_UNAVAILABLE"

    def test_default_binary_from_path(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_CLI_PATH", raising=False)
        with patch("shutil.which", return_value="/usr/bin/claude") as which:
            assert ClaudeProvider()._cli_binary() == "/usr/bin/claude"
        which.assert_called_once_with("claude")


class TestConfigFallback:
    def test_default_model_is_sonnet(self):
        assert ClaudeProvider().model == "sonnet"

    def test_config_model_and_temperature(self):
        config = ProviderConfig(
            name="claude",
            provider_type=ProviderType.ANTHROPIC,
            model_name="opus",
            timeout=33,
            provider_specific={"default_temperature": 0.2, "effort": "low"},
        )
        provider = ClaudeProvider(config=config)
        assert provider.model == "opus"
        assert provider.temperature == 0.2
        assert provider.effort == "low"
        assert provider.timeout == 33

    def test_explicit_args_win_over_config(self):
        config = ProviderConfig(
            name="claude",
            provider_type=ProviderType.ANTHROPIC,
            model_name="opus",
            provider_specific={"default_temperature": 0.2},
        )
        provider = ClaudeProvider(config=config, model="haiku", temperature=0.9)
        assert (provider.model, provider.temperature) == ("haiku", 0.9)

    def test_config_default_literal_maps_to_sonnet(self):
        config = ProviderConfig(
            name="claude", provider_type=ProviderType.ANTHROPIC, model_name="default"
        )
        assert ClaudeProvider(config=config).model == "sonnet"


class TestExecution:
    @pytest.mark.asyncio
    async def test_generate_success_and_plumbing(self, fake_binary):
        provider = ClaudeProvider()
        request = LLMRequest(
            prompt="hello there", model="opus", system_prompt="be terse"
        )
        process = _process(stdout=SUCCESS_JSON)
        response, mock_exec = await _run(provider, request, process)

        assert response.content == "OK"
        assert response.provider == "claude"
        assert response.model == "opus"
        assert response.tokens_used == 15
        assert response.metadata["session_id"] == "sess-1"
        assert response.metadata["total_cost_usd"] == 0.0123

        argv = list(mock_exec.call_args.args)
        assert "--model=opus" in argv
        assert "--system-prompt=be terse" in argv

    @pytest.mark.asyncio
    async def test_prompt_via_stdin_not_argv(self):
        provider = ClaudeProvider()
        process = _process(stdout=SUCCESS_JSON)
        _, mock_exec = await _run(
            provider, LLMRequest(prompt="SECRET-PROMPT-TEXT"), process
        )

        assert "SECRET-PROMPT-TEXT" not in " ".join(mock_exec.call_args.args)
        process.communicate.assert_awaited_once_with(b"SECRET-PROMPT-TEXT")
        assert mock_exec.call_args.kwargs["stdin"] == asyncio.subprocess.PIPE

    @pytest.mark.asyncio
    async def test_cwd_is_fresh_empty_temp_dir_removed_afterwards(self):
        seen: list[str] = []

        async def fake_exec(*_argv, **kwargs):
            cwd = kwargs["cwd"]
            assert Path(cwd).is_dir()
            assert list(Path(cwd).iterdir()) == []
            seen.append(cwd)
            return _process(stdout=SUCCESS_JSON)

        provider = ClaudeProvider()
        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await provider.generate(LLMRequest(prompt="a"))
            await provider.generate(LLMRequest(prompt="b"))

        assert len(set(seen)) == 2
        assert all(not Path(path).exists() for path in seen)
        assert all(Path(path) != Path.cwd() for path in seen)

    @pytest.mark.asyncio
    async def test_claude_env_stripped_but_home_path_kept(self, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abc")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR_KEEP", "x")
        provider = ClaudeProvider()
        _, mock_exec = await _run(
            provider, LLMRequest(prompt="hi"), _process(stdout=SUCCESS_JSON)
        )

        env = mock_exec.call_args.kwargs["env"]
        assert "CLAUDECODE" not in env
        assert not [key for key in env if key.startswith("CLAUDE_CODE_")]
        assert env["CLAUDE_CONFIG_DIR_KEEP"] == "x"
        assert env["PATH"] == os.environ["PATH"]
        assert env.get("HOME") == os.environ.get("HOME")

    @pytest.mark.asyncio
    async def test_is_error_true_raises_provider_error(self):
        payload = json.dumps({"is_error": True, "result": "Something broke"})
        provider = ClaudeProvider(max_retries=1)
        with patch(
            "asyncio.create_subprocess_exec", return_value=_process(stdout=payload)
        ):
            with pytest.raises(ProviderError, match="Something broke"):
                await provider.generate(LLMRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_malformed_json_raises(self):
        provider = ClaudeProvider(max_retries=1)
        with patch(
            "asyncio.create_subprocess_exec",
            return_value=_process(stdout="not json {"),
        ):
            with pytest.raises(ProviderError, match="malformed JSON"):
                await provider.generate(LLMRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_non_zero_exit_raises_with_stderr(self):
        provider = ClaudeProvider(max_retries=1)
        with patch(
            "asyncio.create_subprocess_exec",
            return_value=_process(stderr="kaboom", returncode=2),
        ):
            with pytest.raises(ProviderError, match="kaboom"):
                await provider.generate(LLMRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_auth_failure_is_configuration_error_not_retried(self):
        payload = json.dumps({"is_error": True, "result": "Not logged in"})
        provider = ClaudeProvider(max_retries=3)
        process = _process(stdout=payload, returncode=1)
        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            with pytest.raises(ProviderError, match="authentication") as exc:
                await provider.generate(LLMRequest(prompt="hi"))
        # generate() wraps non-ProviderError exceptions; the classification
        # must still be a ConfigurationError underneath.
        assert isinstance(exc.value.__cause__, ConfigurationError)
        assert mock_exec.call_count == 1
        with pytest.raises(ConfigurationError, match="authentication"):
            ClaudeProvider._raise_for_error("Not logged in")

    @pytest.mark.asyncio
    async def test_rate_limit_maps_to_rate_limit_error(self):
        payload = json.dumps({"is_error": True, "result": "Usage limit reached"})
        provider = ClaudeProvider(max_retries=1)
        with patch(
            "asyncio.create_subprocess_exec",
            return_value=_process(stdout=payload, returncode=1),
        ):
            with pytest.raises(RateLimitError):
                await provider.generate(LLMRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_json_array_result_entry_is_used(self):
        stream = json.dumps(
            [{"type": "system"}, {"type": "result", "is_error": False, "result": "hi"}]
        )
        response, _ = await _run(
            ClaudeProvider(), LLMRequest(prompt="x"), _process(stdout=stream)
        )
        assert response.content == "hi"

    @pytest.mark.asyncio
    async def test_missing_usage_falls_back_to_estimate(self):
        payload = json.dumps({"is_error": False, "result": "abcdefgh"})
        response, _ = await _run(
            ClaudeProvider(), LLMRequest(prompt="abcd"), _process(stdout=payload)
        )
        assert response.tokens_used == 3

    @pytest.mark.asyncio
    async def test_binary_vanishes_between_check_and_exec(self):
        provider = ClaudeProvider(max_retries=1)
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            with pytest.raises(ProviderError) as exc:
                await provider.generate(LLMRequest(prompt="hi"))
        assert exc.value.error_code == "PROVIDER_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_missing_binary_does_not_consume_quota(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CLI_PATH", "/nonexistent/claude")
        provider = ClaudeProvider(daily_quota=1)
        with pytest.raises(ProviderError):
            await provider.generate(LLMRequest(prompt="hi"))
        assert provider.quota_manager.get_current_usage() == 0


class TestArgvInjection:
    @pytest.mark.parametrize(
        "bad", ["--dangerously-skip-permissions", "-x", "a b", "a;rm", ""]
    )
    def test_bad_model_rejected(self, bad):
        provider = ClaudeProvider()
        if bad:
            with pytest.raises(ValueError, match="Invalid claude model"):
                provider._build_command(model=bad)

    @pytest.mark.asyncio
    async def test_bad_model_rejected_via_generate_without_spawn(self):
        provider = ClaudeProvider(daily_quota=1)
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            with pytest.raises(ProviderError, match="Invalid claude model"):
                await provider.generate(
                    LLMRequest(prompt="hi", model="--dangerously-skip-permissions")
                )
        mock_exec.assert_not_called()
        assert provider.quota_manager.get_current_usage() == 0

    @pytest.mark.parametrize("good", ["sonnet", "claude-opus-4-6", "x.y_z:1[1m]"])
    def test_good_model_accepted(self, good):
        cmd = ClaudeProvider()._build_command(model=good)
        assert f"--model={good}" in cmd

    def test_system_prompt_starting_with_dash_is_one_argv_element(self):
        cmd = ClaudeProvider()._build_command(system_prompt="--bare -p evil")
        assert "--system-prompt=--bare -p evil" in cmd
        assert "--bare" not in cmd


class TestDefaultProviderExplicitOnly:
    @pytest.mark.parametrize("hint", ["claude", "anthropic"])
    def test_default_hint_never_selects_claude(self, hint):
        registry = _StubRegistry(
            {
                "claude": _instance(ProviderType.ANTHROPIC),
                "llama": _instance(ProviderType.LLAMA),
            }
        )
        router = ProviderRouter(registry)  # type: ignore[arg-type]
        decision = router.route_request(
            LLMRequest(prompt="Hi", metadata={"preferred_provider": hint})
        )
        assert decision.selected_provider == ProviderType.LLAMA

    @pytest.mark.parametrize("name", ["claude", "anthropic"])
    def test_config_warns_when_default_is_claude(self, tmp_path, monkeypatch, name):
        from mcp_server_llm_cli_runner.utils import config as config_mod
        from mcp_server_llm_cli_runner.utils.config import ConfigManager

        monkeypatch.setenv("MCP_DEFAULT_PROVIDER", name)
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({}))
        with patch.object(config_mod, "logger") as mock_logger:
            ConfigManager(str(config_file)).load_configuration()
        warnings = [c.args[0] for c in mock_logger.warning.call_args_list]
        assert warnings.count("claude is explicit-only; default ignored") == 1


class TestTimeoutAndCancellation:
    @pytest.mark.asyncio
    async def test_temp_dir_removed_after_timeout(self):
        seen: list[str] = []
        process = _process()

        async def hang(_input=None):
            await asyncio.sleep(30)

        process.communicate.side_effect = hang

        async def fake_exec(*_argv, **kwargs):
            seen.append(kwargs["cwd"])
            return process

        provider = ClaudeProvider(max_retries=1)
        provider.timeout = 0.05
        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            with pytest.raises(ProviderError):
                await provider.generate(LLMRequest(prompt="hi"))
        assert len(seen) == 1
        assert not Path(seen[0]).exists()
        process.kill.assert_called()

    @pytest.mark.asyncio
    async def test_timeout_kills_reaps_and_releases_quota(self):
        process = _process()

        async def hang(_input=None):
            await asyncio.sleep(30)

        process.communicate.side_effect = hang
        config = ProviderConfig(
            name="claude", provider_type=ProviderType.ANTHROPIC, timeout=1
        )
        provider = ClaudeProvider(config=config, max_retries=1, daily_quota=5)
        provider.timeout = 0.05

        with patch("asyncio.create_subprocess_exec", return_value=process):
            with pytest.raises(ProviderError):
                await provider.generate(LLMRequest(prompt="hi"))

        process.kill.assert_called()
        process.wait.assert_awaited()
        assert provider.quota_manager.get_current_usage() == 0

    @pytest.mark.asyncio
    async def test_timeout_is_not_retried(self):
        process = _process()

        async def hang(_input=None):
            await asyncio.sleep(30)

        process.communicate.side_effect = hang
        provider = ClaudeProvider(max_retries=3, retry_delay=0)
        provider.timeout = 0.05
        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            with pytest.raises(ProviderError, match="timed out"):
                await provider.generate(LLMRequest(prompt="hi"))
        assert mock_exec.call_count == 1

    @pytest.mark.asyncio
    async def test_cli_reported_overloaded_error_not_retried(self):
        # The request may already have been accepted/billed: never resubmit.
        bad = json.dumps({"is_error": True, "result": "API Error: 529 overloaded"})
        procs = [_process(stdout=bad, returncode=1), _process(stdout=SUCCESS_JSON)]
        provider = ClaudeProvider(max_retries=3, retry_delay=0)
        with patch("asyncio.create_subprocess_exec", side_effect=procs) as mock_exec:
            with pytest.raises(ProviderError, match="overloaded"):
                await provider.generate(LLMRequest(prompt="hi"))
        assert mock_exec.call_count == 1

    @pytest.mark.asyncio
    async def test_cancel_during_spawn_kills_late_child(self):
        process = _process()
        gate = asyncio.Event()

        async def slow_spawn(*_args, **_kwargs):
            await gate.wait()
            return process

        provider = ClaudeProvider(daily_quota=5)
        with patch("asyncio.create_subprocess_exec", side_effect=slow_spawn):
            task = asyncio.create_task(provider.generate(LLMRequest(prompt="hi")))
            await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.sleep(0.05)
            gate.set()  # the OS "started" the child after cancellation
            with pytest.raises(asyncio.CancelledError):
                await task
        process.kill.assert_called()
        process.wait.assert_awaited()
        assert provider.quota_manager.get_current_usage() == 0

    @pytest.mark.asyncio
    async def test_success_commits_quota_and_updates_usage(self):
        provider = ClaudeProvider(daily_quota=5)
        await _run(provider, LLMRequest(prompt="hi"), _process(stdout=SUCCESS_JSON))
        assert provider.quota_manager.get_current_usage() == 1
        assert provider.quota_manager._outstanding == {}
        stats = (await provider.get_usage()).model_copy()
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.total_tokens == 15
        provider.get_usage_stats()
        assert provider._usage_stats.total_requests == 1  # reads do not inflate

    @pytest.mark.asyncio
    async def test_explicit_zero_temperature_preserved(self):
        provider = ClaudeProvider()
        response, _ = await _run(
            provider,
            LLMRequest(prompt="hi", temperature=0.0),
            _process(stdout=SUCCESS_JSON),
        )
        assert response.metadata["temperature"] == 0.0
        assert response.metadata["sampling_params_enforced"] is False

    def test_streaming_not_advertised(self):
        from mcp_server_llm_cli_runner.providers.base import ProviderCapabilities

        assert ProviderCapabilities.STREAMING not in ClaudeProvider().capabilities

    def test_child_env_strips_anthropic_and_session_vars(self, monkeypatch):
        for key in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "CLAUDECODE",
            "CLAUDE_CODE_ENTRYPOINT",
        ):
            monkeypatch.setenv(key, "x")
        env = ClaudeProvider._child_env()
        assert "PATH" in env
        assert not [k for k in env if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))]
        assert "CLAUDECODE" not in env

    @pytest.mark.asyncio
    async def test_spawn_failure_retries(self):
        provider = ClaudeProvider(max_retries=3, retry_delay=0)
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[OSError("EAGAIN"), _process(stdout=SUCCESS_JSON)],
        ) as mock_exec:
            response = await provider.generate(LLMRequest(prompt="hi"))
        assert response.content == "OK"
        assert mock_exec.call_count == 2

    @pytest.mark.asyncio
    async def test_non_transient_error_not_retried(self):
        provider = ClaudeProvider(max_retries=3, retry_delay=0)
        with patch(
            "asyncio.create_subprocess_exec",
            return_value=_process(stderr="kaboom", returncode=2),
        ) as mock_exec:
            with pytest.raises(ProviderError):
                await provider.generate(LLMRequest(prompt="hi"))
        assert mock_exec.call_count == 1

    @pytest.mark.asyncio
    async def test_cancellation_kills_reaps_and_releases_quota(self):
        process = _process()
        started = asyncio.Event()

        async def hang(_input=None):
            started.set()
            await asyncio.sleep(30)

        process.communicate.side_effect = hang
        provider = ClaudeProvider(daily_quota=5)

        with patch("asyncio.create_subprocess_exec", return_value=process):
            task = asyncio.create_task(provider.generate(LLMRequest(prompt="hi")))
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        process.kill.assert_called()
        process.wait.assert_awaited()
        assert provider.quota_manager.get_current_usage() == 0

    @pytest.mark.asyncio
    async def test_quota_exhausted_raises_rate_limit(self):
        provider = ClaudeProvider(daily_quota=0)
        with pytest.raises(RateLimitError):
            await provider.generate(LLMRequest(prompt="hi"))


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_check_version(self):
        provider = ClaudeProvider()
        with patch(
            "asyncio.create_subprocess_exec",
            return_value=_process(stdout="2.1.323 (Claude Code)\n"),
        ) as mock_exec:
            result = await provider.health_check()
        assert mock_exec.call_args.args == (provider._cli_binary(), "--version")
        assert result["status"] == "healthy"
        assert result["version"].startswith("2.1.323")

    @pytest.mark.asyncio
    async def test_health_check_missing_binary(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CLI_PATH", "/nonexistent/claude")
        result = await ClaudeProvider().health_check()
        assert result["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_is_available_missing_binary(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CLI_PATH", "/nonexistent/claude")
        assert await ClaudeProvider().is_available() is False

    @pytest.mark.asyncio
    async def test_is_available_true_and_timeout_kills(self):
        provider = ClaudeProvider()
        with patch(
            "asyncio.create_subprocess_exec", return_value=_process(returncode=0)
        ) as mock_exec:
            assert await provider.is_available() is True
        assert mock_exec.call_args.args == (provider._cli_binary(), "--version")

        hung = _process()

        waits = 0

        async def hang():
            nonlocal waits
            waits += 1
            if waits == 1:  # the probe wait hangs; the post-kill reap returns
                await asyncio.sleep(30)

        hung.wait.side_effect = hang
        provider.PROBE_TIMEOUT = 0.05
        with patch("asyncio.create_subprocess_exec", return_value=hung):
            assert await provider.is_available() is False
        hung.kill.assert_called()


class TestNamingAndConfig:
    def test_anthropic_is_a_non_deprecated_synonym(self):
        assert resolve_provider_name("anthropic") == ("claude", False)
        assert resolve_provider_name("claude") == ("claude", False)
        assert resolve_provider_name("gemini") == ("agy", True)

    def test_provider_type_accepts_claude(self):
        assert ProviderType("claude") is ProviderType.ANTHROPIC
        assert ClaudeProvider.PROVIDER_TYPE is ProviderType.ANTHROPIC

    @pytest.mark.parametrize("name", ["claude", "anthropic"])
    def test_dict_form_config_accepts_name_without_api_key(
        self, tmp_path, monkeypatch, name
    ):
        from mcp_server_llm_cli_runner.utils.config import ConfigManager

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "x" * 40)
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({"providers": {name: {"enabled": True, "api_key": "junk"}}})
        )
        manager = ConfigManager(str(config_file))
        manager.load_configuration()
        provider = manager.get_provider_config(name)
        assert provider is not None
        assert provider["provider_type"] == ProviderType.ANTHROPIC
        assert provider["model_name"] == "sonnet"
        assert provider["api_key"] is None

    @pytest.mark.parametrize("ptype", ["claude", "anthropic"])
    def test_list_form_config_accepts_type(self, tmp_path, ptype):
        from mcp_server_llm_cli_runner.utils.config import ConfigManager

        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({"providers": [{"name": "claude", "provider_type": ptype}]})
        )
        manager = ConfigManager(str(config_file))
        manager.load_configuration()
        assert manager.get_provider_config("claude") is not None

    def test_list_form_config_drops_api_key(self, tmp_path):
        from mcp_server_llm_cli_runner.utils.config import ConfigManager

        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "providers": [
                        {
                            "name": "claude",
                            "provider_type": "claude",
                            "api_key": "sk-ant-api03-" + "x" * 40,
                        }
                    ]
                }
            )
        )
        manager = ConfigManager(str(config_file))
        manager.load_configuration()
        assert manager.get_provider_config("claude")["api_key"] is None

    def test_config_without_default_provider_loads(self, tmp_path):
        from mcp_server_llm_cli_runner.utils.config import ConfigManager

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"providers": {"claude": {}}}))
        manager = ConfigManager(str(config_file))
        manager.load_configuration()
        assert manager.get_provider_config("claude") is not None

    def test_unknown_name_error_lists_claude(self, tmp_path):
        from mcp_server_llm_cli_runner.utils.config import ConfigManager

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"providers": {"bogus": {}}}))
        with pytest.raises(ConfigurationError, match="claude"):
            ConfigManager(str(config_file)).load_configuration()


class _StubRegistry:
    def __init__(self, instances: dict[str, Mock]) -> None:
        self._instances = instances

    def list_available_types(self) -> list[ProviderType]:
        return [inst.provider_type for inst in self._instances.values()]

    def list_providers(self) -> list[str]:
        return list(self._instances)

    def get_provider(self, name: str) -> Mock | None:
        return self._instances.get(name)


def _instance(provider_type: ProviderType) -> Mock:
    inst = Mock()
    inst.provider_type = provider_type
    inst.is_healthy = True
    return inst


class TestRoutingExplicitOnly:
    def test_claude_absent_from_every_rule_and_fallback(self):
        for level in ComplexityLevel:
            assert (
                ProviderType.ANTHROPIC
                not in ProviderRouter.ROUTING_RULES[level]["primary"]
            )

        router = ProviderRouter(_StubRegistry({}))  # type: ignore[arg-type]
        for primary in ProviderType:
            assert ProviderType.ANTHROPIC not in router._get_fallback_providers(primary)

    @pytest.mark.parametrize(
        "prompt",
        [
            "Hi",
            "Compare two sorting algorithms and explain the trade-offs in detail.",
            "Design a distributed consensus algorithm with Byzantine fault "
            "tolerance. Analyze the performance characteristics of Raft vs PBFT. "
            "Implement optimization strategies using advanced data structures. "
            "Provide mathematical proofs of correctness and performance analysis. "
            "Research the theoretical framework and methodology for consensus.",
        ],
    )
    def test_never_auto_selected_even_when_only_claude_is_available(self, prompt):
        registry = _StubRegistry({"claude": _instance(ProviderType.ANTHROPIC)})
        router = ProviderRouter(registry)  # type: ignore[arg-type]
        with pytest.raises(ProviderError) as exc:
            router.route_request(LLMRequest(prompt=prompt))
        assert exc.value.error_code == "NO_PROVIDERS_AVAILABLE"

    def test_auto_selection_prefers_others_when_claude_is_registered(self):
        registry = _StubRegistry(
            {
                "claude": _instance(ProviderType.ANTHROPIC),
                "llama": _instance(ProviderType.LLAMA),
            }
        )
        router = ProviderRouter(registry)  # type: ignore[arg-type]
        decision = router.route_request(LLMRequest(prompt="Hi"))
        assert decision.selected_provider == ProviderType.LLAMA
        assert ProviderType.ANTHROPIC not in decision.fallback_providers

    @pytest.mark.parametrize("name", ["claude", "anthropic", "Claude"])
    def test_explicit_request_reaches_claude(self, name):
        registry = _StubRegistry({"claude": _instance(ProviderType.ANTHROPIC)})
        router = ProviderRouter(registry)  # type: ignore[arg-type]
        decision = router.route_request(LLMRequest(prompt="Hi", provider=name))
        assert decision.selected_provider == ProviderType.ANTHROPIC
        assert ProviderType.ANTHROPIC not in decision.fallback_providers

    def test_explicit_claude_unavailable_does_not_fall_back(self):
        registry = _StubRegistry({"llama": _instance(ProviderType.LLAMA)})
        router = ProviderRouter(registry)  # type: ignore[arg-type]
        with pytest.raises(ProviderError) as exc:
            router.route_request(LLMRequest(prompt="Hi", provider="claude"))
        assert exc.value.error_code == "PROVIDER_UNAVAILABLE"


@pytest.mark.skipif(
    os.environ.get("LLM_CLI_RUNNER_LIVE_CLAUDE") != "1",
    reason="live claude smoke test; set LLM_CLI_RUNNER_LIVE_CLAUDE=1",
)
@pytest.mark.asyncio
async def test_live_claude_smoke(monkeypatch):
    monkeypatch.delenv("CLAUDE_CLI_PATH", raising=False)
    provider = ClaudeProvider(max_retries=1)
    response = await provider.generate(
        LLMRequest(prompt="Reply with exactly the word: pong", max_tokens=50)
    )
    assert "pong" in response.content.lower()
