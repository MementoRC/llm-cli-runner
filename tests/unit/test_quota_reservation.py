"""Unit tests for atomic quota reservation (check-then-act race, issue #20).

These tests are fully mocked and do not need Gemini credentials, so unlike
test_gemini_provider.py they are not skipped in CI.
"""

import asyncio
import threading
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_server_llm_cli_runner.core.errors import ProviderError, RateLimitError
from mcp_server_llm_cli_runner.core.models import LLMRequest, LLMResponse
from mcp_server_llm_cli_runner.providers.agy import AgyProvider
from mcp_server_llm_cli_runner.providers.gemini import (
    GeminiProvider,
    GeminiQuotaManager,
)

DATETIME_TARGET = "mcp_server_llm_cli_runner.providers.gemini.datetime"
RESULT_LINE = b'{"type": "result", "response": "done"}\n'


def _hanging_process(started: asyncio.Event) -> Mock:
    """Mock subprocess whose communicate() never completes."""

    async def communicate():
        started.set()
        await asyncio.Event().wait()

    process = Mock()
    process.communicate = AsyncMock(side_effect=communicate)
    process.kill = Mock()
    process.terminate = Mock()
    process.wait = AsyncMock(return_value=0)
    process.returncode = None
    return process


class _Stdout:
    """Async-iterable stdout mock yielding lines, then optionally blocking."""

    def __init__(self, lines: list[bytes], block: bool = False) -> None:
        self._lines = list(lines)
        self._block = block

    def __aiter__(self) -> "_Stdout":
        return self

    async def __anext__(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        if self._block:
            await asyncio.Event().wait()
        raise StopAsyncIteration


class TestQuotaManagerReservation:
    def test_reserve_up_to_limit_then_none(self):
        manager = GeminiQuotaManager(daily_limit=3)

        tokens = [manager.reserve() for _ in range(3)]

        assert all(token is not None for token in tokens)
        assert len(set(tokens)) == 3
        assert manager.requests_today == 3
        assert manager.reserve() is None
        assert manager.requests_today == 3

    def test_release_decrements(self):
        manager = GeminiQuotaManager(daily_limit=3)
        token = manager.reserve()
        assert token is not None

        manager.release(token)

        assert manager.requests_today == 0

    def test_repeated_release_does_not_free_other_slot(self):
        manager = GeminiQuotaManager(daily_limit=3)
        first = manager.reserve()
        second = manager.reserve()
        assert first is not None
        assert second is not None

        manager.release(first)
        manager.release(first)

        assert manager.requests_today == 1
        manager.release(second)
        assert manager.requests_today == 0

    def test_release_unknown_token_is_noop(self):
        manager = GeminiQuotaManager(daily_limit=3)
        assert manager.reserve() is not None

        manager.release(9999)

        assert manager.requests_today == 1

    def test_release_after_daily_reset_is_noop(self):
        day1 = datetime.now().date()
        day2 = day1 + timedelta(days=1)

        with patch(DATETIME_TARGET) as mock_dt:
            mock_dt.now.return_value.date.return_value = day1
            manager = GeminiQuotaManager(daily_limit=3)
            token = manager.reserve()
            assert token is not None

            mock_dt.now.return_value.date.return_value = day2
            assert manager.reserve() is not None
            assert manager.requests_today == 1

            manager.release(token)
            assert manager.requests_today == 1

    def test_reserve_rejects_non_positive_amount(self):
        manager = GeminiQuotaManager(daily_limit=3)

        with pytest.raises(ValueError):
            manager.reserve(0)
        with pytest.raises(ValueError):
            manager.reserve(-1)

        assert manager.requests_today == 0

    def test_reserve_amount_released_in_full(self):
        manager = GeminiQuotaManager(daily_limit=5)
        token = manager.reserve(3)
        assert token is not None
        assert manager.requests_today == 3

        manager.release(token)

        assert manager.requests_today == 0

    def test_consume_quota_still_works(self):
        manager = GeminiQuotaManager(daily_limit=1)

        assert manager.consume_quota() is True
        assert manager.consume_quota() is False
        assert manager.requests_today == 1

    @pytest.mark.parametrize("getter", ["get_remaining_quota", "get_current_usage"])
    def test_getters_hold_lock(self, getter):
        manager = GeminiQuotaManager(daily_limit=3)
        done = threading.Event()

        def call() -> None:
            getattr(manager, getter)()
            done.set()

        with manager._lock:
            thread = threading.Thread(target=call)
            thread.start()
            assert not done.wait(0.1)
        thread.join(timeout=2)
        assert done.is_set()

    def test_getter_reset_clears_outstanding(self):
        day1 = datetime.now().date()
        with patch(DATETIME_TARGET) as mock_dt:
            mock_dt.now.return_value.date.return_value = day1
            manager = GeminiQuotaManager(daily_limit=3)
            token = manager.reserve()
            assert token is not None

            mock_dt.now.return_value.date.return_value = day1 + timedelta(days=1)
            assert manager.get_current_usage() == 0

            manager.release(token)
            assert manager.get_current_usage() == 0


@pytest.mark.parametrize("provider_cls", [GeminiProvider, AgyProvider])
class TestGenerateReservation:
    @pytest.mark.asyncio
    async def test_concurrent_generate_respects_quota(self, provider_cls):
        quota, extra = 3, 4
        provider = provider_cls(daily_quota=quota)
        gate = asyncio.Event()

        async def stub(cmd, request):
            await gate.wait()
            return {"response": "ok"}

        with patch.object(provider, "_execute_with_retries", side_effect=stub):
            tasks = [
                asyncio.ensure_future(provider.generate(LLMRequest(prompt="hi")))
                for _ in range(quota + extra)
            ]
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            gate.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)

        assert sum(isinstance(r, LLMResponse) for r in results) == quota
        assert sum(isinstance(r, RateLimitError) for r in results) == extra
        assert provider.quota_manager.requests_today == quota

    @pytest.mark.asyncio
    async def test_failed_generate_releases_quota(self, provider_cls):
        provider = provider_cls(daily_quota=2)

        async def stub(cmd, request):
            assert provider.quota_manager.requests_today == 1
            raise ProviderError("boom", provider=provider.name)

        with patch.object(provider, "_execute_with_retries", side_effect=stub):
            with pytest.raises(ProviderError):
                await provider.generate(LLMRequest(prompt="hi"))

        assert provider.quota_manager.requests_today == 0

    @pytest.mark.asyncio
    async def test_cancelled_generate_releases_quota(self, provider_cls):
        provider = provider_cls(daily_quota=2)
        started = asyncio.Event()

        async def stub(cmd, request):
            assert provider.quota_manager.requests_today == 1
            started.set()
            await asyncio.Event().wait()

        with patch.object(provider, "_execute_with_retries", side_effect=stub):
            task = asyncio.ensure_future(provider.generate(LLMRequest(prompt="hi")))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert provider.quota_manager.requests_today == 0

    @pytest.mark.asyncio
    async def test_validation_failure_does_not_reserve(self, provider_cls):
        provider = provider_cls(daily_quota=2)

        with pytest.raises(ProviderError):
            await provider.generate(LLMRequest(prompt=" "))

        assert provider.quota_manager.requests_today == 0

    @pytest.mark.asyncio
    async def test_cancel_kills_and_reaps_child_then_releases(self, provider_cls):
        provider = provider_cls(daily_quota=2)
        started = asyncio.Event()
        process = _hanging_process(started)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            task = asyncio.ensure_future(provider.generate(LLMRequest(prompt="hi")))
            await started.wait()
            assert provider.quota_manager.requests_today == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        process.kill.assert_called_once()
        process.wait.assert_awaited()
        assert provider.quota_manager.requests_today == 0

    @pytest.mark.asyncio
    async def test_second_cancel_during_reap_still_reaps(self, provider_cls):
        provider = provider_cls(daily_quota=2)
        started = asyncio.Event()
        process = _hanging_process(started)
        reap_gate = asyncio.Event()
        reaped = asyncio.Event()

        async def slow_wait():
            await reap_gate.wait()
            reaped.set()
            return 0

        process.wait = AsyncMock(side_effect=slow_wait)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            task = asyncio.ensure_future(provider.generate(LLMRequest(prompt="hi")))
            await started.wait()
            task.cancel()
            await asyncio.sleep(0.01)  # now inside the reap
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            reap_gate.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert reaped.is_set()
        assert provider.quota_manager.requests_today == 0


class TestStreamReservation:
    @pytest.mark.asyncio
    async def test_stream_quota_exhausted_raises_without_subprocess(self):
        provider = GeminiProvider(daily_quota=1)
        provider.quota_manager.requests_today = 1

        with patch("asyncio.create_subprocess_exec", AsyncMock()) as mock_exec:
            with pytest.raises(RateLimitError) as exc_info:
                await provider.stream_generate(LLMRequest(prompt="hi"))

        mock_exec.assert_not_called()
        assert "quota exhausted" in str(exc_info.value).lower()
        assert provider.quota_manager.requests_today == 1

    @pytest.mark.asyncio
    async def test_stream_success_keeps_reservation(self):
        provider = GeminiProvider(daily_quota=2)
        process = Mock()
        process.stdout = _Stdout([RESULT_LINE])
        process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            chunks = await provider.stream_generate(LLMRequest(prompt="hi"))

        assert chunks[-1].is_final is True
        assert provider.quota_manager.requests_today == 1

    @pytest.mark.asyncio
    async def test_stream_failure_releases_reservation(self):
        provider = GeminiProvider(daily_quota=2)
        process = Mock()
        process.stdout = None
        process.terminate = Mock()
        process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            with pytest.raises(ProviderError):
                await provider.stream_generate(LLMRequest(prompt="hi"))

        assert provider.quota_manager.requests_today == 0

    @pytest.mark.asyncio
    async def test_stream_cancel_kills_child_and_releases(self):
        provider = GeminiProvider(daily_quota=2)
        process = Mock()
        process.stdout = _Stdout([], block=True)
        process.kill = Mock()
        process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            task = asyncio.ensure_future(
                provider.stream_generate(LLMRequest(prompt="hi"))
            )
            await asyncio.sleep(0.05)
            assert provider.quota_manager.requests_today == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        process.kill.assert_called_once()
        process.wait.assert_awaited()
        assert provider.quota_manager.requests_today == 0

    @pytest.mark.asyncio
    async def test_agy_stream_reserves_once(self):
        provider = AgyProvider(daily_quota=5)
        seen: list[int] = []

        async def stub(cmd, request):
            seen.append(provider.quota_manager.requests_today)
            return {"response": "ok"}

        with patch.object(provider, "_execute_with_retries", side_effect=stub):
            chunks = await provider.stream_generate(LLMRequest(prompt="hi"))

        assert len(chunks) == 1
        assert seen == [1]
        assert provider.quota_manager.requests_today == 1
