"""
Unit tests for Google Gemini Provider

Tests the Gemini provider implementation including quota management,
CLI interaction, retry logic, and streaming responses.

Note: These tests are skipped in CI when GEMINI_API_KEY is not set.
"""

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.mcp_server_cheap_llm.core.errors import ProviderError, RateLimitError
from src.mcp_server_cheap_llm.core.models import (
    LLMRequest,
    LLMResponse,
    StreamingResponse,
)

# Skip entire module if Gemini credentials are not available
# This prevents CI failures when API keys are not configured
GEMINI_AVAILABLE = bool(
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
    or os.environ.get("GOOGLE_GENAI_USE_GCA")
    or os.path.exists(os.path.expanduser("~/.gemini/settings.json"))
)

if not GEMINI_AVAILABLE:
    pytest.skip(
        "Gemini credentials not available (set GEMINI_API_KEY or configure ~/.gemini/settings.json)",
        allow_module_level=True,
    )

try:
    from src.mcp_server_cheap_llm.providers.gemini import (
        GeminiProvider,
        GeminiQuotaManager,
    )
except ImportError:
    pytest.skip("google-generativeai dependency not available", allow_module_level=True)


class TestGeminiQuotaManager:
    """Test Gemini quota management functionality"""

    def test_quota_manager_initialization(self):
        """Test quota manager initializes with correct defaults"""
        manager = GeminiQuotaManager()

        assert manager.daily_limit == 1000
        assert manager.requests_today == 0
        assert manager.last_reset == datetime.now().date()

    def test_quota_manager_custom_limit(self):
        """Test quota manager with custom daily limit"""
        manager = GeminiQuotaManager(daily_limit=500)

        assert manager.daily_limit == 500
        assert manager.get_remaining_quota() == 500

    def test_quota_consumption(self):
        """Test quota consumption tracking"""
        manager = GeminiQuotaManager(daily_limit=10)

        # Initial state
        assert manager.check_quota() is True
        assert manager.get_remaining_quota() == 10

        # Consume some quota
        manager.consume_quota()
        assert manager.requests_today == 1
        assert manager.get_remaining_quota() == 9

        # Consume all quota
        for _ in range(9):
            manager.consume_quota()

        assert manager.requests_today == 10
        assert manager.get_remaining_quota() == 0
        assert manager.check_quota() is False

    def test_quota_warning_threshold(self):
        """Test quota warning at 80% usage"""
        manager = GeminiQuotaManager(daily_limit=10)

        # Consume to 80% (8 requests)
        for _ in range(8):
            manager.consume_quota()

        # Verify quota state that would trigger warning
        assert manager.requests_today == 8
        assert manager.requests_today >= (manager.daily_limit * 0.8)
        assert manager.get_remaining_quota() == 2

    def test_quota_reset_new_day(self):
        """Test quota resets for new day"""
        manager = GeminiQuotaManager(daily_limit=10)

        # Consume some quota
        manager.consume_quota()
        assert manager.requests_today == 1

        # Simulate new day
        manager.last_reset = datetime.now().date() - timedelta(days=1)

        # Check quota (should reset)
        assert manager.check_quota() is True
        assert manager.requests_today == 0
        assert manager.get_remaining_quota() == 10


class TestGeminiProvider:
    """Test Gemini provider functionality"""

    def test_provider_initialization(self):
        """Test provider initializes with correct defaults"""
        provider = GeminiProvider()

        assert provider.name == "gemini"
        assert provider.model == "gemini-1.5-flash"  # Default model from implementation
        assert provider.temperature == 0.7
        assert provider.max_tokens == 4096
        assert isinstance(provider.quota_manager, GeminiQuotaManager)
        assert provider.config.provider_type.value == "gemini"

    def test_provider_custom_parameters(self):
        """Test provider with custom parameters"""
        provider = GeminiProvider(
            model="gemini-pro-vision",
            temperature=0.5,
        )

        assert provider.model == "gemini-pro-vision"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 4096  # Default value
        assert isinstance(provider.quota_manager, GeminiQuotaManager)

    @pytest.mark.asyncio
    async def test_is_available_cli_not_found(self):
        """Test availability check when CLI not found"""
        provider = GeminiProvider()

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock process that returns non-zero exit code
            mock_process = AsyncMock()
            mock_process.wait.return_value = 1
            mock_subprocess.return_value = mock_process

            available = await provider.is_available()
            assert available is False

    @pytest.mark.asyncio
    async def test_is_available_quota_exhausted(self):
        """Test availability check when quota exhausted"""
        provider = GeminiProvider()

        # Exhaust quota
        provider.quota_manager.requests_today = 1000

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock successful CLI check
            mock_process = AsyncMock()
            mock_process.wait.return_value = 0
            mock_subprocess.return_value = mock_process

            available = await provider.is_available()
            assert available is False

    @pytest.mark.asyncio
    async def test_is_available_success(self):
        """Test availability check when everything is available"""
        provider = GeminiProvider()

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock successful CLI check
            mock_process = AsyncMock()
            mock_process.wait.return_value = 0
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            available = await provider.is_available()
            assert available is True

    def test_build_cli_command(self):
        """Test CLI command building"""
        provider = GeminiProvider(
            model="gemini-pro",
            temperature=0.8,
        )

        cmd = provider._build_command(
            prompt="Test prompt",
            model="gemini-pro",
            temperature=0.8,
        )

        expected_parts = [
            "gemini",
            "--model",
            "gemini-pro",
            "--temperature",
            "0.8",
            "--format",
            "json",
            "Test prompt",
        ]

        for part in expected_parts:
            assert part in cmd

    @pytest.mark.asyncio
    async def test_generate_response_quota_exhausted(self):
        """Test generate response when quota exhausted"""
        provider = GeminiProvider()
        provider.quota_manager.requests_today = 1000  # Exhaust quota

        request = LLMRequest(prompt="Test prompt")

        with pytest.raises(RateLimitError) as exc_info:
            await provider.generate(request)

        assert "quota exhausted" in str(exc_info.value).lower()
        assert exc_info.value.provider == "gemini"

    @pytest.mark.asyncio
    async def test_generate_response_success(self):
        """Test successful response generation"""
        provider = GeminiProvider()
        request = LLMRequest(prompt="Test prompt")

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock successful CLI execution
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (
                b"Generated response from Gemini",
                b"",
            )
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            response = await provider.generate(request)

            assert isinstance(response, LLMResponse)
            assert response.content == "Generated response from Gemini"
            assert response.provider == "gemini" or str(response.provider) == "gemini"
            assert (
                "model" in response.metadata
                and response.metadata["model"] == "gemini-1.5-flash"
            )
            assert "usage" in response.metadata or hasattr(response, "usage")

    @pytest.mark.asyncio
    async def test_generate_response_retry_logic(self):
        """Test retry logic on CLI failures"""
        provider = GeminiProvider(max_retries=2, retry_delay=0.1)
        request = LLMRequest(prompt="Test prompt")

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock process that fails first time, succeeds second time
            call_count = 0

            async def mock_create_process(*args, **kwargs):
                nonlocal call_count
                call_count += 1

                mock_process = AsyncMock()
                if call_count == 1:
                    # First call fails
                    mock_process.communicate.return_value = (b"", b"API Error")
                    mock_process.returncode = 1
                else:
                    # Second call succeeds
                    mock_process.communicate.return_value = (b"Success", b"")
                    mock_process.returncode = 0

                return mock_process

            mock_subprocess.side_effect = mock_create_process

            response = await provider.generate(request)

            assert response.content == "Success"
            assert call_count == 2  # Should have retried once

    @pytest.mark.asyncio
    async def test_generate_response_max_retries_exceeded(self):
        """Test failure after max retries exceeded"""
        provider = GeminiProvider(max_retries=2, retry_delay=0.1)
        request = LLMRequest(prompt="Test prompt")

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock process that always fails
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"Persistent error")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(request)

            assert "failed after 2 attempts" in str(exc_info.value)
            assert exc_info.value.provider == "gemini"

    @pytest.mark.asyncio
    async def test_generate_streaming_response(self):
        """Test streaming response generation"""
        provider = GeminiProvider()
        request = LLMRequest(prompt="Test prompt")

        with patch.object(provider, "generate") as mock_generate:
            # Mock regular response
            mock_generate.return_value = LLMResponse(
                content="This is a test response with multiple words",
                provider="gemini",
                model="gemini-pro",
            )

            # Collect streaming chunks
            chunks = []
            async for chunk in provider.generate_streaming_response(request):
                chunks.append(chunk)

            assert len(chunks) > 1  # Should be split into multiple chunks
            assert all(isinstance(chunk, StreamingResponse) for chunk in chunks)
            assert chunks[-1].is_final is True  # Last chunk should be final

            # Reconstruct content
            full_content = " ".join(chunk.content for chunk in chunks)
            assert "This is a test response with multiple words" in full_content

    def test_get_health_status(self):
        """Test health status reporting"""
        provider = GeminiProvider(model="gemini-pro", temperature=0.8, daily_quota=500)

        with patch.object(provider, "is_available", return_value=True):
            status = provider.get_health_status()

            assert status["provider"] == "gemini"
            assert status["model"] == "gemini-pro"
            assert "quota_remaining" in status
            assert "quota_limit" in status
            assert status["quota_limit"] == 500
            assert "configuration" in status
            assert status["configuration"]["temperature"] == 0.8


@pytest.mark.integration
class TestGeminiProviderIntegration:
    """Integration tests for Gemini provider (require actual CLI)"""

    @pytest.mark.asyncio
    async def test_real_availability_check(self):
        """Test real availability check (requires gemini CLI)"""
        provider = GeminiProvider()

        # This will actually check if gemini CLI is installed
        # Should not fail even if CLI is not available
        try:
            available = await provider.is_available()
            # Test passes regardless of availability
            assert isinstance(available, bool)
        except Exception as e:
            pytest.fail(f"Availability check should not raise exception: {e}")
