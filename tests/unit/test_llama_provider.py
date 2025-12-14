"""Unit tests for LLaMA Provider implementation.

Tests cover model management, generation capabilities, health monitoring,
and integration with the provider interface.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mcp_server_cheap_llm.core.errors import ProviderError
from mcp_server_cheap_llm.core.models import LLMRequest, ProviderConfig, ProviderType

try:
    from mcp_server_cheap_llm.providers.llama import (
        LLAMA_CPP_AVAILABLE,
        LLaMAModelManager,
        LLaMAProvider,
    )
except ImportError:
    pytest.skip("llama-cpp-python dependency not available", allow_module_level=True)


class TestLLaMAModelManager:
    """Test LLaMA model manager functionality."""

    def test_model_manager_initialization(self):
        """Test model manager initialization with default parameters."""
        manager = LLaMAModelManager()

        assert manager.model_path is None
        assert manager.n_ctx == 2048
        assert manager.n_gpu_layers == -1
        assert manager.use_mmap is True
        assert manager.use_mlock is False
        assert manager.n_threads is not None
        assert manager.model is None
        assert manager.model_loaded is False

    def test_model_manager_custom_parameters(self):
        """Test model manager initialization with custom parameters."""
        manager = LLaMAModelManager(
            model_path="/test/model.gguf",
            n_ctx=4096,
            n_gpu_layers=10,
            use_mmap=False,
            use_mlock=True,
            n_threads=4,
        )

        assert manager.model_path == "/test/model.gguf"
        assert manager.n_ctx == 4096
        assert manager.n_gpu_layers == 10
        assert manager.use_mmap is False
        assert manager.use_mlock is True
        assert manager.n_threads == 4

    def test_is_model_available_no_path(self):
        """Test model availability check with no path."""
        manager = LLaMAModelManager()
        assert manager.is_model_available() is False

    def test_is_model_available_file_not_exists(self):
        """Test model availability check with non-existent file."""
        manager = LLaMAModelManager(model_path="/nonexistent/model.gguf")
        assert manager.is_model_available() is False

    def test_is_model_available_file_exists(self):
        """Test model availability check with existing file."""
        with tempfile.NamedTemporaryFile(suffix=".gguf") as tmp_file:
            manager = LLaMAModelManager(model_path=tmp_file.name)
            assert manager.is_model_available() is True

    @patch("mcp_server_cheap_llm.providers.llama.LLAMA_CPP_AVAILABLE", False)
    async def test_load_model_llama_cpp_unavailable(self):
        """Test model loading when llama-cpp-python is unavailable."""
        manager = LLaMAModelManager(model_path="/test/model.gguf")
        result = await manager.load_model()
        assert result is False
        assert manager.model_loaded is False

    async def test_load_model_file_not_available(self):
        """Test model loading with unavailable model file."""
        manager = LLaMAModelManager(model_path="/nonexistent/model.gguf")
        result = await manager.load_model()
        assert result is False
        assert manager.model_loaded is False

    @pytest.mark.skipif(
        not LLAMA_CPP_AVAILABLE,
        reason="llama-cpp-python not available",
    )
    @patch("mcp_server_cheap_llm.providers.llama.Llama", autospec=True)
    async def test_load_model_success(self, mock_llama_class):
        """Test successful model loading."""
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix=".gguf") as tmp_file:
            # Mock the Llama class
            mock_model = Mock()
            mock_llama_class.return_value = mock_model

            manager = LLaMAModelManager(model_path=tmp_file.name)

            # Mock GPU detection
            with patch.object(manager, "_detect_gpu_layers", return_value=0):
                result = await manager.load_model()

            assert result is True
            assert manager.model_loaded is True
            assert manager.model is mock_model
            assert manager.load_time > 0

    def test_detect_gpu_layers_explicit(self):
        """Test GPU layer detection with explicit setting."""
        manager = LLaMAModelManager(n_gpu_layers=10)
        layers = manager._detect_gpu_layers()
        assert layers == 10

    @patch("platform.system")
    @patch("platform.processor")
    def test_detect_gpu_layers_apple_silicon(self, mock_processor, mock_system):
        """Test GPU layer detection on Apple Silicon."""
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "arm"

        manager = LLaMAModelManager(n_gpu_layers=-1)
        layers = manager._detect_gpu_layers()
        assert layers == 999  # Metal acceleration

    def test_detect_gpu_layers_cpu_fallback(self):
        """Test GPU layer detection falling back to CPU."""
        manager = LLaMAModelManager(n_gpu_layers=-1)

        # Mock no GPU detection
        with patch(
            "mcp_server_cheap_llm.providers.llama.platform.system",
            return_value="linux",
        ):
            layers = manager._detect_gpu_layers()

        assert layers == 0  # CPU-only

    def test_get_model_info(self):
        """Test model information retrieval."""
        manager = LLaMAModelManager(
            model_path="/test/model.gguf",
            n_ctx=4096,
            n_gpu_layers=10,
            n_threads=8,
        )

        info = manager.get_model_info()

        assert info["model_path"] == "/test/model.gguf"
        assert info["loaded"] is False
        assert info["context_size"] == 4096
        assert info["gpu_layers"] == 10
        assert info["threads"] == 8


class TestLLaMAProvider:
    """Test LLaMA provider functionality."""

    def test_provider_initialization(self):
        """Test provider initialization with default configuration."""
        provider = LLaMAProvider()

        assert provider.config.name == "llama"
        assert provider.config.provider_type == ProviderType.LLAMA
        # Local models have no cost - this is handled in the provider logic, not config
        assert provider.model_name == "llama-local"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 512

    def test_provider_custom_configuration(self):
        """Test provider initialization with custom configuration."""
        config = ProviderConfig(
            name="custom-llama",
            provider_type=ProviderType.LLAMA,
            model_name="custom-model",
            enabled=True,
        )

        provider = LLaMAProvider(
            config=config,
            model_path="/custom/model.gguf",
            model_name="custom-llama-7b",
            temperature=0.9,
            max_tokens=1024,
        )

        assert provider.config.name == "custom-llama"
        assert provider.model_path == "/custom/model.gguf"
        assert provider.model_name == "custom-llama-7b"
        assert provider.temperature == 0.9
        assert provider.max_tokens == 1024

    def test_provider_capabilities(self):
        """Test provider capabilities."""
        provider = LLaMAProvider()
        capabilities = provider.capabilities

        assert "streaming" in capabilities
        assert "async_generation" in capabilities

    @patch.dict(os.environ, {"LLAMA_MODEL_PATH": "/env/model.gguf"})
    def test_provider_model_path_from_env(self):
        """Test model path configuration from environment variable."""
        provider = LLaMAProvider()
        assert provider.model_path == "/env/model.gguf"

    @patch("mcp_server_cheap_llm.providers.llama.LLAMA_CPP_AVAILABLE", False)
    async def test_is_available_llama_cpp_unavailable(self):
        """Test availability check when llama-cpp-python is unavailable."""
        provider = LLaMAProvider(model_path="/test/model.gguf")
        available = await provider.is_available()
        assert available is False

    async def test_is_available_no_model_path(self):
        """Test availability check with no model path."""
        provider = LLaMAProvider()
        available = await provider.is_available()
        assert available is False

    async def test_is_available_model_not_accessible(self):
        """Test availability check with inaccessible model."""
        provider = LLaMAProvider(model_path="/nonexistent/model.gguf")
        available = await provider.is_available()
        assert available is False

    @patch("mcp_server_cheap_llm.providers.llama.LLAMA_CPP_AVAILABLE", True)
    async def test_is_available_success(self):
        """Test successful availability check."""
        with tempfile.NamedTemporaryFile(suffix=".gguf") as tmp_file:
            provider = LLaMAProvider(model_path=tmp_file.name)

            # Mock successful model loading
            provider.model_manager.load_model = AsyncMock(return_value=True)
            provider.model_manager.model_loaded = True

            available = await provider.is_available()
            assert available is True

    async def test_generate_response_unavailable(self):
        """Test response generation when provider is unavailable."""
        provider = LLaMAProvider()
        provider.is_available = AsyncMock(return_value=False)

        request = LLMRequest(prompt="Test prompt")

        with pytest.raises(ProviderError, match="LLaMA provider not available"):
            await provider.generate(request)

    @patch("mcp_server_cheap_llm.providers.llama.LLAMA_CPP_AVAILABLE", True)
    async def test_generate_response_success(self):
        """Test successful response generation."""
        with tempfile.NamedTemporaryFile(suffix=".gguf") as tmp_file:
            provider = LLaMAProvider(model_path=tmp_file.name)

            # Mock model availability and generation
            provider.is_available = AsyncMock(return_value=True)

            mock_response = {
                "choices": [{"text": "Generated response", "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }

            provider._generate_sync = Mock(return_value=mock_response)

            request = LLMRequest(prompt="Test prompt", max_tokens=100, temperature=0.8)

            response = await provider.generate(request)

            assert response.content == "Generated response"
            assert response.metadata["model"] == "llama-local"
            assert response.provider == ProviderType.LLAMA
            assert response.metadata["prompt_tokens"] == 10
            assert response.metadata["completion_tokens"] == 20
            assert response.tokens_used == 30
            assert response.metadata["finish_reason"] == "stop"

    async def test_generate_response_error(self):
        """Test response generation with error."""
        provider = LLaMAProvider()
        provider.is_available = AsyncMock(return_value=True)
        provider._generate_sync = Mock(side_effect=Exception("Generation failed"))

        request = LLMRequest(prompt="Test prompt")

        with pytest.raises(ProviderError, match="Failed to generate response"):
            await provider.generate(request)

    async def test_generate_streaming_response_unavailable(self):
        """Test streaming response when provider is unavailable."""
        provider = LLaMAProvider()
        provider.is_available = AsyncMock(return_value=False)

        request = LLMRequest(prompt="Test prompt")

        with pytest.raises(ProviderError, match="LLaMA provider not available"):
            async for _ in provider.generate_streaming_response(request):
                pass

    async def test_get_health_status_healthy(self):
        """Test health status when provider is healthy."""
        provider = LLaMAProvider(model_path="/test/model.gguf")
        provider.is_available = AsyncMock(return_value=True)
        provider.model_manager.get_model_info = Mock(
            return_value={
                "model_path": "/test/model.gguf",
                "loaded": True,
                "load_time": 2.5,
                "memory_usage_mb": 1024.0,
                "gpu_enabled": True,
            },
        )

        health = await provider.get_health_status_async()

        assert health["healthy"] is True
        assert health["provider"] == "llama"
        assert health["model_path"] == "/test/model.gguf"
        assert "model_info" in health
        assert "performance_stats" in health
        assert "system_resources" in health

    async def test_get_health_status_unhealthy(self):
        """Test health status when provider is unhealthy."""
        provider = LLaMAProvider()
        provider.is_available = AsyncMock(return_value=False)

        health = await provider.get_health_status_async()

        assert health["healthy"] is False
        assert "error" in health

    async def test_get_quota_status(self):
        """Test quota status for local provider (unlimited)."""
        provider = LLaMAProvider()
        provider.request_count = 50

        quota = await provider.get_quota_status()

        assert quota.provider_name == "llama"
        assert quota.current_usage == 50
        assert quota.quota_limit == float("inf")
        assert quota.quota_remaining == float("inf")

    async def test_get_usage_stats(self):
        """Test usage statistics."""
        provider = LLaMAProvider()
        provider.request_count = 10
        provider.total_tokens = 1000
        provider.total_time = 25.0

        stats = await provider.get_usage_stats()

        assert stats.provider_name == "llama"
        assert stats.total_requests == 10
        assert stats.total_tokens_consumed == 1000
        assert stats.total_cost_usd == 0.0
        assert stats.average_response_time_ms == 2500.0  # 2.5 seconds * 1000
        assert stats.success_rate == 100.0

    async def test_shutdown(self):
        """Test provider shutdown."""
        provider = LLaMAProvider()
        provider.model_manager.unload_model = AsyncMock()

        await provider.shutdown()

        provider.model_manager.unload_model.assert_called_once()


class TestLLaMAProviderIntegration:
    """Integration tests for LLaMA provider."""

    @pytest.mark.skipif(
        not LLAMA_CPP_AVAILABLE,
        reason="llama-cpp-python not available",
    )
    async def test_real_availability_check(self):
        """Test availability check with real llama-cpp-python."""
        # This test only checks that the import and basic initialization work
        provider = LLaMAProvider()

        # Should fail because no model path is configured
        available = await provider.is_available()
        assert available is False

    async def test_provider_lifecycle(self):
        """Test complete provider lifecycle."""
        provider = LLaMAProvider(model_path="/test/model.gguf")

        # Mock all dependencies
        provider.model_manager.is_model_available = Mock(return_value=True)
        provider.model_manager.load_model = AsyncMock(return_value=True)
        provider.model_manager.unload_model = AsyncMock()
        provider.model_manager.model_loaded = True

        # Test availability
        available = await provider.is_available()
        assert available is True

        # Test health status
        health = await provider.get_health_status_async()
        assert health["healthy"] is True

        # Test shutdown
        await provider.shutdown()
        provider.model_manager.unload_model.assert_called_once()

    async def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        provider = LLaMAProvider()
        provider.is_available = AsyncMock(return_value=True)

        # Mock successful responses
        mock_response = {
            "choices": [{"text": "Response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        provider._generate_sync = Mock(return_value=mock_response)

        # Create multiple concurrent requests
        requests = [LLMRequest(prompt=f"Test prompt {i}") for i in range(5)]

        # Execute requests concurrently
        tasks = [provider.generate_response(request) for request in requests]

        responses = await asyncio.gather(*tasks)

        # Verify all responses
        assert len(responses) == 5
        for response in responses:
            assert response.content == "Response"
            assert response.total_tokens == 15

        # Verify statistics updated correctly
        assert provider.request_count == 5
        assert provider.total_tokens == 75  # 5 requests * 15 tokens each
