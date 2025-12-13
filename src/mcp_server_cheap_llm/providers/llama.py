"""Local LLaMA Model Provider Implementation.

This module implements the LLaMA provider for the cheap LLM server,
providing access to local LLaMA models through llama-cpp-python.

Features:
- Local model execution with no API costs
- GPU acceleration support (CUDA/Metal)
- Context window management and memory optimization
- Model switching capabilities
- Performance monitoring and resource tracking
- Support for quantized models
- Model download and installation utilities
"""

import asyncio
import os
import platform
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import psutil

from mcp_server_cheap_llm.core.errors import ProviderError
from mcp_server_cheap_llm.core.models import (
    CostEstimate,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderType,
    QuotaStatusInfo,
    StreamingResponse,
    UsageStats,
)
from mcp_server_cheap_llm.utils.logging import get_logger

from .base import LLMProvider, ProviderCapabilities

logger = get_logger(__name__)

# Optional import for llama-cpp-python
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    logger.warning("llama-cpp-python not available. LLaMA provider will be disabled.")
    Llama = None
    LLAMA_CPP_AVAILABLE = False


class LLaMAModelManager:
    """Manages LLaMA model loading, switching, and resource monitoring."""

    def __init__(
        self,
        model_path=None,  # type: str | None
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        use_mmap: bool = True,
        use_mlock: bool = False,
        n_threads=None,  # type: int | None
    ) -> None:
        """Initialize model manager with configuration."""
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.n_threads = n_threads or min(8, os.cpu_count() or 4)

        self.model = None  # type: ignore
        self.model_loaded = False
        self.load_time = 0.0
        self.memory_usage = 0.0
        self.gpu_enabled = False

    def is_model_available(self) -> bool:
        """Check if model file exists and is accessible."""
        if not self.model_path or self.model_path == "":
            return False

        try:
            model_file = Path(self.model_path)
            return model_file.exists() and model_file.is_file()
        except (OSError, ValueError):
            return False

    async def load_model(self) -> bool:
        """Load the LLaMA model with error handling."""
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python not available")
            return False

        if not self.is_model_available():
            logger.error(f"Model file not found: {self.model_path}")
            return False

        try:
            start_time = time.time()
            logger.info(f"Loading LLaMA model from {self.model_path}")

            # Detect GPU capabilities
            gpu_layers = self._detect_gpu_layers()

            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                self._load_model_sync,
                gpu_layers,
            )

            self.load_time = time.time() - start_time
            self.model_loaded = True
            self.memory_usage = self._get_memory_usage()

            logger.info(
                f"Model loaded successfully in {self.load_time:.2f}s, "
                f"Memory usage: {self.memory_usage:.1f}MB, "
                f"GPU layers: {gpu_layers}",
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            self.model = None
            self.model_loaded = False
            return False

    def _load_model_sync(self, gpu_layers: int):
        """Synchronous model loading for thread pool execution."""
        if not LLAMA_CPP_AVAILABLE or not Llama:
            msg = "llama-cpp-python not available"
            raise RuntimeError(msg)
        if not self.model_path:
            msg = "Model path not configured"
            raise RuntimeError(msg)
        return Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=gpu_layers,
            use_mmap=self.use_mmap,
            use_mlock=self.use_mlock,
            n_threads=self.n_threads,
            verbose=False,
        )

    def _detect_gpu_layers(self) -> int:
        """Detect optimal GPU layers based on available hardware."""
        if self.n_gpu_layers >= 0:
            return self.n_gpu_layers

        # Auto-detect GPU capabilities
        system = platform.system().lower()

        # Check for NVIDIA GPU (CUDA)
        try:
            import GPUtil  # type: ignore

            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_enabled = True
                # Use all layers for GPU acceleration
                return 999  # Let llama-cpp-python decide optimal layers
        except ImportError:
            pass

        # Check for Apple Silicon (Metal)
        if system == "darwin" and platform.processor() == "arm":
            self.gpu_enabled = True
            return 999  # Metal acceleration

        # Default to CPU-only
        logger.info("No GPU acceleration detected, using CPU-only mode")
        return 0

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    async def unload_model(self) -> None:
        """Unload the current model to free memory."""
        if self.model:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._unload_model_sync)

    def _unload_model_sync(self) -> None:
        """Synchronous model unloading."""
        if self.model:
            del self.model
            self.model = None
            self.model_loaded = False
            logger.info("Model unloaded successfully")

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "loaded": self.model_loaded,
            "load_time": self.load_time,
            "memory_usage_mb": self.memory_usage,
            "gpu_enabled": self.gpu_enabled,
            "context_size": self.n_ctx,
            "gpu_layers": self.n_gpu_layers,
            "threads": self.n_threads,
        }


class LLaMAProvider(LLMProvider):
    """Local LLaMA Model Provider.

    Implements LLM provider interface for local LLaMA models using llama-cpp-python.
    Includes model management, GPU acceleration, and performance monitoring.
    """

    PROVIDER_TYPE = ProviderType.LLAMA

    def __init__(
        self,
        config: ProviderConfig | None = None,
        model_path=None,  # type: str | None
        model_name: str = "llama-local",
        n_ctx: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 512,
        n_gpu_layers: int = -1,
        **kwargs,
    ) -> None:
        """Initialize LLaMA provider with configuration."""
        # Create default config if none provided
        if config is None:
            config = ProviderConfig(
                name="llama",
                provider_type=ProviderType.LLAMA,
                models=[model_name] if model_name else [],
                enabled=True,
                base_url="local",
                api_key=None,  # Not needed for local models
                provider_specific={
                    "rate_limit_per_minute": 100,
                },  # High rate limit for local processing
                timeout=60,
            )

        super().__init__(config)

        # Override name to always be "llama" regardless of config name
        self.name = "llama"

        # Model configuration
        self.model_path = model_path or os.getenv("LLAMA_MODEL_PATH")
        self.model_name = model_name
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

        # Initialize model manager
        if self.model_path:
            self.model_manager = LLaMAModelManager(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                **kwargs,
            )
        else:
            # Create manager without path for configuration testing
            self.model_manager = LLaMAModelManager(
                model_path="",
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                **kwargs,
            )

        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        self.total_time = 0.0

        # Set capabilities
        self.capabilities = {
            ProviderCapabilities.STREAMING,
            ProviderCapabilities.ASYNC_GENERATION,
        }

    async def is_available(self) -> bool:
        """Check if the LLaMA provider is available."""
        if not LLAMA_CPP_AVAILABLE:
            logger.warning("llama-cpp-python not available")
            return False

        if not self.model_path:
            logger.warning("No model path configured for LLaMA provider")
            return False

        if not self.model_manager.is_model_available():
            logger.warning(f"Model file not accessible: {self.model_path}")
            return False

        # Ensure model is loaded
        if not self.model_manager.model_loaded:
            await self.model_manager.load_model()

        return self.model_manager.model_loaded

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the local LLaMA model."""
        if not await self.is_available():
            msg = "LLaMA provider not available"
            raise ProviderError(msg, provider="llama")

        start_time = time.time()

        try:
            # Prepare generation parameters
            generation_params = {
                "prompt": request.prompt,
                "max_tokens": min(
                    request.max_tokens or self.max_tokens,
                    self.max_tokens,
                ),
                "temperature": request.temperature or self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "stop": [],
                "echo": False,
            }

            # Run generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_sync,
                generation_params,
            )

            # Extract response data
            response_text = response["choices"][0]["text"]
            finish_reason = response["choices"][0]["finish_reason"]

            # Calculate timing and token usage
            generation_time = time.time() - start_time
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
            total_tokens = response["usage"]["total_tokens"]

            # Update statistics
            self.request_count += 1
            self.total_tokens += total_tokens
            self.total_time += generation_time

            logger.info(
                f"LLaMA generation completed in {generation_time:.2f}s, "
                f"tokens: {prompt_tokens}+{completion_tokens}={total_tokens}",
            )

            return LLMResponse(
                content=response_text,
                provider=ProviderType.LLAMA,
                success=True,
                tokens_used=total_tokens,
                response_time_ms=int(generation_time * 1000),
                metadata={
                    "model": self.model_name,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                },
            )

        except Exception as e:
            logger.exception(f"LLaMA generation failed: {e}")
            msg = f"Failed to generate response: {e}"
            raise ProviderError(
                msg,
                provider="llama",
            ) from e

    # Alias for test compatibility
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Alias for generate method to maintain test compatibility."""
        return await self.generate(request)

    def _generate_sync(self, params: dict[str, Any]) -> Any:
        """Synchronous generation for thread pool execution."""
        if not self.model_manager.model:
            msg = "Model not loaded"
            raise ProviderError(msg, provider="llama")

        return self.model_manager.model(**params)
        # llama-cpp-python returns response objects that can be used as dicts

    async def generate_streaming_response(
        self,
        request: LLMRequest,
    ) -> AsyncGenerator[StreamingResponse, None]:
        """Generate a streaming response using the local LLaMA model."""
        if not await self.is_available():
            msg = "LLaMA provider not available"
            raise ProviderError(msg, provider="llama")

        try:
            # Prepare generation parameters for streaming
            generation_params = {
                "prompt": request.prompt,
                "max_tokens": min(
                    request.max_tokens or self.max_tokens,
                    self.max_tokens,
                ),
                "temperature": request.temperature or self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "stop": [],
                "stream": True,
                "echo": False,
            }

            # Run streaming generation
            loop = asyncio.get_event_loop()

            # Create async generator for streaming
            async for chunk in self._stream_generate_async(generation_params, loop):
                if chunk:
                    yield StreamingResponse(
                        content=chunk,
                        provider="llama",
                        model=self.model_name,
                        is_final=False,
                    )

        except Exception as e:
            logger.exception(f"LLaMA streaming failed: {e}")
            msg = f"Failed to generate streaming response: {e}"
            raise ProviderError(
                msg,
                provider="llama",
            ) from e

    async def _stream_generate_async(
        self,
        params: dict[str, Any],
        loop: asyncio.AbstractEventLoop,
    ) -> AsyncGenerator[str, None]:
        """Async generator for streaming LLaMA responses."""

        def stream_sync():
            """Synchronous streaming generator."""
            if not self.model_manager.model:
                msg = "Model not loaded"
                raise ProviderError(msg, provider="llama")

            for output in self.model_manager.model(**params):
                # llama-cpp-python returns TypedDict objects
                if (
                    output
                    and isinstance(output, dict)
                    and "choices" in output
                    and output["choices"]
                ):
                    choices = output["choices"]
                    if choices and len(choices) > 0:
                        choice = choices[0]
                        if (
                            isinstance(choice, dict)
                            and "delta" in choice
                            and isinstance(choice["delta"], dict)
                            and "content" in choice["delta"]
                        ):
                            content = choice["delta"]["content"]
                            if content:
                                yield content

        # Run in thread pool with queue for async communication
        import queue
        import threading

        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def producer() -> None:
            try:
                for chunk in stream_sync():
                    result_queue.put(chunk)
                result_queue.put(None)  # Signal completion
            except Exception as e:
                exception_queue.put(e)
                result_queue.put(None)

        # Start producer thread
        thread = threading.Thread(target=producer)
        thread.start()

        try:
            while True:
                # Check for exceptions
                if not exception_queue.empty():
                    raise exception_queue.get()

                # Get next chunk
                try:
                    chunk = result_queue.get(timeout=0.1)
                    if chunk is None:
                        break
                    yield chunk
                except queue.Empty:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

        finally:
            thread.join(timeout=1.0)

    def validate_config(self, config: ProviderConfig) -> bool:
        """Validate provider configuration."""
        # LLaMA provider has minimal configuration requirements
        return bool(config.name) and config.provider_type == ProviderType.LLAMA

    async def get_usage(self) -> UsageStats:
        """Get current usage statistics."""
        return await self.get_usage_stats()

    async def check_quota(self) -> QuotaStatusInfo:
        """Check current quota status."""
        return await self.get_quota_status()

    async def estimate_cost(self, request: LLMRequest) -> CostEstimate:
        """Estimate cost for a request (always $0 for local models)."""
        # Local models have no cost
        return CostEstimate(
            provider_name="llama",
            estimated_tokens=request.max_tokens,
            cost_per_token=0.0,
            estimated_cost_usd=0.0,
        )

    def get_health_status(self) -> dict[str, Any]:
        """Get detailed health status of the LLaMA provider."""
        # Keep sync to match base class interface
        base_health = super().get_health_status()

        # Check if provider is healthy using simple availability check
        # For sync version, check basic requirements without async operations
        is_healthy = (
            LLAMA_CPP_AVAILABLE
            and self.model_path is not None
            and self.model_manager.model_loaded
        )

        # Add LLaMA-specific health info
        llama_health = {
            "healthy": is_healthy,
            "model_path": self.model_path,
            "llama_cpp_available": LLAMA_CPP_AVAILABLE,
            "model_loaded": self.model_manager.model_loaded,
            "model_info": self.model_manager.get_model_info()
            if self.model_manager.model_loaded
            else None,
            "performance_stats": {
                "request_count": self.request_count,
                "total_tokens": self.total_tokens,
                "total_time": self.total_time,
            },
            "system_resources": {
                "memory_usage_mb": self.model_manager.memory_usage,
                "gpu_enabled": self.model_manager.gpu_enabled,
                "n_gpu_layers": self.model_manager.n_gpu_layers,
            },
        }

        # Add error key when unhealthy for test compatibility
        if not is_healthy:
            llama_health["error"] = "Provider not available or model not loaded"

        base_health.update(llama_health)
        return base_health

    async def get_health_status_async(self) -> dict[str, Any]:
        """Get detailed health status of the LLaMA provider (async version)."""
        # Get base health from sync method
        base_health = self.get_health_status()

        # Update with async availability check
        try:
            is_healthy = await self.is_available()
            base_health.update(
                {
                    "healthy": is_healthy,
                    "error": "Provider not available or model not loaded"
                    if not is_healthy
                    else None,
                },
            )
            # Remove error key if healthy
            if is_healthy and "error" in base_health:
                del base_health["error"]
        except Exception:
            base_health.update(
                {"healthy": False, "error": "Failed to check availability"},
            )

        return base_health

    async def get_detailed_health_status(self) -> dict[str, Any]:
        """Get detailed health status of the LLaMA provider (async version)."""
        is_healthy = await self.is_available()

        health_status = {
            "healthy": is_healthy,
            "provider": "llama",
            "model_path": self.model_path,
            "llama_cpp_available": LLAMA_CPP_AVAILABLE,
            "model_info": self.model_manager.get_model_info(),
            "performance_stats": {
                "request_count": self.request_count,
                "total_tokens": self.total_tokens,
                "total_time": self.total_time,
                "avg_time_per_request": (
                    self.total_time / self.request_count
                    if self.request_count > 0
                    else 0.0
                ),
                "tokens_per_second": (
                    self.total_tokens / self.total_time if self.total_time > 0 else 0.0
                ),
            },
            "system_resources": {
                "cpu_count": os.cpu_count(),
                "memory_usage_mb": self.model_manager.memory_usage,
                "gpu_enabled": self.model_manager.gpu_enabled,
            },
        }

        if not is_healthy:
            health_status["error"] = "Provider not available"
            if not LLAMA_CPP_AVAILABLE:
                health_status["error"] = "llama-cpp-python not installed"
            elif not self.model_path:
                health_status["error"] = "No model path configured"
            elif not self.model_manager.is_model_available():
                health_status["error"] = f"Model file not found: {self.model_path}"

        return health_status

    async def get_quota_status(self) -> QuotaStatusInfo:
        """Get quota status for local LLaMA (unlimited for local models)."""
        return QuotaStatusInfo(
            provider_name="llama",
            quota_type="requests",
            current_usage=self.request_count,
            quota_limit=float("inf"),  # Unlimited for local models
            quota_remaining=float("inf"),
            reset_time=None,  # No reset needed
            estimated_reset_duration=None,
        )

    async def get_usage_stats(self) -> UsageStats:
        """Get usage statistics for the LLaMA provider."""
        return UsageStats(
            provider_name="llama",
            total_requests=self.request_count,
            successful_requests=self.request_count,  # Assume all successful for now
            total_tokens=self.total_tokens,
            total_cost=0.0,  # No cost for local models
            average_response_time=self.total_time * 1000 / self.request_count
            if self.request_count > 0
            else 0.0,
        )

    async def shutdown(self) -> None:
        """Cleanup provider resources."""
        logger.info("Shutting down LLaMA provider...")
        await self.model_manager.unload_model()
        # No super().shutdown() needed for base class
