"""Core data models for MCP Server Cheap LLM.

This module defines Pydantic models for configuration, requests, and responses.
Follows atomic design patterns with clear data structures (200-300 lines).

Key models:
    ProviderConfig: Configuration for LLM providers
    LLMRequest: Standardized request format
    LLMResponse: Standardized response format
    ProviderStatus: Runtime status information

Example:
    >>> config = ProviderConfig(name="gemini", enabled=True)
    >>> request = LLMRequest(prompt="Hello", provider="gemini")
    >>> response = LLMResponse(content="Hi there!", provider="gemini")

"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    GEMINI = "gemini"
    OPENAI = "openai"
    CODEX = "codex"
    LLAMA = "llama"
    ANTHROPIC = "anthropic"


class ProviderStatus(str, Enum):
    """Provider operational status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class QuotaStatus(str, Enum):
    """Quota status for providers."""

    HEALTHY = "healthy"
    WARNING = "warning"
    EXCEEDED = "exceeded"


class BatchPriority(str, Enum):
    """Priority levels for batch processing."""

    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class BatchStatus(str, Enum):
    """Status of batch processing operations."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LLMRequest(BaseModel):
    """Standardized request format for LLM providers.

    Attributes:
        prompt: The text prompt to send to the LLM
        provider: Target provider (optional, uses default if not specified)
        model: Specific model to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)
        system_prompt: Optional system prompt for context
        metadata: Additional provider-specific parameters

    Example:
        >>> request = LLMRequest(
        ...     prompt="Explain Python decorators",
        ...     provider="gemini",
        ...     model="gemini-1.5-flash",
        ...     max_tokens=500,
        ...     temperature=0.7
        ... )

    """

    prompt: str = Field(..., min_length=1, max_length=10000)
    provider: str | None = None
    model: str | None = None
    max_tokens: int = Field(default=1000, ge=1, le=8000)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    system_prompt: str | None = Field(None, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure metadata doesn't exceed reasonable size."""
        if len(str(v)) > 5000:
            msg = "Metadata too large (max 5000 characters)"
            raise ValueError(msg)
        return v


class LLMResponse(BaseModel):
    """Standardized response format from LLM providers.

    Attributes:
        content: The generated text response
        provider: Which provider generated the response
        model: Model used for generation
        success: Whether the request succeeded
        error_message: Error details if success=False
        tokens_used: Number of tokens consumed
        token_count: Alias for tokens_used
        cost: Cost of the request in dollars
        response_time_ms: Response time in milliseconds
        metadata: Provider-specific response data

    Example:
        >>> response = LLMResponse(
        ...     content="Decorators are a way to modify functions...",
        ...     provider="gemini",
        ...     model="gemini-1.5-flash",
        ...     success=True,
        ...     tokens_used=45,
        ...     response_time_ms=1250
        ... )

    """

    content: str = ""
    provider: str
    model: str | None = None
    success: bool = True
    error_message: str | None = None
    tokens_used: int = Field(default=0, ge=0)
    token_count: int = Field(default=0, ge=0)  # Alias for compatibility
    cost: float = Field(default=0.0, ge=0.0)
    response_time_ms: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    def __init__(self, **data: Any) -> None:
        """Initialize with token_count alias handling."""
        super().__init__(**data)
        # Sync token_count with tokens_used if one is provided
        if self.token_count == 0 and self.tokens_used > 0:
            self.token_count = self.tokens_used
        elif self.tokens_used == 0 and self.token_count > 0:
            self.tokens_used = self.token_count

    @property
    def total_tokens(self) -> int:
        """Alias for tokens_used for backward compatibility."""
        return max(self.tokens_used, self.token_count)

    def to_debug_dict(self) -> dict[str, Any]:
        """Convert to dictionary for debugging purposes.

        Returns:
            Dictionary containing all response data for logging

        """
        return {
            "provider": self.provider,
            "model": self.model,
            "success": self.success,
            "content_length": len(self.content),
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "response_time_ms": self.response_time_ms,
            "has_error": self.error_message is not None,
            "created_at": self.created_at.isoformat(),
        }


class StreamingResponse(BaseModel):
    """Streaming response chunk from LLM providers.

    Attributes:
        content: The text chunk for this streaming response
        provider: Which provider generated the response
        model: Model used for generation
        is_final: Whether this is the final chunk
        chunk_index: Index of this chunk in the stream
        metadata: Provider-specific chunk data

    Example:
        >>> chunk = StreamingResponse(
        ...     content="Hello ",
        ...     provider="gemini",
        ...     model="gemini-pro",
        ...     is_final=False,
        ...     chunk_index=0
        ... )

    """

    content: str = ""
    provider: str
    model: str | None = None
    is_final: bool = False
    chunk_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ProviderConfig(BaseModel):
    """Configuration for individual LLM providers.

    Attributes:
        name: Unique provider identifier
        provider_type: Type of provider (gemini/openai/etc)
        enabled: Whether provider is active
        api_key: Authentication key (optional)
        base_url: Custom endpoint URL (optional)
        models: List of available models
        model_name: Default model name to use for this provider
        max_tokens: Maximum token limit
        rate_limit: Rate limiting configuration
        timeout: Request timeout in seconds
        provider_specific: Provider-specific configuration

    Example:
        >>> config = ProviderConfig(
        ...     name="gemini",
        ...     provider_type=ProviderType.GEMINI,
        ...     enabled=True,
        ...     api_key="your-api-key",
        ...     models=["gemini-1.5-flash", "gemini-1.5-pro"],
        ...     model_name="gemini-1.5-flash"
        ... )

    """

    name: str = Field(..., min_length=1, max_length=50)
    provider_type: ProviderType
    enabled: bool = True
    api_key: str | None = None
    base_url: str | None = None
    models: list[str] = Field(default_factory=list)
    model_name: str | None = None
    max_tokens: int = Field(default=4096, ge=1, le=32000)
    rate_limit: dict[str, Any] = Field(default_factory=dict)
    timeout: int = Field(default=30, ge=1, le=300)
    provider_specific: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is lowercase and alphanumeric with underscores."""
        if not v.replace("_", "").replace("-", "").isalnum():
            msg = "Name must contain only letters, numbers, hyphens, and underscores"
            raise ValueError(
                msg,
            )
        return v.lower()

    @field_validator("models")
    @classmethod
    def validate_models(cls, v: list[str]) -> list[str]:
        """Ensure models list is not empty if provided."""
        if v and len(v) == 0:
            msg = "Models list cannot be empty if provided"
            raise ValueError(msg)
        return v


class ServerConfig(BaseModel):
    """Configuration for the entire server.

    Attributes:
        providers: List of provider configurations
        default_provider: Name of default provider to use
        cache: Cache configuration
        logging: Logging configuration
        security: Security configuration
        monitoring: Monitoring configuration

    Example:
        >>> config = ServerConfig(
        ...     providers=[gemini_config, openai_config],
        ...     default_provider="gemini"
        ... )

    """

    providers: list[ProviderConfig] = Field(default_factory=list)
    default_provider: str | None = None
    cache: dict[str, Any] = Field(default_factory=dict)
    logging: dict[str, Any] = Field(default_factory=dict)
    security: dict[str, Any] = Field(default_factory=dict)
    monitoring: dict[str, Any] = Field(default_factory=dict)

    @field_validator("default_provider")
    @classmethod
    def validate_default_provider(cls, v: str | None, values: Any) -> str | None:
        """Ensure default provider exists in providers list."""
        if v is None:
            return v

        # Note: In Pydantic v2, we can't access other fields during validation
        # This validation would need to be done elsewhere or as a model validator
        return v

    def get_provider_config(self, name: str) -> ProviderConfig | None:
        """Get configuration for a specific provider.

        Args:
            name: Provider name

        Returns:
            Provider configuration or None if not found

        """
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None

    def get_enabled_providers(self) -> list[str]:
        """Get list of enabled provider names.

        Returns:
            List of provider names that are enabled

        """
        return [p.name for p in self.providers if p.enabled]


class UsageStats(BaseModel):
    """Usage statistics for tracking provider performance.

    Attributes:
        provider_name: Name of the provider
        total_requests: Total number of requests made
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        total_tokens: Total tokens processed
        total_cost: Total cost incurred
        average_response_time: Average response time in milliseconds
        last_updated: When these stats were last updated

    Example:
        >>> stats = UsageStats(
        ...     provider_name="gemini",
        ...     total_requests=1000,
        ...     successful_requests=950,
        ...     total_tokens=50000,
        ...     total_cost=12.50
        ... )

    """

    provider_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)

    @property
    def requests_count(self) -> int:
        """Alias for total_requests for backward compatibility."""
        return self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        return 100.0 - self.success_rate

    @property
    def total_tokens_consumed(self) -> int:
        """Alias for total_tokens for backward compatibility."""
        return self.total_tokens

    @property
    def total_cost_usd(self) -> float:
        """Alias for total_cost for backward compatibility."""
        return self.total_cost

    @property
    def average_response_time_ms(self) -> float:
        """Alias for average_response_time for backward compatibility."""
        return self.average_response_time

    def update_stats(
        self,
        success: bool,
        tokens: int = 0,
        cost: float = 0.0,
        response_time: float = 0.0,
    ) -> None:
        """Update statistics with new request data.

        Args:
            success: Whether the request was successful
            tokens: Number of tokens processed
            cost: Cost of the request
            response_time: Response time in milliseconds

        """
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_tokens += tokens
        self.total_cost += cost

        # Update average response time (simple moving average)
        if self.total_requests == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                self.average_response_time * (self.total_requests - 1) + response_time
            ) / self.total_requests

        self.last_updated = datetime.now()


class ErrorInfo(BaseModel):
    """Error information for tracking and debugging.

    Attributes:
        error_code: Structured error code
        error_message: Human-readable error message
        error_type: Type/category of error
        provider: Provider that generated the error
        request_id: Unique request identifier
        timestamp: When the error occurred
        context: Additional error context

    Example:
        >>> error = ErrorInfo(
        ...     error_code="RATE_LIMIT_EXCEEDED",
        ...     error_message="API rate limit exceeded",
        ...     error_type="rate_limit",
        ...     provider="openai"
        ... )

    """

    error_code: str
    error_message: str
    error_type: str
    provider: str | None = None
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    context: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert error info to dictionary for logging."""
        return {
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "provider": self.provider,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


class CostEstimate(BaseModel):
    """Cost estimation for LLM requests.

    Provides detailed cost analysis including token estimates,
    pricing breakdown, and confidence metrics.

    Attributes:
        provider_name: Name of the provider
        estimated_tokens: Estimated total tokens required
        cost_per_token: Cost per token in USD
        estimated_cost_usd: Total estimated cost in USD
        confidence_score: Confidence in the estimate (0.0-1.0)
        estimation_method: Method used for estimation
        cost_breakdown: Detailed cost breakdown

    Example:
        >>> estimate = CostEstimate(
        ...     provider_name="openai",
        ...     estimated_tokens=1500,
        ...     cost_per_token=0.000002,
        ...     estimated_cost_usd=0.003,
        ...     confidence_score=0.8,
        ...     estimation_method="token_based"
        ... )

    """

    provider_name: str
    estimated_tokens: int = Field(ge=0)
    cost_per_token: float = Field(ge=0.0)
    estimated_cost_usd: float = Field(ge=0.0)
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    estimation_method: str = Field(default="token_based")
    cost_breakdown: dict[str, Any] = Field(default_factory=dict)


class ProviderStatusInfo(BaseModel):
    """Detailed provider status information.

    Contains comprehensive status data for provider monitoring
    and health checking.

    Attributes:
        provider_name: Name of the provider
        status: Current operational status
        health_score: Overall health score (0.0-1.0)
        last_check: Timestamp of last status check
        response_time_ms: Average response time in milliseconds
        error_rate: Current error rate (0.0-1.0)
        quota_remaining: Remaining quota percentage
        metadata: Additional status metadata

    Example:
        >>> status_info = ProviderStatusInfo(
        ...     provider_name="gemini",
        ...     status=ProviderStatus.HEALTHY,
        ...     health_score=0.95,
        ...     response_time_ms=250,
        ...     error_rate=0.02
        ... )

    """

    provider_name: str
    status: ProviderStatus
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    last_check: datetime = Field(default_factory=datetime.now)
    response_time_ms: float = Field(default=0.0, ge=0.0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    quota_remaining: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuotaStatusInfo(BaseModel):
    """Quota status information for tracking provider quotas.

    Contains detailed quota information including current usage,
    limits, and reset times.

    Attributes:
        provider_name: Name of the provider
        quota_type: Type of quota (requests, tokens, etc.)
        current_usage: Current usage count
        quota_limit: Maximum quota limit
        quota_remaining: Remaining quota count
        reset_time: When quota resets (None for unlimited)
        estimated_reset_duration: Time until quota reset

    Example:
        >>> quota = QuotaStatusInfo(
        ...     provider_name="openai",
        ...     quota_type="requests",
        ...     current_usage=500,
        ...     quota_limit=1000,
        ...     quota_remaining=500
        ... )

    """

    provider_name: str
    quota_type: str = "requests"
    current_usage: int | float = 0
    quota_limit: int | float = Field(default=float("inf"))
    quota_remaining: int | float = Field(default=float("inf"))
    reset_time: datetime | None = None
    estimated_reset_duration: float | None = None

    @property
    def value(self) -> str:
        """Computed quota status value.

        Returns:
            str: "healthy", "warning", or "exceeded"
        """
        if self.quota_limit == float("inf"):
            return "healthy"

        usage_percentage = self.current_usage / self.quota_limit

        if usage_percentage >= 1.0:
            return "exceeded"
        elif usage_percentage >= 0.8:  # 80% threshold
            return "warning"
        else:
            return "healthy"


class BatchRequest(BaseModel):
    """Request for batch processing of multiple LLM requests.

    Attributes:
        batch_id: Unique identifier for the batch
        requests: List of LLM requests to process
        priority: Processing priority for the batch
        similarity_threshold: Threshold for similarity analysis (0.0-1.0)
        max_parallel: Maximum parallel processing for this batch
        callback_url: Optional URL for batch completion notifications
        metadata: Additional batch metadata
        created_at: Timestamp when batch was created
        estimated_processing_time: Estimated processing time in seconds

    Example:
        >>> batch = BatchRequest(
        ...     batch_id="batch_123",
        ...     requests=[request1, request2, request3],
        ...     priority=BatchPriority.HIGH,
        ...     similarity_threshold=0.7
        ... )

    """

    batch_id: str = Field(..., min_length=1, max_length=100)
    requests: list[LLMRequest] = Field(..., min_length=1, max_length=100)
    priority: BatchPriority = BatchPriority.NORMAL
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_parallel: int = Field(default=5, ge=1, le=20)
    callback_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    estimated_processing_time: float = Field(default=0.0, ge=0.0)

    @field_validator("batch_id")
    @classmethod
    def validate_batch_id(cls, v: str) -> str:
        """Ensure batch ID is alphanumeric with allowed separators."""
        if not v.replace("_", "").replace("-", "").isalnum():
            msg = (
                "Batch ID must contain only letters, numbers, hyphens, and underscores"
            )
            raise ValueError(msg)
        return v


class BatchResponse(BaseModel):
    """Response from batch processing operation.

    Attributes:
        batch_id: ID of the processed batch
        status: Current processing status
        responses: List of successful LLM responses
        failed_requests: Indices of requests that failed processing
        processing_time_ms: Total processing time in milliseconds
        queue_time_ms: Time spent in queue before processing
        cache_hits: Number of requests served from cache
        total_tokens_used: Total tokens consumed across all requests
        total_cost_usd: Total cost in USD for the entire batch
        similarity_groups: Information about detected similarity groups
        completed_at: Timestamp when batch processing completed
        error_details: Error information if batch failed
        metadata: Additional response metadata

    Example:
        >>> response = BatchResponse(
        ...     batch_id="batch_123",
        ...     status=BatchStatus.COMPLETED,
        ...     responses=[response1, response2],
        ...     processing_time_ms=5000,
        ...     total_tokens_used=1500
        ... )

    """

    batch_id: str
    status: BatchStatus
    responses: list[LLMResponse] = Field(default_factory=list)
    failed_requests: list[int] = Field(default_factory=list)
    processing_time_ms: int = Field(default=0, ge=0)
    queue_time_ms: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    total_tokens_used: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    similarity_groups: list[dict[str, Any]] = Field(default_factory=list)
    completed_at: datetime | None = None
    error_details: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total_requests = len(self.responses) + len(self.failed_requests)
        if total_requests == 0:
            return 0.0
        return (len(self.responses) / total_requests) * 100.0


class BatchQueueInfo(BaseModel):
    """Information about a batch's position and status in the processing queue.

    Attributes:
        queue_position: Current position in the queue (1-based)
        queue_depth: Total number of batches in queue
        estimated_wait_time_ms: Estimated wait time in milliseconds
        processing_capacity: Number of concurrent processing slots
        average_batch_time_ms: Average processing time for recent batches
        priority_queue_depth: Number of batches by priority level

    Example:
        >>> queue_info = BatchQueueInfo(
        ...     queue_position=3,
        ...     queue_depth=10,
        ...     estimated_wait_time_ms=45000,
        ...     processing_capacity=3
        ... )

    """

    queue_position: int = Field(ge=1)
    queue_depth: int = Field(ge=0)
    estimated_wait_time_ms: int = Field(ge=0)
    processing_capacity: int = Field(ge=1)
    average_batch_time_ms: float = Field(ge=0.0)
    priority_queue_depth: dict[str, int] = Field(default_factory=dict)


class BatchMetrics(BaseModel):
    """Comprehensive metrics for batch processing system performance.

    Attributes:
        total_batches_processed: Total number of batches processed
        total_requests_processed: Total number of individual requests processed
        average_batch_time_ms: Average batch processing time in milliseconds
        average_queue_time_ms: Average time batches spend in queue
        cache_hit_rate_percent: Cache hit rate as percentage
        throughput_batches_per_hour: Number of batches processed per hour
        throughput_requests_per_hour: Number of requests processed per hour
        error_rate_percent: Error rate as percentage
        similarity_optimization_rate: Rate of similarity-based optimizations
        current_queue_depth: Current number of batches in queue
        active_processing_slots: Number of currently active processing workers
        system_load_percent: Overall system load as percentage

    Example:
        >>> metrics = BatchMetrics(
        ...     total_batches_processed=1000,
        ...     average_batch_time_ms=30000,
        ...     cache_hit_rate_percent=25.5,
        ...     throughput_batches_per_hour=120
        ... )

    """

    total_batches_processed: int = Field(default=0, ge=0)
    total_requests_processed: int = Field(default=0, ge=0)
    average_batch_time_ms: float = Field(default=0.0, ge=0.0)
    average_queue_time_ms: float = Field(default=0.0, ge=0.0)
    cache_hit_rate_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    throughput_batches_per_hour: float = Field(default=0.0, ge=0.0)
    throughput_requests_per_hour: float = Field(default=0.0, ge=0.0)
    error_rate_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    similarity_optimization_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    current_queue_depth: int = Field(default=0, ge=0)
    active_processing_slots: int = Field(default=0, ge=0)
    system_load_percent: float = Field(default=0.0, ge=0.0, le=100.0)


__all__ = [
    # Enums
    "ProviderType",
    "ProviderStatus",
    "QuotaStatus",
    "BatchStatus",
    "BatchPriority",
    # Core models
    "ProviderConfig",
    "LLMRequest",
    "LLMResponse",
    "CostEstimate",
    "UsageStats",
    "QuotaStatusInfo",
    "ProviderStatusInfo",
    # Batch models
    "BatchRequest",
    "BatchResponse",
    "BatchQueueInfo",
    "BatchMetrics",
]
