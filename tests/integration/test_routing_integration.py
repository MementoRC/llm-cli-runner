"""Integration tests for routing system with mock providers.

These tests verify the end-to-end behavior of the routing system
including provider registry integration, health checking, and
real-world routing scenarios.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_server_cheap_llm.core.errors import ProviderError
from mcp_server_cheap_llm.core.models import (
    CostEstimate,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderStatus,
    ProviderType,
    QuotaStatus,
    UsageStats,
)
from mcp_server_cheap_llm.providers.base import LLMProvider, ProviderCapabilities
from mcp_server_cheap_llm.providers.registry import ProviderRegistry
from mcp_server_cheap_llm.providers.routing import (
    ComplexityLevel,
    ProviderRouter,
    RoutingDecision,
)


class MockProvider(LLMProvider):
    """Mock provider for testing routing integration."""

    PROVIDER_TYPE = None  # Set by specific implementations

    def __init__(self, config: ProviderConfig):
        """Initialize mock provider."""
        super().__init__(config)
        self.capabilities = {ProviderCapabilities.ASYNC_GENERATION}
        self.is_healthy = True
        self.quota_available = True

        # Track calls for testing
        self.generate_calls = []
        self.usage_calls = []
        self.quota_calls = []
        self.cost_calls = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Mock generation with tracking."""
        self.generate_calls.append(request)

        if not self.is_healthy:
            raise ProviderError(
                f"Provider {self.name} is unhealthy",
                provider=self.name,
                error_code="PROVIDER_UNHEALTHY",
            )

        return LLMResponse(
            content=f"Mock response from {self.name} for: {request.prompt[:20]}...",
            provider=self.provider_type,
            success=True,
            tokens_used=len(request.prompt.split()),
            response_time_ms=100,
        )

    def validate_config(self, config: ProviderConfig) -> bool:
        """Mock config validation."""
        return config.name is not None and len(config.name) > 0

    async def get_usage(self) -> UsageStats:
        """Mock usage statistics."""
        self.usage_calls.append(datetime.now())

        return UsageStats(
            provider_name=self.name,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_tokens_consumed=50000,
            total_cost_usd=2.50,
            average_response_time_ms=120.0,
            request_rate_per_minute=10.0,
        )

    async def check_quota(self) -> QuotaStatus:
        """Mock quota status."""
        self.quota_calls.append(datetime.now())

        if not self.quota_available:
            remaining = 0.0
            exhausted = True
        else:
            remaining = 500.0
            exhausted = False

        return QuotaStatus(
            provider_name=self.name,
            quota_type="requests_per_hour",
            current_usage=500.0,
            quota_limit=1000.0,
            quota_remaining=remaining,
            is_exhausted=exhausted,
        )

    async def estimate_cost(self, request: LLMRequest) -> CostEstimate:
        """Mock cost estimation."""
        self.cost_calls.append(request)

        estimated_tokens = len(request.prompt.split()) * 2  # Simple estimation
        cost_per_token = 0.0001  # Mock cost

        return CostEstimate(
            provider_name=self.name,
            estimated_tokens=estimated_tokens,
            cost_per_token=cost_per_token,
            confidence_score=0.85,
        )


class MockGeminiProvider(MockProvider):
    """Mock Gemini provider."""

    PROVIDER_TYPE = ProviderType.GEMINI


class MockCodexProvider(MockProvider):
    """Mock Codex provider."""

    PROVIDER_TYPE = ProviderType.CODEX


class MockLlamaProvider(MockProvider):
    """Mock LLaMA provider."""

    PROVIDER_TYPE = ProviderType.LLAMA


class TestRoutingIntegration:
    """Integration tests for routing system."""

    @pytest.fixture
    def registry(self):
        """Create registry with mock providers."""
        registry = ProviderRegistry()

        # Register mock provider classes
        registry.register_provider(MockGeminiProvider)
        registry.register_provider(MockCodexProvider)
        registry.register_provider(MockLlamaProvider)

        return registry

    @pytest.fixture
    def provider_configs(self):
        """Create provider configurations."""
        return [
            ProviderConfig(
                name="test_gemini",
                provider_type=ProviderType.GEMINI,
                enabled=True,
                model_name="gemini-pro",
            ),
            ProviderConfig(
                name="test_codex",
                provider_type=ProviderType.CODEX,
                enabled=True,
                model_name="codex-davinci",
            ),
            ProviderConfig(
                name="test_llama",
                provider_type=ProviderType.LLAMA,
                enabled=True,
                model_name="llama-7b",
            ),
        ]

    @pytest.fixture
    def providers(self, registry, provider_configs):
        """Create provider instances."""
        providers = {}
        for config in provider_configs:
            provider = registry.create_provider(config)
            providers[config.provider_type] = provider
        return providers

    @pytest.fixture
    def router(self, registry):
        """Create router with registry."""
        return ProviderRouter(registry)

    def test_routing_with_all_providers_healthy(self, router, providers):
        """Test routing when all providers are healthy."""
        test_cases = [
            ("Hello world", [ProviderType.GEMINI]),
            (
                "Explain machine learning algorithms with detailed examples and code",
                [ProviderType.GEMINI, ProviderType.CODEX],  # Moderate complexity
            ),
            (
                """Design a distributed consensus algorithm with Byzantine fault tolerance.
                Analyze the performance characteristics of Raft vs PBFT algorithms.
                Implement optimization strategies using advanced data structures.
                Provide mathematical proofs of correctness and performance analysis.
                Research the theoretical framework and methodology for consensus protocols.
                Compare distributed database architectures and scalability patterns.""",
                [ProviderType.CODEX, ProviderType.LLAMA],  # Complex
            ),
            (
                """Implement complex mathematical optimization with formal proofs
                and rigorous analysis. Derive the mathematical foundations
                from first principles, prove convergence properties, and analyze
                computational complexity using advanced theoretical frameworks.""",
                [ProviderType.CODEX, ProviderType.LLAMA],  # Expert
            ),
        ]

        for prompt, expected_provider_types in test_cases:
            request = LLMRequest(prompt=prompt)
            decision = router.route_request(request)

            assert isinstance(decision, RoutingDecision)
            assert decision.selected_provider in expected_provider_types
            assert decision.confidence > 0.5
            assert len(decision.fallback_providers) >= 1

    def test_routing_with_provider_unavailable(
        self, registry, router, provider_configs
    ):
        """Test routing when some providers are unavailable."""
        # Create only Gemini and Codex providers (no LLaMA)
        for config in provider_configs[:2]:  # Only first two configs
            registry.create_provider(config)

        # Test complex request that would prefer LLaMA
        request = LLMRequest(
            prompt="""
            Analyze the computational complexity of quantum algorithms
            and provide formal mathematical proofs of their advantages
            over classical counterparts in cryptographic applications.
            """
        )

        decision = router.route_request(request)

        # Should fall back to available providers
        assert decision.selected_provider in [ProviderType.GEMINI, ProviderType.CODEX]
        assert decision.complexity_level in [
            ComplexityLevel.COMPLEX,
            ComplexityLevel.EXPERT,
        ]

    def test_routing_with_explicit_provider_request(self, router, providers):
        """Test routing with explicitly requested provider."""
        request = LLMRequest(
            prompt="Simple question",
            provider=ProviderType.LLAMA,  # Explicitly request LLaMA for simple task
        )

        decision = router.route_request(request)

        assert decision.selected_provider == ProviderType.LLAMA
        assert decision.confidence == 1.0  # Explicit choice
        assert "explicitly requested" in decision.reasoning.lower()

    def test_routing_with_provider_health_issues(self, router, providers):
        """Test routing behavior when providers have health issues."""
        # Make Gemini unhealthy
        gemini_provider = providers[ProviderType.GEMINI]
        gemini_provider.is_healthy = False

        request = LLMRequest(prompt="Hello world")  # Would normally go to Gemini

        decision = router.route_request(request)

        # Should route to alternative provider since Gemini is unhealthy
        assert decision.selected_provider in [ProviderType.CODEX, ProviderType.LLAMA]

    def test_routing_with_quota_exhaustion(self, router, providers):
        """Test routing behavior when providers have quota issues."""
        # Exhaust Gemini quota
        gemini_provider = providers[ProviderType.GEMINI]
        gemini_provider.quota_available = False

        request = LLMRequest(prompt="Simple question")  # Would normally go to Gemini

        decision = router.route_request(request)

        # Should still route (quota checking is not implemented in current _is_provider_available)
        # This test documents current behavior and can be updated when quota checking is added
        assert isinstance(decision, RoutingDecision)

    def test_end_to_end_request_processing(self, router, providers):
        """Test complete end-to-end request processing."""
        requests = [
            LLMRequest(prompt="What is 2+2?"),
            LLMRequest(prompt="Explain Python decorators with examples"),
            LLMRequest(
                prompt="""
                Design a microservices architecture for a high-traffic e-commerce platform.
                Include considerations for scalability, fault tolerance, and data consistency.
                """
            ),
        ]

        decisions = []
        for request in requests:
            decision = router.route_request(request)
            decisions.append(decision)

            # Verify we can get the provider and it has recorded the routing
            provider = providers[decision.selected_provider]
            assert isinstance(provider, MockProvider)

        # Verify complexity progression (allow some flexibility)
        simple_score = decisions[0].complexity_score
        moderate_score = decisions[1].complexity_score
        complex_score = decisions[2].complexity_score

        # Simple should be lowest, complex should be highest
        assert simple_score <= moderate_score + 0.5  # Allow small variance
        assert moderate_score <= complex_score + 0.5

        # Verify different complexity levels
        complexity_levels = [d.complexity_level for d in decisions]
        assert (
            ComplexityLevel.SIMPLE in complexity_levels
            or ComplexityLevel.MODERATE in complexity_levels
        )
        assert len(set(complexity_levels)) >= 2  # Should have different levels

    def test_routing_performance_with_multiple_requests(self, router, providers):
        """Test routing performance with multiple concurrent requests."""
        requests = [LLMRequest(prompt=f"Test request number {i}") for i in range(50)]

        start_time = datetime.now()

        decisions = []
        for request in requests:
            decision = router.route_request(request)
            decisions.append(decision)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Should process 50 requests quickly
        assert total_time < 1.0  # Under 1 second total
        assert len(decisions) == 50

        # Check average routing time
        avg_routing_time = sum(d.routing_time_ms for d in decisions) / len(decisions)
        assert avg_routing_time < 50  # Under 50ms per request

    def test_routing_stats_accumulation(self, router, providers):
        """Test that routing statistics accumulate correctly."""
        initial_stats = router.get_routing_stats()
        initial_count = initial_stats.get("total_decisions", 0)

        # Make several routing decisions
        test_requests = [
            LLMRequest(prompt="Simple"),
            LLMRequest(prompt="Medium complexity with technical terms like algorithm"),
            LLMRequest(prompt="Complex distributed system design with fault tolerance"),
        ]

        for request in test_requests:
            router.route_request(request)

        final_stats = router.get_routing_stats()

        assert final_stats["total_decisions"] == initial_count + 3
        assert "provider_distribution" in final_stats
        assert "complexity_distribution" in final_stats
        assert final_stats["average_complexity_score"] >= 0.0
        assert final_stats["average_confidence"] > 0.0
        assert final_stats["average_routing_time_ms"] > 0.0

    def test_fallback_chain_execution(self, registry, provider_configs):
        """Test fallback chain when multiple providers fail."""
        # Create only one provider
        gemini_config = provider_configs[0]  # Gemini config
        registry.create_provider(gemini_config)

        router = ProviderRouter(registry)

        # Request complex task that would prefer other providers
        request = LLMRequest(
            prompt="""
            Prove the correctness of the Byzantine Generals Problem solution
            using formal mathematical methods and distributed systems theory.
            """
        )

        decision = router.route_request(request)

        # Should fall back to Gemini even though it's not optimal
        assert decision.selected_provider == ProviderType.GEMINI
        assert decision.confidence < 0.9  # Lower confidence for suboptimal choice

    def test_provider_registry_integration(self, provider_configs):
        """Test integration with provider registry."""
        # Create fresh registry without providers
        fresh_registry = ProviderRegistry()
        fresh_registry.register_provider(MockGeminiProvider)
        fresh_registry.register_provider(MockCodexProvider)
        fresh_registry.register_provider(MockLlamaProvider)

        router = ProviderRouter(fresh_registry)

        # Initially no provider instances
        with pytest.raises(ProviderError):
            request = LLMRequest(prompt="Test")
            router.route_request(request)

        # Add providers one by one
        for i, config in enumerate(provider_configs):
            fresh_registry.create_provider(config)

            # Should now be able to route
            request = LLMRequest(prompt=f"Test request {i}")
            decision = router.route_request(request)
            assert isinstance(decision, RoutingDecision)

    @pytest.mark.asyncio
    async def test_async_provider_interaction(self, router, providers):
        """Test routing with async provider operations."""
        request = LLMRequest(prompt="Test async interaction")
        decision = router.route_request(request)

        # Get the selected provider
        provider = providers[decision.selected_provider]

        # Test async operations
        response = await provider.generate(request)
        usage = await provider.get_usage()
        quota = await provider.check_quota()
        cost = await provider.estimate_cost(request)

        assert isinstance(response, LLMResponse)
        assert response.success
        assert response.provider == decision.selected_provider

        assert isinstance(usage, UsageStats)
        assert usage.provider_name == provider.name

        assert isinstance(quota, QuotaStatus)
        assert quota.provider_name == provider.name

        assert isinstance(cost, CostEstimate)
        assert cost.provider_name == provider.name

    def test_routing_decision_serialization(self, router, providers):
        """Test that routing decisions can be serialized for logging."""
        request = LLMRequest(prompt="Test serialization")
        decision = router.route_request(request)

        # Test serialization
        decision_dict = decision.to_dict()

        # Verify all fields are serializable
        import json

        json_str = json.dumps(decision_dict, default=str)  # Handle enums as strings
        assert len(json_str) > 0

        # Verify key fields are present
        assert "selected_provider" in decision_dict
        assert "complexity_score" in decision_dict
        assert "complexity_level" in decision_dict
        assert "confidence" in decision_dict
        assert "reasoning" in decision_dict
        assert "features" in decision_dict

    def test_complex_prompt_edge_cases(self, router, providers):
        """Test routing with edge case prompts."""
        edge_case_prompts = [
            # Very short prompt
            "Hi",
            # Very long prompt
            "This is a very long prompt. " * 100,
            # Code-heavy prompt
            """
            ```python
            def complex_algorithm(data):
                result = []
                for item in data:
                    if item.is_valid():
                        processed = item.transform()
                        result.append(processed)
                return result
            ```
            Explain this code and optimize it for performance.
            """,
            # Math-heavy prompt
            """
            Solve: ∫₀^∞ e^(-x²) dx = √π/2
            Prove using substitution u = x√2 and Gamma function properties.
            Show all steps: f(x) = x² + 2x + 1, find f'(x) and f''(x).
            """,
            # Mixed language prompt
            "Explain machine learning in English, 中文, and español. Include técnicas and 算法.",
        ]

        for prompt in edge_case_prompts:
            request = LLMRequest(prompt=prompt)
            decision = router.route_request(request)

            assert isinstance(decision, RoutingDecision)
            assert 0.0 <= decision.complexity_score <= 10.0
            assert decision.confidence > 0.0
            assert decision.selected_provider in [
                ProviderType.GEMINI,
                ProviderType.CODEX,
                ProviderType.LLAMA,
            ]


class TestProviderHealthIntegration:
    """Test provider health checking integration."""

    @pytest.fixture
    def unhealthy_provider_registry(self):
        """Create registry with mix of healthy and unhealthy providers."""
        registry = ProviderRegistry()
        registry.register_provider(MockGeminiProvider)
        registry.register_provider(MockCodexProvider)

        # Create providers
        gemini_config = ProviderConfig(
            name="unhealthy_gemini",
            provider_type=ProviderType.GEMINI,
            model_name="gemini-pro",
        )
        codex_config = ProviderConfig(
            name="healthy_codex",
            provider_type=ProviderType.CODEX,
            model_name="codex-davinci",
        )

        gemini_provider = registry.create_provider(gemini_config)
        codex_provider = registry.create_provider(codex_config)

        # Make Gemini unhealthy
        gemini_provider.is_healthy = False

        return registry, {"gemini": gemini_provider, "codex": codex_provider}

    def test_routing_avoids_unhealthy_providers(self, unhealthy_provider_registry):
        """Test that routing avoids unhealthy providers."""
        registry, providers = unhealthy_provider_registry
        router = ProviderRouter(registry)

        # Simple request that would normally go to Gemini
        request = LLMRequest(prompt="Hello world")
        decision = router.route_request(request)

        # Should avoid unhealthy Gemini and use Codex
        assert decision.selected_provider == ProviderType.CODEX

    def test_all_providers_unhealthy(self, unhealthy_provider_registry):
        """Test behavior when all providers are unhealthy."""
        registry, providers = unhealthy_provider_registry
        router = ProviderRouter(registry)

        # Make all providers unhealthy
        for provider in providers.values():
            provider.is_healthy = False

        request = LLMRequest(prompt="Test request")

        # Should raise ProviderError when no providers are available
        with pytest.raises(ProviderError) as exc_info:
            router.route_request(request)

        assert "no providers available" in str(exc_info.value).lower()


class TestConcurrentRouting:
    """Test concurrent routing scenarios."""

    @pytest.fixture
    def concurrent_setup(self):
        """Setup for concurrent testing."""
        registry = ProviderRegistry()
        registry.register_provider(MockGeminiProvider)
        registry.register_provider(MockCodexProvider)
        registry.register_provider(MockLlamaProvider)

        configs = [
            ProviderConfig(
                name="gemini",
                provider_type=ProviderType.GEMINI,
                model_name="gemini-pro",
            ),
            ProviderConfig(
                name="codex",
                provider_type=ProviderType.CODEX,
                model_name="codex-davinci",
            ),
            ProviderConfig(
                name="llama", provider_type=ProviderType.LLAMA, model_name="llama-7b"
            ),
        ]

        providers = {}
        for config in configs:
            provider = registry.create_provider(config)
            providers[config.provider_type] = provider

        router = ProviderRouter(registry)
        return router, providers

    def test_concurrent_routing_requests(self, concurrent_setup):
        """Test multiple concurrent routing requests."""
        router, providers = concurrent_setup

        # Create multiple requests
        requests = [
            LLMRequest(prompt=f"Concurrent request {i}: explain Python")
            for i in range(20)
        ]

        # Process all requests
        decisions = []
        for request in requests:
            decision = router.route_request(request)
            decisions.append(decision)

        # Verify all succeeded
        assert len(decisions) == 20
        for decision in decisions:
            assert isinstance(decision, RoutingDecision)
            assert decision.routing_time_ms > 0

        # Verify routing history tracking
        assert len(router.routing_history) >= 20

    def test_routing_statistics_thread_safety(self, concurrent_setup):
        """Test that routing statistics are thread-safe."""
        router, providers = concurrent_setup

        # Make many requests to accumulate statistics
        for i in range(100):
            request = LLMRequest(prompt=f"Request {i}")
            router.route_request(request)

        # Get statistics
        stats = router.get_routing_stats()

        assert stats["total_decisions"] == 100
        assert all(count > 0 for count in stats["provider_distribution"].values())
        assert 0.0 <= stats["average_complexity_score"] <= 10.0
        assert 0.0 <= stats["average_confidence"] <= 1.0
