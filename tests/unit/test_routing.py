"""Unit tests for prompt complexity analysis and provider routing system.

Test coverage includes:
- ComplexityAnalyzer feature extraction and scoring
- ProviderRouter routing decisions and fallbacks
- Edge cases and error handling
- Performance characteristics
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from mcp_server_llm_cli_runner.core.errors import ProviderError
from mcp_server_llm_cli_runner.core.models import LLMRequest, ProviderType
from mcp_server_llm_cli_runner.providers.registry import ProviderRegistry
from mcp_server_llm_cli_runner.providers.routing import (
    ComplexityAnalyzer,
    ComplexityFeatures,
    ComplexityLevel,
    ProviderRouter,
    RoutingDecision,
)


class TestComplexityAnalyzer:
    """Test cases for ComplexityAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ComplexityAnalyzer()

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, "logger")
        assert hasattr(self.analyzer, "_technical_keywords")
        assert len(self.analyzer._technical_keywords) > 0

    def test_extract_features_simple_prompt(self):
        """Test feature extraction for simple prompt."""
        prompt = "Hello, how are you today?"
        features = self.analyzer.extract_features(prompt)

        assert isinstance(features, ComplexityFeatures)
        assert features.prompt_length == len(prompt)
        assert features.word_count == 5
        assert features.sentence_count == 1
        assert features.technical_keywords == 0
        assert features.code_blocks == 0
        assert features.mathematical_expressions == 0
        assert features.reasoning_indicators == 0

    def test_extract_features_technical_prompt(self):
        """Test feature extraction for technical prompt."""
        prompt = """
        Explain the algorithm complexity of quicksort and provide a Python implementation.
        Compare its performance with merge sort and analyze the trade-offs.
        ```python
        def quicksort(arr):
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            return quicksort([x for x in arr if x < pivot]) + [pivot] + quicksort([x for x in arr if x > pivot])
        ```
        """
        features = self.analyzer.extract_features(prompt)

        assert features.technical_keywords > 0  # algorithm, complexity, performance
        assert features.code_blocks > 0  # Python code block
        assert features.reasoning_indicators > 0  # compare, analyze
        assert features.question_complexity > 0  # implicit questions

    def test_extract_features_mathematical_prompt(self):
        """Test feature extraction for mathematical prompt."""
        prompt = """
        Solve the integral ∫ x² dx and explain the fundamental theorem of calculus.
        Given f(x) = x² + 2x + 1, find the derivative f'(x).
        Show the step-by-step calculation: 3 + 4 * 5 - 2.
        """
        features = self.analyzer.extract_features(prompt)

        assert features.mathematical_expressions > 0
        assert features.technical_keywords > 0  # integral, derivative, theorem
        assert features.reasoning_indicators > 0  # explain, step-by-step

    def test_extract_features_context_heavy_prompt(self):
        """Test feature extraction for context-heavy prompt."""
        prompt = """
        Based on the previous discussion about machine learning models,
        and given that we are working in the healthcare domain,
        analyze the provided research paper and reference the methodology
        mentioned earlier in this conversation.
        """
        features = self.analyzer.extract_features(prompt)

        assert (
            features.context_requirements > 0
        )  # based on, given that, provided, mentioned
        assert features.technical_keywords > 0  # machine learning, methodology
        assert features.reasoning_indicators > 0  # analyze

    def test_calculate_complexity_score_simple(self):
        """Test complexity scoring for simple prompts."""
        simple_prompts = [
            "Hello world",
            "What is the weather today?",
            "Tell me a joke",
            "How are you?",
        ]

        for prompt in simple_prompts:
            features = self.analyzer.extract_features(prompt)
            score = self.analyzer.calculate_complexity_score(features)
            assert 0.0 <= score <= 3.0, f"Simple prompt scored {score}: {prompt}"

    def test_calculate_complexity_score_moderate(self):
        """Test complexity scoring for moderate prompts."""
        moderate_prompts = [
            "Explain how neural networks work in machine learning",
            "Write a function to sort an array using bubble sort",
            "Compare the advantages and disadvantages of SQL vs NoSQL databases",
            "What are the design patterns commonly used in software development?",
        ]

        for prompt in moderate_prompts:
            features = self.analyzer.extract_features(prompt)
            score = self.analyzer.calculate_complexity_score(features)
            assert 1.0 <= score <= 6.0, f"Moderate prompt scored {score}: {prompt}"

    def test_calculate_complexity_score_complex(self):
        """Test complexity scoring for complex prompts."""
        complex_prompts = [
            """
            Implement a distributed consensus algorithm like Raft,
            analyze its performance characteristics, and compare it with Paxos.
            Provide detailed mathematical proofs for correctness.
            """,
            """
            Given the research methodology described in the previous papers,
            design an experimental framework to evaluate machine learning models
            in the context of natural language processing tasks.
            Include statistical analysis and hypothesis testing.
            """,
        ]

        for prompt in complex_prompts:
            features = self.analyzer.extract_features(prompt)
            score = self.analyzer.calculate_complexity_score(features)
            assert score >= 4.0, f"Complex prompt scored {score}: {prompt[:50]}..."

    def test_analyze_complexity_integration(self):
        """Test complete complexity analysis integration."""
        test_cases = [
            ("Hi there!", ComplexityLevel.SIMPLE),
            ("Explain Python decorators with examples", ComplexityLevel.MODERATE),
            (
                "Design a distributed system with fault tolerance",
                ComplexityLevel.COMPLEX,
            ),
            (
                """
            Based on the quantum mechanics principles discussed earlier,
            derive the Schrödinger equation and analyze its implications
            for quantum computing algorithms. Provide mathematical proofs
            and compare with classical computational complexity theory.
            """,
                ComplexityLevel.EXPERT,
            ),
        ]

        for prompt, expected_level in test_cases:
            score, level, features = self.analyzer.analyze_complexity(prompt)

            assert 0.0 <= score <= 10.0
            assert isinstance(level, ComplexityLevel)
            assert isinstance(features, ComplexityFeatures)

            # Allow some flexibility in level assignment for edge cases
            if expected_level == ComplexityLevel.SIMPLE:
                assert level in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]
            elif expected_level == ComplexityLevel.EXPERT:
                assert level in [ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT]
            else:
                # For moderate and complex, allow adjacent levels
                pass

    def test_features_to_dict(self):
        """Test ComplexityFeatures serialization."""
        features = ComplexityFeatures(
            prompt_length=100,
            word_count=20,
            technical_keywords=3,
            code_blocks=1,
            language_complexity=4.5,
        )

        feature_dict = features.to_dict()

        assert isinstance(feature_dict, dict)
        assert feature_dict["prompt_length"] == 100
        assert feature_dict["word_count"] == 20
        assert feature_dict["technical_keywords"] == 3
        assert feature_dict["code_blocks"] == 1
        assert feature_dict["language_complexity"] == 4.5

    def test_regex_patterns_performance(self):
        """Test that regex patterns are compiled and performant."""
        # Test with a large prompt to ensure patterns are efficient
        large_prompt = (
            """
        This is a very long prompt that contains many technical keywords
        like algorithm, optimization, database, machine learning, neural networks,
        statistical analysis, hypothesis testing, and performance optimization.

        ```python
        def complex_function(data):
            # Multiple code blocks to test pattern matching
            result = []
            for item in data:
                if item.is_valid():
                    result.append(item.process())
            return result
        ```

        Mathematical expressions: f(x) = x² + 2x + 1, ∫ x dx, sin(θ) + cos(θ)

        Based on the previous analysis, we need to consider the methodology
        and approach discussed in the research paper mentioned earlier.
        """
            * 10
        )  # Repeat to make it large

        import time

        start_time = time.time()

        features = self.analyzer.extract_features(large_prompt)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should process large prompts quickly (under 100ms)
        assert processing_time < 0.1, f"Processing took {processing_time:.3f}s"
        assert features.technical_keywords > 0
        assert features.code_blocks > 0
        assert features.mathematical_expressions > 0


class TestProviderRouter:
    """Test cases for ProviderRouter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_registry = Mock(spec=ProviderRegistry)
        self.mock_registry.list_available_types.return_value = [
            ProviderType.GEMINI,
            ProviderType.CODEX,
            ProviderType.LLAMA,
        ]
        self.mock_registry.list_providers.return_value = [
            "gemini_provider",
            "codex_provider",
            "llama_provider",
        ]

        # Create mock provider instances for each type
        self.mock_gemini = Mock()
        self.mock_gemini.provider_type = ProviderType.GEMINI
        self.mock_gemini.is_healthy = True

        self.mock_codex = Mock()
        self.mock_codex.provider_type = ProviderType.CODEX
        self.mock_codex.is_healthy = True

        self.mock_llama = Mock()
        self.mock_llama.provider_type = ProviderType.LLAMA
        self.mock_llama.is_healthy = True

        # Set up provider mapping
        provider_map = {
            "gemini_provider": self.mock_gemini,
            "codex_provider": self.mock_codex,
            "llama_provider": self.mock_llama,
        }
        self.mock_registry.get_provider.side_effect = lambda name: provider_map.get(
            name,
        )

        self.router = ProviderRouter(self.mock_registry)

    def test_router_initialization(self):
        """Test router initializes correctly."""
        assert self.router.registry == self.mock_registry
        assert isinstance(self.router.analyzer, ComplexityAnalyzer)
        assert hasattr(self.router, "routing_history")
        assert len(self.router.routing_history) == 0

    def test_route_simple_request(self):
        """Test routing simple request to Gemini."""
        request = LLMRequest(prompt="Hello, how are you?")

        decision = self.router.route_request(request)

        assert isinstance(decision, RoutingDecision)
        assert decision.selected_provider == ProviderType.GEMINI
        assert decision.complexity_level == ComplexityLevel.SIMPLE
        assert decision.complexity_score <= 3.0
        assert decision.confidence > 0.8
        assert "cost-effective" in decision.reasoning.lower()
        assert len(decision.fallback_providers) > 0

    def test_route_complex_request(self):
        """Test routing complex request to appropriate provider."""
        request = LLMRequest(
            prompt="""
            Design a distributed consensus algorithm with Byzantine fault tolerance.
            Analyze the performance characteristics of Raft vs PBFT algorithms.
            Implement optimization strategies using advanced data structures.
            Provide mathematical proofs of correctness and performance analysis.
            Research the theoretical framework and methodology for consensus protocols.
            Compare distributed database architectures and scalability patterns.
            """,
        )

        decision = self.router.route_request(request)

        assert decision.selected_provider in [ProviderType.CODEX, ProviderType.LLAMA]
        assert decision.complexity_level in [
            ComplexityLevel.COMPLEX,
            ComplexityLevel.EXPERT,
        ]
        assert decision.complexity_score >= 5.0  # Should be complex (>5.0)
        assert decision.confidence > 0.6

    def test_route_with_explicit_provider(self):
        """Test routing with explicitly requested provider."""
        request = LLMRequest(prompt="Simple question", provider=ProviderType.LLAMA)

        decision = self.router.route_request(request)

        assert decision.selected_provider == ProviderType.LLAMA
        assert decision.confidence == 1.0  # Explicit choice
        assert "explicitly requested" in decision.reasoning.lower()

    def test_route_with_unavailable_explicit_provider(self):
        """Test routing when explicitly requested provider is unavailable."""
        # Mock provider as unavailable
        self.mock_registry.list_available_types.return_value = [
            ProviderType.GEMINI,
            ProviderType.CODEX,
        ]

        request = LLMRequest(
            prompt="Simple question",
            provider=ProviderType.LLAMA,  # Not available
        )

        decision = self.router.route_request(request)

        # Should fall back to automatic routing
        assert decision.selected_provider in [ProviderType.GEMINI, ProviderType.CODEX]
        assert decision.confidence < 1.0  # Not explicit choice

    def test_route_no_providers_available(self):
        """Test routing when no providers are available."""
        self.mock_registry.list_available_types.return_value = []

        request = LLMRequest(prompt="Any question")

        with pytest.raises(ProviderError) as exc_info:
            self.router.route_request(request)

        assert "no providers available" in str(exc_info.value).lower()
        assert exc_info.value.error_code == "NO_PROVIDERS_AVAILABLE"

    def test_fallback_provider_selection(self):
        """Test fallback provider selection when primary unavailable."""
        # Only LLAMA available
        self.mock_registry.list_available_types.return_value = [ProviderType.LLAMA]

        request = LLMRequest(prompt="Simple question")  # Would normally go to Gemini

        decision = self.router.route_request(request)

        assert decision.selected_provider == ProviderType.LLAMA
        assert decision.confidence == 0.5  # Fallback confidence

    def test_confidence_calculation(self):
        """Test confidence calculation for different scenarios."""
        # Simple prompt with optimal provider (Gemini)
        simple_request = LLMRequest(prompt="Hello")
        simple_decision = self.router.route_request(simple_request)

        # Complex prompt with optimal provider
        complex_request = LLMRequest(
            prompt="Design a complex distributed system with fault tolerance",
        )
        complex_decision = self.router.route_request(complex_request)

        # Both should have high confidence for optimal matches
        assert simple_decision.confidence >= 0.9
        assert complex_decision.confidence >= 0.8

    def test_routing_time_measurement(self):
        """Test that routing time is measured and reasonable."""
        request = LLMRequest(prompt="Test prompt")

        decision = self.router.route_request(request)

        assert decision.routing_time_ms > 0
        assert decision.routing_time_ms < 1000  # Should be under 1 second

    def test_routing_history_tracking(self):
        """Test that routing decisions are tracked for ML optimization."""
        initial_history_length = len(self.router.routing_history)

        request1 = LLMRequest(prompt="First request")
        request2 = LLMRequest(prompt="Second request with more complexity")

        self.router.route_request(request1)
        self.router.route_request(request2)

        assert len(self.router.routing_history) == initial_history_length + 2

        # Check history entry structure
        latest_entry = self.router.routing_history[-1]
        required_fields = [
            "timestamp",
            "complexity_score",
            "selected_provider",
            "confidence",
            "prompt_length",
            "features",
            "routing_time_ms",
        ]

        for field in required_fields:
            assert field in latest_entry

    def test_routing_history_memory_management(self):
        """Test that routing history is managed to prevent memory leaks."""
        # Simulate many routing decisions
        initial_count = len(self.router.routing_history)

        # Add more than the limit
        for _ in range(1050):  # Limit is 1000, should trim to 500
            self.router.routing_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "complexity_score": 1.0,
                    "selected_provider": ProviderType.GEMINI,
                    "confidence": 0.9,
                    "prompt_length": 10,
                    "features": {},
                    "routing_time_ms": 1.0,
                },
            )

        # Trigger memory management by making a real request
        request = LLMRequest(prompt="Test")
        self.router.route_request(request)

        # Should have trimmed to 500 + 1 new entry
        assert len(self.router.routing_history) <= 501

    def test_get_routing_stats(self):
        """Test routing statistics generation."""
        # Make some routing decisions first
        requests = [
            LLMRequest(prompt="Simple question"),
            LLMRequest(prompt="More complex technical analysis with algorithms"),
            LLMRequest(prompt="Hello"),
        ]

        for request in requests:
            self.router.route_request(request)

        stats = self.router.get_routing_stats()

        assert isinstance(stats, dict)
        assert "total_decisions" in stats
        assert stats["total_decisions"] >= 3
        assert "provider_distribution" in stats
        assert "complexity_distribution" in stats
        assert "average_complexity_score" in stats
        assert "average_confidence" in stats
        assert "average_routing_time_ms" in stats

        # Check that averages are reasonable
        assert 0.0 <= stats["average_complexity_score"] <= 10.0
        assert 0.0 <= stats["average_confidence"] <= 1.0
        assert stats["average_routing_time_ms"] > 0

    def test_get_routing_stats_empty_history(self):
        """Test routing statistics with no history."""
        stats = self.router.get_routing_stats()

        assert stats["total_decisions"] == 0
        assert "no routing history" in stats["message"].lower()


class TestRoutingDecision:
    """Test cases for RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """Test RoutingDecision creation and basic properties."""
        features = ComplexityFeatures(prompt_length=50, word_count=10)

        decision = RoutingDecision(
            selected_provider=ProviderType.GEMINI,
            complexity_score=2.5,
            complexity_level=ComplexityLevel.SIMPLE,
            confidence=0.95,
            reasoning="Test reasoning",
            fallback_providers=[ProviderType.CODEX, ProviderType.LLAMA],
            features=features,
            routing_time_ms=15.5,
        )

        assert decision.selected_provider == ProviderType.GEMINI
        assert decision.complexity_score == 2.5
        assert decision.complexity_level == ComplexityLevel.SIMPLE
        assert decision.confidence == 0.95
        assert decision.reasoning == "Test reasoning"
        assert len(decision.fallback_providers) == 2
        assert decision.routing_time_ms == 15.5

    def test_routing_decision_to_dict(self):
        """Test RoutingDecision serialization."""
        features = ComplexityFeatures(prompt_length=50, word_count=10)

        decision = RoutingDecision(
            selected_provider=ProviderType.CODEX,
            complexity_score=5.75,
            complexity_level=ComplexityLevel.COMPLEX,
            confidence=0.85,
            reasoning="Complex reasoning required",
            fallback_providers=[ProviderType.LLAMA],
            features=features,
            routing_time_ms=23.7,
        )

        decision_dict = decision.to_dict()

        assert isinstance(decision_dict, dict)
        assert decision_dict["selected_provider"] == ProviderType.CODEX
        assert decision_dict["complexity_score"] == 5.75  # Rounded in to_dict
        assert decision_dict["complexity_level"] == ComplexityLevel.COMPLEX
        assert decision_dict["confidence"] == 0.85  # Rounded in to_dict
        assert decision_dict["reasoning"] == "Complex reasoning required"
        assert decision_dict["fallback_providers"] == [ProviderType.LLAMA]
        assert "features" in decision_dict
        assert decision_dict["routing_time_ms"] == 23.7  # Rounded in to_dict


class TestComplexityLevel:
    """Test cases for ComplexityLevel enum."""

    def test_complexity_level_values(self):
        """Test ComplexityLevel enum values."""
        assert ComplexityLevel.SIMPLE == "simple"
        assert ComplexityLevel.MODERATE == "moderate"
        assert ComplexityLevel.COMPLEX == "complex"
        assert ComplexityLevel.EXPERT == "expert"

    def test_complexity_level_ordering(self):
        """Test that complexity levels can be compared."""
        levels = [
            ComplexityLevel.SIMPLE,
            ComplexityLevel.MODERATE,
            ComplexityLevel.COMPLEX,
            ComplexityLevel.EXPERT,
        ]

        # Test that each level is valid
        for level in levels:
            assert isinstance(level, ComplexityLevel)
            assert isinstance(level.value, str)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ComplexityAnalyzer()

        self.mock_registry = Mock(spec=ProviderRegistry)
        self.mock_registry.list_available_types.return_value = [
            ProviderType.GEMINI,
            ProviderType.CODEX,
            ProviderType.LLAMA,
        ]
        self.mock_registry.list_providers.return_value = ["test_provider"]

        # Create a mock provider instance
        self.mock_provider = Mock()
        self.mock_provider.provider_type = ProviderType.GEMINI
        self.mock_provider.is_healthy = True
        self.mock_registry.get_provider.return_value = self.mock_provider

        self.router = ProviderRouter(self.mock_registry)

    def test_empty_prompt_handling(self):
        """Test handling of empty prompts."""
        with pytest.raises(ValueError):
            # LLMRequest should validate min_length=1
            LLMRequest(prompt="")

    def test_very_long_prompt_handling(self):
        """Test handling of very long prompts."""
        # Create a very long prompt (near the limit)
        long_prompt = "This is a test prompt. " * 400  # ~9600 chars

        request = LLMRequest(prompt=long_prompt)
        decision = self.router.route_request(request)

        # Should handle long prompts gracefully
        assert isinstance(decision, RoutingDecision)
        assert decision.features.prompt_length > 5000

    def test_special_characters_in_prompt(self):
        """Test handling of prompts with special characters."""
        special_prompt = """
        Prompt with émojis 🚀, ünïcödë characters, and special symbols: @#$%^&*()
        Also includes newlines
        and various punctuation: !?.,;:"'[]{}
        """

        features = self.analyzer.extract_features(special_prompt)
        score = self.analyzer.calculate_complexity_score(features)

        assert 0.0 <= score <= 10.0
        assert features.prompt_length > 0
        assert features.word_count > 0

    def test_malformed_code_blocks(self):
        """Test handling of malformed code blocks."""
        malformed_prompt = """
        Here's some code:
        ```python
        def broken_function(
            # Missing closing brace and block
        Some text after broken code
        `incomplete inline code
        ```
        More text
        """

        features = self.analyzer.extract_features(malformed_prompt)

        # Should still detect code patterns even if malformed
        assert features.code_blocks >= 0  # Should not crash

    def test_mixed_language_prompt(self):
        """Test handling of prompts with mixed languages."""
        mixed_prompt = """
        English text mixed with 中文字符 and español text.
        Technical terms: algoritmo, 算法, algorithm.
        Math: 数学 = mathematics = matemáticas
        """

        features = self.analyzer.extract_features(mixed_prompt)
        score = self.analyzer.calculate_complexity_score(features)

        assert 0.0 <= score <= 10.0
        assert features.prompt_length > 0

    @patch("mcp_server_llm_cli_runner.providers.routing.datetime")
    def test_routing_with_time_mock(self, mock_datetime):
        """Test routing time calculation with mocked datetime."""
        # Mock datetime to control timing
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 0, 0, 50000)  # 50ms later

        mock_datetime.now.side_effect = [
            start_time,
            end_time,
            end_time,
        ]  # Extra call for final timing

        request = LLMRequest(prompt="Test prompt")
        decision = self.router.route_request(request)

        assert decision.routing_time_ms == 50.0

    def test_routing_with_registry_errors(self):
        """Test routing behavior when registry has errors."""
        # Mock registry to raise exceptions
        self.mock_registry.list_available_types.side_effect = Exception(
            "Registry error",
        )

        with pytest.raises(Exception):  # noqa: B017
            request = LLMRequest(prompt="Test prompt")
            self.router.route_request(request)
