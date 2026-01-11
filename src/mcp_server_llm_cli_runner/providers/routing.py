"""Intelligent prompt complexity analysis and provider routing system.

This module implements a sophisticated routing algorithm that analyzes prompt complexity
and determines optimal provider selection based on scoring metrics.

Key classes:
    ComplexityAnalyzer: Analyzes prompt complexity on 0-10 scale
    ProviderRouter: Routes requests to optimal providers
    RoutingDecision: Result of routing analysis

Example:
    >>> analyzer = ComplexityAnalyzer()
    >>> score = analyzer.analyze_complexity("Write a simple hello world program")
    >>> # score: 2.5 (low complexity)
    >>>
    >>> router = ProviderRouter(registry)
    >>> decision = router.route_request(request)
    >>> # decision.selected_provider: "gemini" (free tier)

"""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from mcp_server_llm_cli_runner.core.errors import ProviderError
from mcp_server_llm_cli_runner.core.models import LLMRequest, ProviderType
from mcp_server_llm_cli_runner.utils.logging import StructuredLogger

from .registry import ProviderRegistry


class ComplexityLevel(str, Enum):
    """Complexity level classification."""

    SIMPLE = "simple"  # 0-3: Basic queries, simple tasks
    MODERATE = "moderate"  # 3-5: Medium complexity, requires reasoning
    COMPLEX = "complex"  # 5-7: Advanced tasks, specialized knowledge
    EXPERT = "expert"  # 7-10: Highly complex, multi-step reasoning


@dataclass
class ComplexityFeatures:
    """Features extracted from prompt for complexity analysis."""

    prompt_length: int = 0
    word_count: int = 0
    sentence_count: int = 0
    technical_keywords: int = 0
    code_blocks: int = 0
    mathematical_expressions: int = 0
    language_complexity: float = 0.0
    context_requirements: int = 0
    question_complexity: int = 0
    reasoning_indicators: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "prompt_length": self.prompt_length,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "technical_keywords": self.technical_keywords,
            "code_blocks": self.code_blocks,
            "mathematical_expressions": self.mathematical_expressions,
            "language_complexity": round(self.language_complexity, 2),
            "context_requirements": self.context_requirements,
            "question_complexity": self.question_complexity,
            "reasoning_indicators": self.reasoning_indicators,
        }


@dataclass
class RoutingDecision:
    """Result of routing analysis with provider selection and reasoning."""

    selected_provider: ProviderType
    complexity_score: float
    complexity_level: ComplexityLevel
    confidence: float
    reasoning: str
    fallback_providers: list[ProviderType]
    features: ComplexityFeatures
    routing_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "selected_provider": self.selected_provider,
            "complexity_score": round(self.complexity_score, 2),
            "complexity_level": self.complexity_level,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "fallback_providers": self.fallback_providers,
            "features": self.features.to_dict(),
            "routing_time_ms": round(self.routing_time_ms, 2),
        }


class ComplexityAnalyzer:
    """Analyzes prompt complexity using multiple scoring criteria.

    The analyzer uses a weighted scoring system based on various features:
    - Prompt length and structure (weight: 1.0)
    - Technical keyword density (weight: 2.0)
    - Code presence and complexity (weight: 2.5)
    - Mathematical expressions (weight: 2.0)
    - Language complexity patterns (weight: 1.5)
    - Context and reasoning requirements (weight: 3.0)

    Final score is normalized to 0-10 scale.
    """

    # Technical keywords that indicate complexity
    TECHNICAL_KEYWORDS = {
        "programming": [
            "algorithm",
            "optimization",
            "data structure",
            "complexity",
            "performance",
            "scalability",
            "architecture",
            "design pattern",
            "framework",
            "library",
            "api",
            "database",
            "sql",
            "nosql",
        ],
        "science": [
            "analysis",
            "hypothesis",
            "methodology",
            "statistical",
            "empirical",
            "experimental",
            "correlation",
            "causation",
            "regression",
            "distribution",
            "probability",
            "inference",
        ],
        "math": [
            "equation",
            "derivative",
            "integral",
            "matrix",
            "vector",
            "theorem",
            "proof",
            "formula",
            "calculation",
            "optimization",
            "linear algebra",
            "calculus",
            "statistics",
            "probability",
        ],
        "academic": [
            "research",
            "literature",
            "citation",
            "methodology",
            "bibliography",
            "peer review",
            "hypothesis",
            "abstract",
            "conclusion",
            "references",
            "scholarly",
        ],
    }

    # Patterns that indicate reasoning requirements
    REASONING_PATTERNS = [
        r"\b(?:analyze|compare|contrast|evaluate|explain why|justify|reason|deduce|infer)\b",
        r"\b(?:pros and cons|advantages and disadvantages|trade-offs?)\b",
        r"\b(?:cause and effect|root cause|implications|consequences)\b",
        r"\b(?:step by step|walkthrough|methodology|approach)\b",
        r"\b(?:complex|sophisticated|advanced|intricate|nuanced)\b",
        r"\b(?:explain|describe|show|demonstrate|illustrate)\b",
    ]

    # Math expression patterns
    MATH_PATTERNS = [
        r"\\[a-zA-Z]+\{[^}]*\}",  # LaTeX commands
        r"\$[^$]+\$",  # Inline math
        r"\\\([^)]*\\\)",  # LaTeX inline
        r"\\\[[^\]]*\\\]",  # LaTeX display
        r"\b\d+\s*[+\-*/^]\s*\d+",  # Simple arithmetic
        r"[∑∏∫∂∇]+",  # Math symbols
        r"\b(?:sin|cos|tan|log|ln|exp|sqrt)\(",  # Functions
    ]

    # Code block patterns
    CODE_PATTERNS = [
        r"```[\s\S]*?```",  # Markdown code blocks
        r"`[^`]+`",  # Inline code
        r"^\s{4,}.*$",  # Indented code (multiline)
        r"<code>[\s\S]*?</code>",  # HTML code tags
    ]

    # Context requirement indicators
    CONTEXT_INDICATORS = [
        r"\b(?:based on|given that|assuming|provided that|in the context of)\b",
        r"\b(?:reference|document|article|paper|source|citation)\b",
        r"\b(?:previous|earlier|mentioned|discussed|above|below)\b",
        r"\b(?:domain|field|area|industry|sector|specialty)\b",
    ]

    def __init__(self) -> None:
        """Initialize complexity analyzer."""
        self.logger = StructuredLogger(__name__)

        # Compile regex patterns for efficiency
        self._reasoning_regex = re.compile(
            "|".join(self.REASONING_PATTERNS),
            re.IGNORECASE,
        )
        self._math_regex = re.compile("|".join(self.MATH_PATTERNS), re.IGNORECASE)
        self._code_regex = re.compile("|".join(self.CODE_PATTERNS), re.MULTILINE)
        self._context_regex = re.compile(
            "|".join(self.CONTEXT_INDICATORS),
            re.IGNORECASE,
        )

        # Create technical keyword set for fast lookup
        self._technical_keywords = set()
        for category_words in self.TECHNICAL_KEYWORDS.values():
            self._technical_keywords.update(word.lower() for word in category_words)

    def extract_features(self, prompt: str) -> ComplexityFeatures:
        """Extract complexity features from prompt.

        Args:
            prompt: Text prompt to analyze

        Returns:
            ComplexityFeatures object with extracted metrics

        """
        features = ComplexityFeatures()

        # Basic text metrics
        features.prompt_length = len(prompt)
        words = prompt.lower().split()
        features.word_count = len(words)
        features.sentence_count = len(re.findall(r"[.!?]+", prompt))

        # Technical keyword analysis
        word_set = set(words)
        features.technical_keywords = len(
            word_set.intersection(self._technical_keywords),
        )

        # Code analysis
        code_matches = self._code_regex.findall(prompt)
        features.code_blocks = len(code_matches)

        # Mathematical expression analysis
        math_matches = self._math_regex.findall(prompt)
        features.mathematical_expressions = len(math_matches)

        # Language complexity (average word length, vocabulary diversity)
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            unique_words = len(set(words))
            vocabulary_diversity = unique_words / len(words)
            features.language_complexity = (avg_word_length * 0.3) + (
                vocabulary_diversity * 7.0
            )

        # Context requirements
        context_matches = self._context_regex.findall(prompt)
        features.context_requirements = len(context_matches)

        # Question complexity (multiple questions, compound questions)
        question_marks = prompt.count("?")
        question_words = len(
            re.findall(
                r"\b(?:what|how|why|when|where|which|who)\b",
                prompt,
                re.IGNORECASE,
            ),
        )
        implicit_questions = len(
            re.findall(
                r"\b(?:explain|describe|show|demonstrate|tell me)\b",
                prompt,
                re.IGNORECASE,
            ),
        )
        features.question_complexity = (
            question_marks + question_words + implicit_questions
        )

        # Reasoning indicators
        reasoning_matches = self._reasoning_regex.findall(prompt)
        features.reasoning_indicators = len(reasoning_matches)

        return features

    def calculate_complexity_score(self, features: ComplexityFeatures) -> float:
        """Calculate complexity score from extracted features.

        Args:
            features: Extracted complexity features

        Returns:
            Complexity score between 0.0 and 10.0

        """
        score = 0.0

        # Length-based scoring (0-2 points)
        if features.prompt_length < 50:
            length_score = 0.5
        elif features.prompt_length < 200:
            length_score = 1.0
        elif features.prompt_length < 500:
            length_score = 1.5
        else:
            length_score = 2.0

        score += length_score * 1.0  # Weight: 1.0

        # Technical keyword density (0-2 points)
        if features.word_count > 0:
            keyword_density = features.technical_keywords / features.word_count
            keyword_score = min(2.0, keyword_density * 20)  # Scale to 0-2
            score += keyword_score * 2.0  # Weight: 2.0

        # Code complexity (0-2 points)
        code_score = min(2.0, features.code_blocks * 0.5)
        score += code_score * 2.5  # Weight: 2.5

        # Mathematical complexity (0-2 points)
        math_score = min(2.0, features.mathematical_expressions * 0.3)
        score += math_score * 2.0  # Weight: 2.0

        # Language complexity (0-2 points)
        lang_score = min(2.0, features.language_complexity / 5.0)
        score += lang_score * 1.5  # Weight: 1.5

        # Context and reasoning (0-2 points)
        context_score = min(1.0, features.context_requirements * 0.2)
        reasoning_score = min(1.0, features.reasoning_indicators * 0.3)
        score += (context_score + reasoning_score) * 3.0  # Weight: 3.0

        # Question complexity (0-1 point)
        question_score = min(1.0, features.question_complexity * 0.2)
        score += question_score * 1.0  # Weight: 1.0

        # Normalize to 0-10 scale (max possible score is ~23)
        return min(10.0, (score / 23.0) * 10.0)

    def analyze_complexity(
        self,
        prompt: str,
    ) -> tuple[float, ComplexityLevel, ComplexityFeatures]:
        """Analyze prompt complexity and return score, level, and features.

        Args:
            prompt: Text prompt to analyze

        Returns:
            Tuple of (complexity_score, complexity_level, features)

        """
        features = self.extract_features(prompt)
        score = self.calculate_complexity_score(features)

        # Determine complexity level
        if score <= 3.0:
            level = ComplexityLevel.SIMPLE
        elif score <= 5.0:
            level = ComplexityLevel.MODERATE
        elif score <= 7.0:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.EXPERT

        self.logger.debug(
            "Complexity analysis completed",
            extra={
                "prompt_length": len(prompt),
                "complexity_score": round(score, 2),
                "complexity_level": level,
                "features": features.to_dict(),
            },
        )

        return score, level, features


class ProviderRouter:
    """Intelligent routing system for selecting optimal LLM providers.

    Routes requests based on complexity analysis, provider capabilities,
    availability, and cost optimization strategies.
    """

    # Routing rules based on complexity score
    ROUTING_RULES = {
        ComplexityLevel.SIMPLE: {
            "primary": [ProviderType.GEMINI],
            "description": "Simple tasks → Free tier (Gemini)",
        },
        ComplexityLevel.MODERATE: {
            "primary": [ProviderType.GEMINI, ProviderType.CODEX],
            "description": "Moderate complexity → Batch API or cached responses",
        },
        ComplexityLevel.COMPLEX: {
            "primary": [ProviderType.CODEX, ProviderType.LLAMA],
            "description": "Complex tasks → Premium models",
        },
        ComplexityLevel.EXPERT: {
            "primary": [ProviderType.CODEX, ProviderType.LLAMA],
            "description": "Expert-level → Highest capability models",
        },
    }

    def __init__(self, registry: ProviderRegistry) -> None:
        """Initialize provider router.

        Args:
            registry: Provider registry for availability checking

        """
        self.registry = registry
        self.analyzer = ComplexityAnalyzer()
        self.logger = StructuredLogger(__name__)

        # Performance tracking for ML optimization
        self.routing_history: list[dict[str, Any]] = []

    def route_request(self, request: LLMRequest) -> RoutingDecision:
        """Route request to optimal provider based on complexity analysis.

        Args:
            request: LLM request to route

        Returns:
            RoutingDecision with selected provider and reasoning

        Raises:
            ProviderError: If no suitable provider is available

        """
        start_time = datetime.now()

        # Override provider if explicitly specified and available
        if request.provider:
            provider_type = ProviderType(request.provider)
            if self._is_provider_available(provider_type):
                routing_time = (datetime.now() - start_time).total_seconds() * 1000

                # Still analyze complexity for logging/ML
                score, level, features = self.analyzer.analyze_complexity(
                    request.prompt,
                )

                decision = RoutingDecision(
                    selected_provider=provider_type,
                    complexity_score=score,
                    complexity_level=level,
                    confidence=1.0,  # Explicit choice
                    reasoning=f"Explicitly requested provider: {request.provider}",
                    fallback_providers=self._get_fallback_providers(provider_type),
                    features=features,
                    routing_time_ms=routing_time,
                )

                self._log_routing_decision(decision, request)
                return decision
            self.logger.warning(
                "Requested provider unavailable, falling back to automatic routing",
                extra={"requested_provider": request.provider},
            )

        # Analyze prompt complexity
        score, level, features = self.analyzer.analyze_complexity(request.prompt)

        # Get provider selection based on complexity
        primary_providers = self.ROUTING_RULES[level]["primary"]
        selected_provider = None
        confidence = 0.0

        # Select first available provider from primary list
        for provider in primary_providers:
            if self._is_provider_available(provider):
                selected_provider = provider
                confidence = self._calculate_confidence(score, level, provider)
                break

        # If no primary provider available, try fallbacks
        if not selected_provider:
            all_providers = [
                ProviderType.GEMINI,
                ProviderType.CODEX,
                ProviderType.LLAMA,
            ]
            for provider in all_providers:
                if self._is_provider_available(provider):
                    selected_provider = provider
                    confidence = 0.5  # Lower confidence for fallback
                    break

        if not selected_provider:
            msg = "No providers available for routing"
            raise ProviderError(
                msg,
                provider="router",
                error_code="NO_PROVIDERS_AVAILABLE",
                context={
                    "complexity_score": score,
                    "complexity_level": level,
                    "requested_provider": request.provider,
                },
            )

        # Calculate routing time
        routing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Create routing decision
        decision = RoutingDecision(
            selected_provider=selected_provider,
            complexity_score=score,
            complexity_level=level,
            confidence=confidence,
            reasoning=self._generate_reasoning(score, level, selected_provider),
            fallback_providers=self._get_fallback_providers(selected_provider),
            features=features,
            routing_time_ms=routing_time,
        )

        # Log decision and track for ML
        self._log_routing_decision(decision, request)
        self._track_routing_decision(decision, request)

        return decision

    def _is_provider_available(self, provider: ProviderType) -> bool:
        """Check if provider is available and healthy.

        Args:
            provider: Provider to check

        Returns:
            True if provider is available

        """
        try:
            # Check if provider type is registered
            if provider not in self.registry.list_available_types():
                return False

            # Try to get provider instances and check health
            provider_instances = []
            for name in self.registry.list_providers():
                instance = self.registry.get_provider(name)
                if instance and instance.provider_type == provider:
                    provider_instances.append(instance)

            if not provider_instances:
                return False

            # Check if at least one instance is healthy
            for instance in provider_instances:
                # Check if provider has health indicator
                if hasattr(instance, "is_healthy"):
                    if instance.is_healthy:
                        return True
                else:
                    # If no health indicator, assume healthy
                    return True

            # No healthy instances found
            return False

        except Exception as e:
            self.logger.exception(
                "Error checking provider availability",
                extra={"provider": provider, "error": str(e)},
            )
            return False

    def _calculate_confidence(
        self,
        score: float,
        level: ComplexityLevel,
        provider: ProviderType,
    ) -> float:
        """Calculate confidence in provider selection.

        Args:
            score: Complexity score
            level: Complexity level
            provider: Selected provider

        Returns:
            Confidence score between 0.0 and 1.0

        """
        base_confidence = 0.8

        # Adjust based on provider-complexity match
        optimal_providers = self.ROUTING_RULES[level]["primary"]
        if provider in optimal_providers:
            if provider == optimal_providers[0]:
                base_confidence = 0.95  # First choice
            else:
                base_confidence = 0.85  # Secondary choice
        else:
            base_confidence = 0.6  # Fallback provider

        # Adjust for edge cases (scores near boundaries)
        if (
            (level == ComplexityLevel.SIMPLE and score > 2.5)
            or (level == ComplexityLevel.MODERATE and (score < 3.2 or score > 4.8))
            or (level == ComplexityLevel.COMPLEX and (score < 5.2 or score > 6.8))
            or (level == ComplexityLevel.EXPERT and score < 7.2)
        ):
            base_confidence *= 0.9

        return min(1.0, base_confidence)

    def _generate_reasoning(
        self,
        score: float,
        level: ComplexityLevel,
        provider: ProviderType,
    ) -> str:
        """Generate human-readable reasoning for routing decision.

        Args:
            score: Complexity score
            level: Complexity level
            provider: Selected provider

        Returns:
            Reasoning string

        """
        base_reasoning = self.ROUTING_RULES[level]["description"]

        provider_rationale = {
            ProviderType.GEMINI: "cost-effective for basic tasks",
            ProviderType.CODEX: "balanced performance and cost",
            ProviderType.LLAMA: "high performance for complex reasoning",
        }

        return f"{base_reasoning}. Selected {provider} ({provider_rationale[provider]}) for complexity score {score:.1f}."

    def _get_fallback_providers(self, primary: ProviderType) -> list[ProviderType]:
        """Get fallback provider list for primary provider.

        Args:
            primary: Primary selected provider

        Returns:
            List of fallback providers in priority order

        """
        all_providers = [ProviderType.GEMINI, ProviderType.CODEX, ProviderType.LLAMA]
        return [p for p in all_providers if p != primary]

    def _log_routing_decision(
        self, decision: RoutingDecision, request: LLMRequest
    ) -> None:
        """Log routing decision for monitoring and debugging.

        Args:
            decision: Routing decision to log
            request: Original request

        """
        self.logger.info(
            "Routing decision made",
            extra={
                "decision": decision.to_dict(),
                "request_metadata": getattr(request, "metadata", {}),
                "prompt_length": len(request.prompt),
            },
        )

    def _track_routing_decision(
        self, decision: RoutingDecision, request: LLMRequest
    ) -> None:
        """Track routing decision for machine learning optimization.

        Args:
            decision: Routing decision to track
            request: Original request

        """
        tracking_data = {
            "timestamp": datetime.now().isoformat(),
            "complexity_score": decision.complexity_score,
            "selected_provider": decision.selected_provider,
            "confidence": decision.confidence,
            "prompt_length": len(request.prompt),
            "features": decision.features.to_dict(),
            "routing_time_ms": decision.routing_time_ms,
        }

        self.routing_history.append(tracking_data)

        # Keep only recent history for memory management
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-500:]

    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics for monitoring and optimization.

        Returns:
            Dictionary with routing statistics

        """
        if not self.routing_history:
            return {"total_decisions": 0, "message": "No routing history available"}

        total_decisions = len(self.routing_history)

        # Provider selection distribution
        provider_counts = {}
        complexity_distribution = {}

        for decision in self.routing_history:
            provider = decision["selected_provider"]
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

            score_bucket = int(decision["complexity_score"])
            complexity_distribution[score_bucket] = (
                complexity_distribution.get(score_bucket, 0) + 1
            )

        # Calculate averages
        avg_complexity = (
            sum(d["complexity_score"] for d in self.routing_history) / total_decisions
        )
        avg_confidence = (
            sum(d["confidence"] for d in self.routing_history) / total_decisions
        )
        avg_routing_time = (
            sum(d["routing_time_ms"] for d in self.routing_history) / total_decisions
        )

        return {
            "total_decisions": total_decisions,
            "provider_distribution": provider_counts,
            "complexity_distribution": complexity_distribution,
            "average_complexity_score": round(avg_complexity, 2),
            "average_confidence": round(avg_confidence, 2),
            "average_routing_time_ms": round(avg_routing_time, 2),
            "history_window": f"Last {total_decisions} decisions",
        }
