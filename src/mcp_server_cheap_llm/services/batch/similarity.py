"""Similarity analysis for intelligent batch request grouping.

This module provides similarity analysis capabilities for batch requests,
enabling intelligent grouping of similar prompts to optimize cache utilization
and reduce redundant processing. Uses text similarity algorithms and semantic
analysis for effective request clustering.

Key classes:
    SimilarityAnalyzer: Main similarity analysis and grouping engine
    TextSimilarity: Text-based similarity calculation utilities
    SemanticGrouper: Semantic clustering for request optimization

Example:
    >>> analyzer = SimilarityAnalyzer(threshold=0.7)
    >>> groups = await analyzer.analyze_batch(batch_request)
    >>> optimized_requests = await analyzer.optimize_batch(batch_request)

"""

import asyncio
import hashlib
import re
from difflib import SequenceMatcher
from typing import Any

from mcp_server_cheap_llm.core.models import BatchRequest, LLMRequest
from mcp_server_cheap_llm.utils.logging import get_logger

logger = get_logger(__name__)


class TextSimilarity:
    """Text similarity calculation utilities using multiple algorithms.

    Provides various text similarity metrics including character-based,
    word-based, and semantic similarity measurements for request clustering.
    """

    @staticmethod
    def calculate_char_similarity(text1: str, text2: str) -> float:
        """Calculate character-level similarity using SequenceMatcher.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            float: Similarity score between 0.0 and 1.0

        """
        if not text1 or not text2:
            return 0.0

        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    @staticmethod
    def calculate_word_similarity(text1: str, text2: str) -> float:
        """Calculate word-level similarity using Jaccard index.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            float: Similarity score between 0.0 and 1.0

        """
        if not text1 or not text2:
            return 0.0

        # Simple word tokenization (could be enhanced with NLP libraries)
        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def calculate_semantic_similarity(text1: str, text2: str) -> float:
        """Calculate semantic similarity using simple keyword matching.

        This is a simplified implementation. In production, this could use:
        - Sentence transformers
        - Word embeddings (Word2Vec, GloVe)
        - Language models (BERT, etc.)

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            float: Similarity score between 0.0 and 1.0

        """
        if not text1 or not text2:
            return 0.0

        # Simple semantic keywords (could be expanded)
        semantic_keywords = {
            "explain": ["describe", "tell", "what", "how"],
            "code": ["program", "script", "function", "algorithm"],
            "python": ["py", "programming", "syntax"],
            "example": ["sample", "demo", "illustration"],
            "write": ["create", "generate", "make"],
            "list": ["enumerate", "show", "display"],
        }

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Check for semantic keyword matches
        semantic_score = 0.0
        total_checks = 0

        for keyword, synonyms in semantic_keywords.items():
            total_checks += 1

            keyword_in_1 = keyword in text1_lower or any(
                syn in text1_lower for syn in synonyms
            )
            keyword_in_2 = keyword in text2_lower or any(
                syn in text2_lower for syn in synonyms
            )

            if keyword_in_1 and keyword_in_2:
                semantic_score += 1.0
            elif keyword_in_1 or keyword_in_2:
                semantic_score += 0.3  # Partial match

        return semantic_score / total_checks if total_checks > 0 else 0.0

    @classmethod
    def calculate_composite_similarity(cls, text1: str, text2: str) -> float:
        """Calculate composite similarity using multiple metrics.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            float: Weighted composite similarity score

        """
        char_sim = cls.calculate_char_similarity(text1, text2)
        word_sim = cls.calculate_word_similarity(text1, text2)
        semantic_sim = cls.calculate_semantic_similarity(text1, text2)

        # Weighted average (can be tuned based on performance)
        return (
            char_sim * 0.3  # Character similarity
            + word_sim * 0.4  # Word similarity (most important)
            + semantic_sim * 0.3  # Semantic similarity
        )


class SimilarityGroup:
    """Container for a group of similar requests with metadata.

    Attributes:
        representative_index: Index of the representative request
        member_indices: List of all member request indices
        similarity_scores: Similarity scores for each member
        group_size: Number of requests in the group
        optimization_potential: Estimated optimization benefit

    """

    def __init__(self, representative_index: int) -> None:
        """Initialize similarity group.

        Args:
            representative_index: Index of the representative request

        """
        self.representative_index = representative_index
        self.member_indices = [representative_index]
        self.similarity_scores = {representative_index: 1.0}
        self.optimization_potential = 0.0

    def add_member(self, index: int, score: float) -> None:
        """Add member to the group.

        Args:
            index: Request index to add
            score: Similarity score to representative

        """
        self.member_indices.append(index)
        self.similarity_scores[index] = score

        # Update optimization potential (more members = higher potential)
        self.optimization_potential = (len(self.member_indices) - 1) * 0.8

    @property
    def group_size(self) -> int:
        """Get number of requests in group."""
        return len(self.member_indices)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "representative_index": self.representative_index,
            "member_indices": self.member_indices,
            "group_size": self.group_size,
            "average_similarity": sum(self.similarity_scores.values())
            / len(self.similarity_scores),
            "optimization_potential": self.optimization_potential,
        }


class SimilarityAnalyzer:
    """Main similarity analysis engine for batch request optimization.

    Analyzes batches of requests to identify similar prompts and group them
    for optimized processing. Provides caching optimization and reduces
    redundant LLM calls through intelligent similarity detection.

    Attributes:
        similarity_threshold: Minimum similarity score for grouping
        max_group_size: Maximum number of requests per group
        cache_similar_requests: Whether to cache similar request analysis
        analysis_cache: Cache for similarity analysis results

    Example:
        >>> analyzer = SimilarityAnalyzer(threshold=0.75)
        >>> groups = await analyzer.analyze_batch(batch_request)
        >>> print(f"Found {len(groups)} similarity groups")

    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_group_size: int = 10,
        cache_similar_requests: bool = True,
    ) -> None:
        """Initialize similarity analyzer.

        Args:
            similarity_threshold: Minimum similarity score for grouping (0.0-1.0)
            max_group_size: Maximum requests per similarity group
            cache_similar_requests: Whether to cache analysis results

        """
        self.similarity_threshold = similarity_threshold
        self.max_group_size = max_group_size
        self.cache_similar_requests = cache_similar_requests

        # Analysis cache for performance
        self.analysis_cache: dict[str, dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Statistics
        self.total_analyses = 0
        self.total_groups_found = 0
        self.total_optimization_potential = 0.0

        logger.info(
            f"Similarity analyzer initialized: threshold={similarity_threshold}, "
            f"max_group_size={max_group_size}",
        )

    async def analyze_batch(self, batch_request: BatchRequest) -> list[SimilarityGroup]:
        """Analyze batch request for similarity groups.

        Args:
            batch_request: Batch request to analyze

        Returns:
            List[SimilarityGroup]: List of identified similarity groups

        """
        start_time = asyncio.get_event_loop().time()

        requests = batch_request.requests
        if len(requests) < 2:
            logger.debug(
                f"Batch {batch_request.batch_id} too small for similarity analysis",
            )
            return []

        # Check cache first
        cache_key = self._generate_cache_key(requests)
        if self.cache_similar_requests and cache_key in self.analysis_cache:
            self.cache_hits += 1
            cached_result = self.analysis_cache[cache_key]
            logger.debug(
                f"Using cached similarity analysis for batch {batch_request.batch_id}",
            )
            return self._deserialize_groups(cached_result["groups"])

        self.cache_misses += 1

        # Perform similarity analysis
        groups = await self._find_similarity_groups(
            requests,
            batch_request.similarity_threshold,
        )

        # Cache results
        if self.cache_similar_requests:
            self.analysis_cache[cache_key] = {
                "groups": [group.to_dict() for group in groups],
                "timestamp": start_time,
                "batch_size": len(requests),
            }

        # Update statistics
        self.total_analyses += 1
        self.total_groups_found += len(groups)
        self.total_optimization_potential += sum(
            group.optimization_potential for group in groups
        )

        analysis_time = (asyncio.get_event_loop().time() - start_time) * 1000
        logger.info(
            f"Similarity analysis for batch {batch_request.batch_id}: "
            f"found {len(groups)} groups in {analysis_time:.2f}ms",
        )

        return groups

    async def optimize_batch(
        self,
        batch_request: BatchRequest,
    ) -> tuple[BatchRequest, list[SimilarityGroup]]:
        """Optimize batch request based on similarity analysis.

        This method analyzes the batch for similar requests and potentially
        reorganizes them for better cache utilization and processing efficiency.

        Args:
            batch_request: Original batch request

        Returns:
            Tuple[BatchRequest, List[SimilarityGroup]]: Optimized batch and similarity groups

        """
        groups = await self.analyze_batch(batch_request)

        if not groups:
            logger.debug(
                f"No optimization opportunities found for batch {batch_request.batch_id}",
            )
            return batch_request, []

        # Create optimized batch with grouped requests
        optimized_requests = []
        processed_indices = set()

        # Process similarity groups first (for better cache locality)
        for group in sorted(
            groups,
            key=lambda g: g.optimization_potential,
            reverse=True,
        ):
            for index in group.member_indices:
                if index not in processed_indices:
                    optimized_requests.append(batch_request.requests[index])
                    processed_indices.add(index)

        # Add remaining requests
        for i, request in enumerate(batch_request.requests):
            if i not in processed_indices:
                optimized_requests.append(request)

        # Create optimized batch request
        optimized_batch = BatchRequest(
            batch_id=batch_request.batch_id,  # Keep same ID
            requests=optimized_requests,
            priority=batch_request.priority,
            similarity_threshold=batch_request.similarity_threshold,
            max_parallel=batch_request.max_parallel,
            callback_url=batch_request.callback_url,
            metadata={
                **batch_request.metadata,
                "optimized": True,
                "similarity_groups": len(groups),
                "optimization_potential": sum(g.optimization_potential for g in groups),
            },
            created_at=batch_request.created_at,
            estimated_processing_time=batch_request.estimated_processing_time,
        )

        logger.info(
            f"Optimized batch {batch_request.batch_id}: "
            f"{len(groups)} groups, potential improvement: "
            f"{sum(g.optimization_potential for g in groups):.2f}",
        )

        return optimized_batch, groups

    async def _find_similarity_groups(
        self,
        requests: list[LLMRequest],
        threshold: float,
    ) -> list[SimilarityGroup]:
        """Find similarity groups in list of requests.

        Args:
            requests: List of LLM requests to analyze
            threshold: Similarity threshold for grouping

        Returns:
            List[SimilarityGroup]: Identified similarity groups

        """
        groups = []
        used_indices = set()

        # Use provided threshold or default
        active_threshold = threshold if threshold > 0 else self.similarity_threshold

        for i, request1 in enumerate(requests):
            if i in used_indices:
                continue

            # Start new group with this request as representative
            group = SimilarityGroup(representative_index=i)
            used_indices.add(i)

            # Find similar requests
            for j, request2 in enumerate(requests[i + 1 :], start=i + 1):
                if (
                    j in used_indices
                    or len(group.member_indices) >= self.max_group_size
                ):
                    continue

                # Calculate similarity
                similarity = TextSimilarity.calculate_composite_similarity(
                    request1.prompt,
                    request2.prompt,
                )

                if similarity >= active_threshold:
                    group.add_member(j, similarity)
                    used_indices.add(j)

            # Only keep groups with multiple members
            if group.group_size > 1:
                groups.append(group)

        return groups

    def _generate_cache_key(self, requests: list[LLMRequest]) -> str:
        """Generate cache key for request list.

        Args:
            requests: List of requests to generate key for

        Returns:
            str: Cache key based on request content

        """
        # Create hash based on prompts and key parameters
        content = "|".join(
            [
                f"{req.prompt}:{req.provider}:{req.max_tokens}:{req.temperature}"
                for req in requests
            ],
        )

        return hashlib.sha256(content.encode("utf-8")).hexdigest()[
            :16
        ]  # Truncate to match MD5 length

    def _deserialize_groups(
        self,
        group_dicts: list[dict[str, Any]],
    ) -> list[SimilarityGroup]:
        """Deserialize similarity groups from cache.

        Args:
            group_dicts: List of serialized group dictionaries

        Returns:
            List[SimilarityGroup]: Deserialized similarity groups

        """
        groups = []

        for group_dict in group_dicts:
            group = SimilarityGroup(group_dict["representative_index"])

            for index in group_dict["member_indices"][1:]:  # Skip representative
                # Estimate similarity score (not stored in simple cache)
                score = group_dict.get("average_similarity", 0.8)
                group.add_member(index, score)

            groups.append(group)

        return groups

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dict[str, Any]: Cache performance metrics

        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (
            (self.cache_hits / total_requests * 100) if total_requests > 0 else 0.0
        )

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": hit_rate,
            "cache_size": len(self.analysis_cache),
            "total_analyses": self.total_analyses,
            "average_groups_per_batch": (
                self.total_groups_found / self.total_analyses
                if self.total_analyses > 0
                else 0.0
            ),
            "average_optimization_potential": (
                self.total_optimization_potential / self.total_analyses
                if self.total_analyses > 0
                else 0.0
            ),
        }

    def clear_cache(self) -> None:
        """Clear similarity analysis cache."""
        self.analysis_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Similarity analysis cache cleared")


class SemanticGrouper:
    """Advanced semantic grouping for batch optimization.

    This class provides more sophisticated semantic analysis for request
    grouping, potentially using machine learning models or external APIs
    for better similarity detection.

    Note: This is a placeholder for future ML-based semantic analysis.
    """

    def __init__(self, model_name: str = "simple") -> None:
        """Initialize semantic grouper.

        Args:
            model_name: Name of the semantic model to use

        """
        self.model_name = model_name
        logger.info(f"Semantic grouper initialized with model: {model_name}")

    async def group_by_semantic_meaning(
        self,
        requests: list[LLMRequest],
        threshold: float = 0.8,
    ) -> list[list[int]]:
        """Group requests by semantic meaning.

        This is a placeholder implementation. In production, this could use:
        - Sentence-BERT for semantic embeddings
        - OpenAI embeddings API
        - Other transformer-based models

        Args:
            requests: List of requests to group
            threshold: Semantic similarity threshold

        Returns:
            List[List[int]]: Lists of request indices grouped by semantic meaning

        """
        # Placeholder implementation using simple keyword matching
        groups = []
        used_indices = set()

        # Simple semantic categories
        categories = {
            "code_related": ["code", "function", "script", "program", "algorithm"],
            "explanation": ["explain", "describe", "what", "how", "why"],
            "examples": ["example", "sample", "demo", "show"],
            "creation": ["write", "create", "generate", "make", "build"],
        }

        for keywords in categories.values():
            group = []

            for i, request in enumerate(requests):
                if i in used_indices:
                    continue

                prompt_lower = request.prompt.lower()
                if any(keyword in prompt_lower for keyword in keywords):
                    group.append(i)
                    used_indices.add(i)

            if len(group) > 1:
                groups.append(group)

        logger.debug(f"Semantic grouping found {len(groups)} semantic groups")
        return groups
