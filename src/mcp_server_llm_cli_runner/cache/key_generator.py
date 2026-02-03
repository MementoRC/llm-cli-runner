"""Cache key generator for consistent hashing across the system.

This module provides utilities for generating deterministic cache keys
for requests, responses, and provider-specific data.

Key features:
    SHA256-based hashing for security and consistency
    Deterministic ordering for reproducible keys
    Support for request metadata inclusion
    Provider-specific key generation
    Configurable cache prefixes

Example:
    >>> generator = CacheKeyGenerator()
    >>> request = LLMRequest(prompt="Hello", provider="gemini")
    >>> key = generator.generate_request_key(request)
    >>> print(key)  # "req:a1b2c3..."

"""

import hashlib
import json
from typing import Any

from mcp_server_llm_cli_runner.core.models import LLMRequest
from mcp_server_llm_cli_runner.utils.logging import get_logger

logger = get_logger(__name__)


class CacheKeyGenerator:
    """Generates deterministic cache keys for LLM requests and responses.

    This class creates consistent, deterministic cache keys based on
    request parameters to enable effective caching while ensuring
    cache hits for identical requests.

    Attributes:
        prefix: Key prefix for request-based keys
        include_metadata: Whether to include request metadata in keys
        hash_algorithm: Hashing algorithm to use (default: sha256)

    """

    def __init__(
        self,
        prefix: str = "req",
        include_metadata: bool = True,
        hash_algorithm: str = "sha256",
    ) -> None:
        """Initialize cache key generator.

        Args:
            prefix: Prefix for generated keys
            include_metadata: Whether to include request metadata
            hash_algorithm: Hash algorithm to use

        """
        self.prefix = prefix
        self.include_metadata = include_metadata
        self.hash_algorithm = hash_algorithm

        logger.debug(
            "Initialized cache key generator",
            prefix=prefix,
            include_metadata=include_metadata,
            algorithm=hash_algorithm,
        )

    def generate_request_key(self, request: LLMRequest) -> str:
        """Generate cache key for an LLM request.

        Creates a deterministic hash based on all relevant request parameters
        that affect the response. Identical requests will always produce
        the same cache key.

        Args:
            request: LLM request to generate key for

        Returns:
            Deterministic cache key string

        Example:
            >>> generator = CacheKeyGenerator()
            >>> request = LLMRequest(prompt="Hello", provider="gemini")
            >>> key = generator.generate_request_key(request)
            >>> print(key)  # "req:a1b2c3..."

        """
        # Build deterministic content for hashing
        key_content = {
            "prompt": request.prompt,
            "provider": request.provider,  # provider is already a string
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system_prompt": request.system_prompt,
        }

        # Include metadata if configured
        if self.include_metadata and request.metadata:
            # Sort metadata keys for deterministic hashing
            key_content["metadata"] = self._normalize_metadata(request.metadata)

        # Create deterministic JSON representation
        content_json = json.dumps(key_content, sort_keys=True, separators=(",", ":"))

        # Generate SHA256 hash
        hash_obj = hashlib.sha256(content_json.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()

        return f"{self.prefix}:{hash_hex}"

    def generate_provider_key(self, provider: str, model: str | None = None) -> str:
        """Generate cache key for provider-specific data.

        Creates cache keys for provider configuration, model information,
        or other provider-specific cached data.

        Args:
            provider: Provider name
            model: Optional model name

        Returns:
            Provider-specific cache key

        Example:
            >>> generator = CacheKeyGenerator()
            >>> key = generator.generate_provider_key("gemini", "gemini-pro")
            >>> print(key)  # "req:provider:gemini:gemini-pro"

        """
        components = [self.prefix, "provider", provider]
        if model:
            components.append(model)

        return ":".join(components)

    def generate_session_key(self, session_id: str) -> str:
        """Generate cache key for session-specific data.

        Args:
            session_id: Session identifier

        Returns:
            Session-specific cache key

        """
        return f"{self.prefix}:session:{session_id}"

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Normalize metadata for consistent hashing.

        Sorts dictionary keys and handles nested structures to ensure
        deterministic representation.

        Args:
            metadata: Metadata dictionary to normalize

        Returns:
            Normalized metadata dictionary

        """
        if not isinstance(metadata, dict):
            return metadata

        normalized = {}
        for key in sorted(metadata.keys()):
            value = metadata[key]
            if isinstance(value, dict):
                normalized[key] = self._normalize_metadata(value)
            elif isinstance(value, list):
                # Sort lists if they contain comparable items
                try:
                    normalized[key] = sorted(value)
                except TypeError:
                    # If items aren't comparable, keep original order
                    normalized[key] = value
            else:
                normalized[key] = value

        return normalized

    def validate_key_format(self, key: str) -> bool:
        """Validate that a key follows expected format.

        Args:
            key: Cache key to validate

        Returns:
            True if key format is valid

        """
        try:
            parts = key.split(":")
            return len(parts) >= 2 and all(part for part in parts)
        except (AttributeError, ValueError):
            return False

    def extract_hash_from_key(self, key: str) -> str | None:
        """Extract hash portion from a cache key.

        Args:
            key: Cache key containing hash

        Returns:
            Hash portion of the key, or None if not found

        """
        try:
            parts = key.split(":")
            if len(parts) >= 2:
                # Assume last part is the hash
                return parts[-1]
        except (AttributeError, ValueError):
            pass

        return None

    def get_key_prefix(self, key: str) -> str | None:
        """Extract prefix from a cache key.

        Args:
            key: Cache key

        Returns:
            Prefix portion of the key, or None if not found

        """
        try:
            parts = key.split(":", 1)
            if len(parts) >= 1:
                return parts[0]
        except (AttributeError, ValueError):
            pass

        return None
