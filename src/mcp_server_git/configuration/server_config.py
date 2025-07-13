"""Server configuration model with comprehensive validation.

This module implements the GitServerConfig class as specified in section 4.3.2 of the PRD,
providing a Pydantic-based configuration model with validation for all configurable aspects
of the MCP Git Server.

Key features:
    - Type-safe configuration with Pydantic BaseModel
    - Comprehensive validation for all fields
    - Default values following best practices
    - Field descriptions for documentation
    - Validation constraints using Field class
    - Custom validators for complex validation logic
    - Environment variable support through Config
    - Schema generation with examples

Configuration sections:
    - Server settings: Host, port, concurrency limits
    - Security settings: Path validation, operation restrictions
    - GitHub integration: Token validation, API configuration
    - Logging and monitoring: Log levels, metrics configuration

Example usage:
    >>> from mcp_server_git.configuration import GitServerConfig
    >>>
    >>> config = GitServerConfig()  # Uses defaults
    >>> print(f"Server: {config.host}:{config.port}")
    Server: localhost:8080

    >>> # Custom configuration
    >>> config = GitServerConfig(
    ...     port=9000,
    ...     max_concurrent_operations=20,
    ...     github_token="ghp_exampletoken123"
    ... )

    >>> # Validation example
    >>> try:
    ...     config = GitServerConfig(port=99999)  # Invalid port
    ... except ValidationError as e:
    ...     print(e.errors())

See also:
    - git_config: Git-specific configuration
    - github_config: GitHub integration configuration
    - security_config: Security and authentication configuration
"""

from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator

# Import custom types - these should be defined in the types module
# For now, we'll use str as a placeholder for GitHubToken
GitHubToken = str  # This should be: from mcp_server_git.types import GitHubToken


class GitServerConfig(BaseModel):
    """Configuration for Git server operations with comprehensive validation.

    This class provides the main configuration model for the MCP Git Server,
    with built-in validation and sensible defaults for all settings.

    Attributes:
        host: Server host address (default: localhost)
        port: Server port number (1024-65535, default: 8080)
        max_concurrent_operations: Maximum concurrent Git operations (1-100, default: 10)
        operation_timeout_seconds: Timeout for Git operations (30-1800 seconds, default: 300)

        enable_security_validation: Enable security checks (default: True)
        allowed_repository_paths: List of allowed repository paths (validated)
        forbidden_operations: List of forbidden Git operations
        require_gpg_signing: Require GPG signing for commits (default: False)

        github_token: GitHub API token (optional, validated format)
        github_api_timeout: GitHub API timeout (5-120 seconds, default: 30)
        github_rate_limit_buffer: Buffer for GitHub rate limits (10-1000, default: 100)

        log_level: Logging level (DEBUG/INFO/WARNING/ERROR, default: INFO)
        enable_metrics_collection: Enable metrics collection (default: True)
        metrics_retention_days: Days to retain metrics (1-365, default: 30)
    """

    # Server settings
    host: str = Field(default="localhost", description="Server host address")
    port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Server port (must be between 1024 and 65535)",
    )
    max_concurrent_operations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of concurrent Git operations",
    )
    operation_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Timeout for Git operations in seconds (30s to 30min)",
    )

    # Security settings
    enable_security_validation: bool = Field(
        default=True, description="Enable security validation for Git operations"
    )
    allowed_repository_paths: list[Path] = Field(
        default_factory=list,
        description="List of allowed repository paths (must exist and be Git repos)",
    )
    forbidden_operations: list[str] = Field(
        default_factory=list,
        description="List of forbidden Git operations (e.g., 'force-push', 'rebase')",
    )
    require_gpg_signing: bool = Field(
        default=False, description="Require GPG signing for all commits"
    )

    # GitHub integration
    github_token: Union[GitHubToken, None] = Field(
        default=None, description="GitHub API token for authentication"
    )
    github_api_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="GitHub API timeout in seconds (5s to 2min)",
    )
    github_rate_limit_buffer: int = Field(
        default=100, ge=10, le=1000, description="Buffer for GitHub API rate limits"
    )

    # Logging and monitoring
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level for the server"
    )
    enable_metrics_collection: bool = Field(
        default=True, description="Enable collection of performance metrics"
    )
    metrics_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to retain metrics (1 to 365)",
    )

    @field_validator("allowed_repository_paths")
    @classmethod
    def validate_repository_paths(cls, v: list[Path]) -> list[Path]:
        """Validate that all repository paths exist and are Git repositories.

        Args:
            v: List of repository paths to validate

        Returns:
            Validated list of repository paths

        Raises:
            ValueError: If a path doesn't exist or isn't a Git repository
        """
        for path in v:
            if not path.exists():
                raise ValueError(f"Repository path does not exist: {path}")
            if not (path / ".git").exists():
                raise ValueError(f"Path is not a Git repository: {path}")
        return v

    @field_validator("github_token")
    @classmethod
    def validate_github_token(cls, v: Union[str, None]) -> Union[str, None]:
        """Validate GitHub token format.

        GitHub tokens must start with specific prefixes:
        - ghp_: Personal access tokens (classic)
        - gho_: OAuth access tokens
        - ghu_: User-to-server tokens for GitHub Apps
        - ghs_: Server-to-server tokens for GitHub Apps
        - ghr_: Refresh tokens for GitHub Apps

        Args:
            v: GitHub token to validate

        Returns:
            Validated GitHub token

        Raises:
            ValueError: If token has invalid format
        """
        if v and not v.startswith(("ghp_", "gho_", "ghu_", "ghs_", "ghr_")):
            raise ValueError(
                "Invalid GitHub token format. Token must start with: "
                "ghp_ (personal), gho_ (OAuth), ghu_ (user-to-server), "
                "ghs_ (server-to-server), or ghr_ (refresh)"
            )
        return v

    @field_validator("forbidden_operations")
    @classmethod
    def validate_forbidden_operations(cls, v: list[str]) -> list[str]:
        """Validate forbidden operations list.

        Ensures that forbidden operations are recognized Git operations.

        Args:
            v: List of forbidden operations

        Returns:
            Validated list of forbidden operations

        Raises:
            ValueError: If an unrecognized operation is specified
        """
        valid_operations = {
            "push",
            "pull",
            "fetch",
            "commit",
            "merge",
            "rebase",
            "cherry-pick",
            "reset",
            "revert",
            "force-push",
            "force-pull",
            "branch-delete",
            "tag-delete",
            "stash",
            "clean",
        }

        for op in v:
            if op not in valid_operations:
                raise ValueError(
                    f"Unrecognized operation '{op}'. Valid operations: "
                    f"{', '.join(sorted(valid_operations))}"
                )
        return v

    model_config = {
        # Allow population by field name (for flexibility)
        "populate_by_name": True,
        # JSON encoders for Path objects
        "json_encoders": {Path: str},
        # Schema generation with examples
        "json_schema_extra": {
            "example": {
                "host": "localhost",
                "port": 8080,
                "max_concurrent_operations": 10,
                "operation_timeout_seconds": 300,
                "enable_security_validation": True,
                "allowed_repository_paths": ["/path/to/repo1", "/path/to/repo2"],
                "forbidden_operations": ["force-push", "rebase"],
                "require_gpg_signing": False,
                "github_token": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                "github_api_timeout": 30,
                "github_rate_limit_buffer": 100,
                "log_level": "INFO",
                "enable_metrics_collection": True,
                "metrics_retention_days": 30,
            }
        },
    }
