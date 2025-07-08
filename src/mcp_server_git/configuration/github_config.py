"""GitHub integration configuration model with validation.

This module provides GitHub-specific configuration for the MCP Git Server,
including API settings, authentication, webhook handling, and rate limiting.

Key features:
    - GitHub API configuration and timeouts
    - Authentication token management
    - Webhook security and validation
    - Rate limiting and retry policies
    - GitHub Enterprise support
    - PR and issue integration settings

Configuration areas:
    - API settings: Base URL, timeouts, retries
    - Authentication: Token validation, scopes
    - Webhooks: Secret validation, event filtering
    - Rate limiting: Buffer management, backoff
    - Integration: PR policies, issue linking

Example usage:
    >>> from mcp_server_git.configuration import GitHubConfig
    >>>
    >>> config = GitHubConfig(
    ...     api_token="ghp_xxxxxxxxxxxx",
    ...     webhook_secret="my-webhook-secret"
    ... )
    >>> print(f"API URL: {config.api_base_url}")
    API URL: https://api.github.com

    >>> # GitHub Enterprise configuration
    >>> enterprise_config = GitHubConfig(
    ...     api_base_url="https://github.enterprise.com/api/v3",
    ...     api_token="ghp_enterprise_token"
    ... )

See also:
    - server_config: Main server configuration
    - git_config: Git-specific configuration
"""

from typing import List, Optional
from pydantic import BaseModel, field_validator, Field, HttpUrl
import re


class GitHubConfig(BaseModel):
    """GitHub integration configuration with comprehensive validation.

    This class provides configuration for GitHub API integration,
    webhook handling, and GitHub-specific policies within the MCP Git Server.

    Attributes:
        API Settings:
            api_base_url: GitHub API base URL (GitHub.com or Enterprise)
            api_version: GitHub API version to use
            api_timeout_seconds: Timeout for API requests
            max_retries: Maximum retry attempts for failed requests
            retry_backoff_factor: Exponential backoff multiplier

        Authentication:
            api_token: GitHub API token (validated format)
            token_scopes: Required token scopes
            validate_token_on_startup: Validate token at startup

        Rate Limiting:
            rate_limit_buffer: Requests to keep in reserve
            rate_limit_threshold: Percentage threshold for warnings
            auto_retry_rate_limited: Auto retry when rate limited
            max_rate_limit_wait: Maximum wait time for rate limits

        Webhook Configuration:
            webhook_secret: Secret for webhook validation
            allowed_webhook_events: List of allowed webhook events
            webhook_timeout: Timeout for webhook processing
            validate_webhook_payload: Enable payload validation

        PR and Issue Settings:
            auto_link_issues: Automatically link issues in PRs
            require_issue_in_pr: Require issue reference in PRs
            allowed_pr_labels: Allowed labels for PRs
            auto_close_stale_prs: Auto close stale PRs
            stale_pr_days: Days before PR is considered stale

        GitHub Enterprise:
            is_enterprise: Whether using GitHub Enterprise
            enterprise_api_path: API path for Enterprise
            skip_ssl_verification: Skip SSL verification (not recommended)
    """

    # API Settings
    api_base_url: HttpUrl = Field(
        default_factory=lambda: HttpUrl("https://api.github.com"),
        description="GitHub API base URL"
    )
    api_version: str = Field(
        default="2022-11-28", description="GitHub API version (use YYYY-MM-DD format)"
    )
    api_timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Timeout for API requests in seconds"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts for failed requests"
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Exponential backoff multiplier for retries",
    )

    # Authentication
    api_token: Optional[str] = Field(
        default=None, description="GitHub API token for authentication"
    )
    token_scopes: List[str] = Field(
        default_factory=lambda: ["repo", "read:org"],
        description="Required GitHub token scopes",
    )
    validate_token_on_startup: bool = Field(
        default=True, description="Validate token and scopes at startup"
    )

    # Rate Limiting
    rate_limit_buffer: int = Field(
        default=100,
        ge=0,
        le=1000,
        description="Number of API requests to keep in reserve",
    )
    rate_limit_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Rate limit warning threshold (0.2 = warn at 20% remaining)",
    )
    auto_retry_rate_limited: bool = Field(
        default=True, description="Automatically retry requests when rate limited"
    )
    max_rate_limit_wait: int = Field(
        default=3600,
        ge=60,
        le=7200,
        description="Maximum seconds to wait for rate limit reset",
    )

    # Webhook Configuration
    webhook_secret: Optional[str] = Field(
        default=None, description="Secret for validating webhook payloads"
    )
    allowed_webhook_events: List[str] = Field(
        default_factory=lambda: [
            "push",
            "pull_request",
            "pull_request_review",
            "issues",
            "issue_comment",
            "workflow_run",
        ],
        description="List of allowed webhook event types",
    )
    webhook_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout for webhook processing in seconds",
    )
    validate_webhook_payload: bool = Field(
        default=True, description="Enable webhook payload signature validation"
    )

    # PR and Issue Settings
    auto_link_issues: bool = Field(
        default=True, description="Automatically link mentioned issues in PRs"
    )
    require_issue_in_pr: bool = Field(
        default=False, description="Require issue reference in PR description"
    )
    allowed_pr_labels: List[str] = Field(
        default_factory=list, description="Allowed labels for PRs (empty = all allowed)"
    )
    auto_close_stale_prs: bool = Field(
        default=False, description="Automatically close stale PRs"
    )
    stale_pr_days: int = Field(
        default=30,
        ge=7,
        le=365,
        description="Days of inactivity before PR is considered stale",
    )

    # GitHub Enterprise
    is_enterprise: bool = Field(
        default=False, description="Whether using GitHub Enterprise"
    )
    enterprise_api_path: str = Field(
        default="/api/v3", description="API path for GitHub Enterprise"
    )
    skip_ssl_verification: bool = Field(
        default=False,
        description="Skip SSL verification for Enterprise (not recommended)",
    )

    @field_validator("api_token")
    @classmethod
    def validate_api_token(cls, v: Optional[str]) -> Optional[str]:
        """Validate GitHub API token format.

        Args:
            v: API token to validate

        Returns:
            Validated token

        Raises:
            ValueError: If token format is invalid
        """
        if v is not None:
            # Check token format
            valid_prefixes = ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")
            if not any(v.startswith(prefix) for prefix in valid_prefixes):
                raise ValueError(
                    f"Invalid GitHub token format. Token must start with one of: "
                    f"{', '.join(valid_prefixes)}"
                )

            # Check token length (approximate)
            if len(v) < 40:
                raise ValueError("GitHub token appears too short")

        return v

    @field_validator("api_version")
    @classmethod
    def validate_api_version(cls, v: str) -> str:
        """Validate GitHub API version format.

        Args:
            v: API version string

        Returns:
            Validated version string

        Raises:
            ValueError: If version format is invalid
        """
        # Should be in YYYY-MM-DD format
        pattern = r"^\d{4}-\d{2}-\d{2}$"
        if not re.match(pattern, v):
            raise ValueError(
                "API version must be in YYYY-MM-DD format (e.g., '2022-11-28')"
            )
        return v

    @field_validator("allowed_webhook_events")
    @classmethod
    def validate_webhook_events(cls, v: List[str]) -> List[str]:
        """Validate webhook event types.

        Args:
            v: List of webhook events

        Returns:
            Validated list of events

        Raises:
            ValueError: If invalid event type
        """
        valid_events = {
            "branch_protection_rule",
            "check_run",
            "check_suite",
            "code_scanning_alert",
            "commit_comment",
            "create",
            "delete",
            "deployment",
            "deployment_status",
            "discussion",
            "discussion_comment",
            "fork",
            "gollum",
            "issue_comment",
            "issues",
            "label",
            "member",
            "membership",
            "milestone",
            "organization",
            "org_block",
            "page_build",
            "project",
            "project_card",
            "project_column",
            "public",
            "pull_request",
            "pull_request_review",
            "pull_request_review_comment",
            "pull_request_review_thread",
            "push",
            "registry_package",
            "release",
            "repository",
            "repository_dispatch",
            "secret_scanning_alert",
            "star",
            "status",
            "team",
            "team_add",
            "watch",
            "workflow_dispatch",
            "workflow_run",
        }

        invalid_events = set(v) - valid_events
        if invalid_events:
            raise ValueError(
                f"Invalid webhook events: {', '.join(invalid_events)}. "
                f"Valid events are: {', '.join(sorted(valid_events))}"
            )

        return v

    @field_validator("token_scopes")
    @classmethod
    def validate_token_scopes(cls, v: List[str]) -> List[str]:
        """Validate GitHub token scopes.

        Args:
            v: List of token scopes

        Returns:
            Validated list of scopes

        Raises:
            ValueError: If invalid scope
        """
        valid_scopes = {
            "repo",
            "repo:status",
            "repo_deployment",
            "public_repo",
            "repo:invite",
            "security_events",
            "admin:repo_hook",
            "write:repo_hook",
            "read:repo_hook",
            "admin:org",
            "write:org",
            "read:org",
            "admin:public_key",
            "write:public_key",
            "read:public_key",
            "admin:org_hook",
            "gist",
            "notifications",
            "user",
            "read:user",
            "user:email",
            "user:follow",
            "delete_repo",
            "write:discussion",
            "read:discussion",
            "write:packages",
            "read:packages",
            "delete:packages",
            "admin:gpg_key",
            "write:gpg_key",
            "read:gpg_key",
            "workflow",
        }

        # Check for invalid scopes
        invalid_scopes = []
        for scope in v:
            # Handle scopes with colons
            base_scope = scope.split(":")[0]
            if base_scope not in valid_scopes and scope not in valid_scopes:
                invalid_scopes.append(scope)

        if invalid_scopes:
            raise ValueError(f"Invalid token scopes: {', '.join(invalid_scopes)}")

        return v

    @field_validator("webhook_secret")
    @classmethod
    def validate_webhook_secret(cls, v: Optional[str]) -> Optional[str]:
        """Validate webhook secret strength.

        Args:
            v: Webhook secret

        Returns:
            Validated secret

        Raises:
            ValueError: If secret is too weak
        """
        if v is not None:
            if len(v) < 16:
                raise ValueError("Webhook secret must be at least 16 characters long")

            # Warn if secret appears weak
            if v.lower() in ["secret", "password", "webhook", "github"]:
                import warnings

                warnings.warn(
                    "Webhook secret appears weak. Use a strong, random secret.",
                    UserWarning,
                )

        return v

    @field_validator("api_base_url", mode="after")
    @classmethod
    def validate_api_base_url(cls, v: HttpUrl) -> HttpUrl:
        """Validate and normalize API base URL.

        Args:
            v: API base URL

        Returns:
            Validated URL
        """
        # HttpUrl normalizes URLs automatically, including trailing slashes
        # We'll accept the default behavior of HttpUrl
        return v

    model_config = {
        # Allow population by field name
        "populate_by_name": True,
        # Schema generation with examples
        "json_schema_extra": {
            "example": {
                "api_base_url": "https://api.github.com",
                "api_version": "2022-11-28",
                "api_timeout_seconds": 30,
                "max_retries": 3,
                "retry_backoff_factor": 2.0,
                "api_token": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                "token_scopes": ["repo", "read:org"],
                "validate_token_on_startup": True,
                "rate_limit_buffer": 100,
                "rate_limit_threshold": 0.2,
                "auto_retry_rate_limited": True,
                "max_rate_limit_wait": 3600,
                "webhook_secret": "my-secure-webhook-secret-123",
                "allowed_webhook_events": ["push", "pull_request"],
                "webhook_timeout": 30,
                "validate_webhook_payload": True,
                "auto_link_issues": True,
                "require_issue_in_pr": False,
                "allowed_pr_labels": ["bug", "feature", "enhancement"],
                "auto_close_stale_prs": False,
                "stale_pr_days": 30,
                "is_enterprise": False,
                "enterprise_api_path": "/api/v3",
                "skip_ssl_verification": False,
            }
        },
    }
