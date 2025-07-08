"""Git-specific configuration model with validation.

This module provides Git-related configuration for the MCP Git Server,
including repository management, operation limits, and Git-specific settings.

Key features:
    - Repository operation configuration
    - Clone and fetch policies
    - Commit message validation rules
    - Branch protection settings
    - Performance tuning options
    - Security constraints

Configuration areas:
    - Repository management: Clone depth, submodule handling
    - Operation limits: File size limits, diff limits
    - Commit policies: Message patterns, author validation
    - Branch protection: Protected branches, merge policies
    - Performance: Cache settings, parallel operations

Example usage:
    >>> from mcp_server_git.configuration import GitConfig
    >>> 
    >>> config = GitConfig()
    >>> print(f"Max file size: {config.max_file_size_mb}MB")
    Max file size: 100MB
    
    >>> # Custom configuration
    >>> config = GitConfig(
    ...     default_clone_depth=10,
    ...     enable_commit_signing=True,
    ...     protected_branches=["main", "release/*"]
    ... )
    
See also:
    - server_config: Main server configuration
    - github_config: GitHub integration configuration
"""

from typing import List, Optional
from pydantic import BaseModel, field_validator, Field, model_validator


class GitConfig(BaseModel):
    """Git-specific configuration with comprehensive validation.
    
    This class provides configuration for Git operations, repository management,
    and Git-specific policies within the MCP Git Server.
    
    Attributes:
        Repository Management:
            default_clone_depth: Default depth for shallow clones (0 = full clone)
            max_repository_size_gb: Maximum allowed repository size in GB
            allow_force_push: Allow force push operations
            allow_submodules: Allow operations on submodules
            auto_gc_threshold: Number of operations before auto garbage collection
            
        Operation Limits:
            max_file_size_mb: Maximum file size in MB for operations
            max_diff_size_mb: Maximum diff size in MB
            max_commit_message_length: Maximum commit message length
            max_files_per_commit: Maximum files in a single commit
            diff_context_lines: Number of context lines in diffs
            
        Commit Policies:
            enable_commit_signing: Require signed commits
            commit_message_pattern: Regex pattern for commit messages
            allowed_authors: List of allowed commit authors
            require_issue_reference: Require issue references in commits
            
        Branch Protection:
            protected_branches: List of protected branch patterns
            allow_delete_branch: Allow branch deletion
            require_pull_request: Require PR for protected branches
            min_approvals: Minimum approvals for PRs
            
        Performance:
            enable_git_cache: Enable Git object caching
            cache_size_mb: Git cache size in MB
            parallel_operations: Number of parallel Git operations
            operation_queue_size: Maximum queued operations
    """
    
    # Repository Management
    default_clone_depth: int = Field(
        default=0,
        ge=0,
        le=1000,
        description="Default depth for shallow clones (0 = full clone)"
    )
    max_repository_size_gb: float = Field(
        default=10.0,
        gt=0,
        le=100.0,
        description="Maximum allowed repository size in GB"
    )
    allow_force_push: bool = Field(
        default=False,
        description="Allow force push operations"
    )
    allow_submodules: bool = Field(
        default=True,
        description="Allow operations on repositories with submodules"
    )
    auto_gc_threshold: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of operations before automatic garbage collection"
    )
    
    # Operation Limits
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum file size in MB for operations"
    )
    max_diff_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum diff size in MB"
    )
    max_commit_message_length: int = Field(
        default=1000,
        ge=50,
        le=5000,
        description="Maximum commit message length in characters"
    )
    max_files_per_commit: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of files in a single commit"
    )
    diff_context_lines: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of context lines to show in diffs"
    )
    
    # Commit Policies
    enable_commit_signing: bool = Field(
        default=False,
        description="Require all commits to be signed"
    )
    commit_message_pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern for validating commit messages"
    )
    allowed_authors: List[str] = Field(
        default_factory=list,
        description="List of allowed commit authors (empty = all allowed)"
    )
    require_issue_reference: bool = Field(
        default=False,
        description="Require issue references in commit messages"
    )
    
    # Branch Protection
    protected_branches: List[str] = Field(
        default_factory=lambda: ["main", "master"],
        description="List of protected branch patterns (supports wildcards)"
    )
    allow_delete_branch: bool = Field(
        default=True,
        description="Allow branch deletion operations"
    )
    require_pull_request: bool = Field(
        default=False,
        description="Require pull requests for protected branches"
    )
    min_approvals: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Minimum approvals required for pull requests"
    )
    
    # Performance
    enable_git_cache: bool = Field(
        default=True,
        description="Enable Git object caching for performance"
    )
    cache_size_mb: int = Field(
        default=256,
        ge=32,
        le=2048,
        description="Git cache size in MB"
    )
    parallel_operations: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel Git operations allowed"
    )
    operation_queue_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum number of queued operations"
    )
    
    @field_validator('commit_message_pattern')
    @classmethod
    def validate_commit_pattern(cls, v: Optional[str]) -> Optional[str]:
        """Validate that commit message pattern is a valid regex.
        
        Args:
            v: Regex pattern to validate
            
        Returns:
            Validated regex pattern
            
        Raises:
            ValueError: If pattern is not a valid regex
        """
        if v is not None:
            import re
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        return v
    
    @field_validator('protected_branches')
    @classmethod
    def validate_protected_branches(cls, v: List[str]) -> List[str]:
        """Validate protected branch patterns.
        
        Args:
            v: List of branch patterns
            
        Returns:
            Validated list of branch patterns
            
        Raises:
            ValueError: If patterns are invalid
        """
        if not v:
            # At least one protected branch recommended
            import warnings
            warnings.warn(
                "No protected branches configured. Consider protecting 'main' or 'master'.",
                UserWarning
            )
        
        # Validate patterns
        for pattern in v:
            if not pattern or pattern.isspace():
                raise ValueError("Branch pattern cannot be empty")
            
            # Basic validation of wildcard patterns
            if '*' in pattern:
                # Ensure wildcards are used correctly
                if pattern.count('*') > 1 and '**' not in pattern:
                    raise ValueError(
                        f"Invalid branch pattern '{pattern}'. "
                        "Use '*' for single level or '**' for multiple levels."
                    )
        
        return v
    
    @field_validator('allowed_authors')
    @classmethod
    def validate_allowed_authors(cls, v: List[str]) -> List[str]:
        """Validate allowed authors list.
        
        Args:
            v: List of allowed authors
            
        Returns:
            Validated list of authors
            
        Raises:
            ValueError: If author format is invalid
        """
        for author in v:
            if not author or author.isspace():
                raise ValueError("Author cannot be empty")
            
            # Basic email validation if email is provided
            if '@' in author and not cls._is_valid_author_format(author):
                raise ValueError(
                    f"Invalid author format '{author}'. "
                    "Use 'Name <email@example.com>' or just 'Name'"
                )
        
        return v
    
    @staticmethod
    def _is_valid_author_format(author: str) -> bool:
        """Check if author string follows Git author format.
        
        Args:
            author: Author string to validate
            
        Returns:
            True if valid format, False otherwise
        """
        import re
        # Match "Name <email@example.com>" or "email@example.com"
        pattern = r'^([^<>]+\s*<[^@\s]+@[^@\s]+\.[^@\s]+>|[^@\s]+@[^@\s]+\.[^@\s]+)$'
        return bool(re.match(pattern, author.strip()))
    
    @model_validator(mode='after')
    def validate_dependent_settings(self) -> 'GitConfig':
        """Validate interdependent settings.
            
        Returns:
            Self after validation
            
        Raises:
            ValueError: If settings conflict
        """
        # If commit signing is enabled, ensure related settings make sense
        if self.enable_commit_signing and self.allow_force_push:
            import warnings
            warnings.warn(
                "Commit signing is enabled but force push is allowed. "
                "This may compromise signature integrity.",
                UserWarning
            )
        
        # If requiring pull requests, ensure min_approvals makes sense
        if self.require_pull_request and self.min_approvals == 0:
            import warnings
            warnings.warn(
                "Pull requests are required but no approvals are needed. "
                "Consider setting min_approvals > 0.",
                UserWarning
            )
        
        return self
    
    model_config = {
        # Environment variable prefix
        "env_prefix": "MCP_GIT_",
        
        # Case sensitivity for environment variables
        "case_sensitive": False,
        
        # Schema generation with examples
        "json_schema_extra": {
            "example": {
                "default_clone_depth": 0,
                "max_repository_size_gb": 10.0,
                "allow_force_push": False,
                "allow_submodules": True,
                "auto_gc_threshold": 1000,
                "max_file_size_mb": 100,
                "max_diff_size_mb": 50,
                "max_commit_message_length": 1000,
                "max_files_per_commit": 1000,
                "diff_context_lines": 3,
                "enable_commit_signing": False,
                "commit_message_pattern": "^(feat|fix|docs|style|refactor|test|chore):\\s.+",
                "allowed_authors": [],
                "require_issue_reference": False,
                "protected_branches": ["main", "master", "release/*"],
                "allow_delete_branch": True,
                "require_pull_request": True,
                "min_approvals": 2,
                "enable_git_cache": True,
                "cache_size_mb": 256,
                "parallel_operations": 4,
                "operation_queue_size": 100
            }
        }
    }