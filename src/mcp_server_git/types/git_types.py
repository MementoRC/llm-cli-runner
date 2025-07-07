"""
Git domain type definitions for the MCP Git Server.

This module provides comprehensive type definitions for Git-related operations,
including repository paths, branches, commits, and operation results. All types
follow TDD specifications and provide runtime validation where appropriate.

Type categories:
    - Path types: GitRepositoryPath for repository validation
    - Reference types: GitBranch, GitCommitHash, GitRemoteName
    - Status types: GitFileStatus, GitOperationResult
    - Information types: GitCommitInfo, GitBranchInfo, GitRemoteInfo
    - Error types: GitValidationError, GitOperationError

Design principles:
    - Semantic clarity through domain-specific types
    - Runtime validation for critical operations
    - Interoperability with pathlib and standard Git tools
    - Comprehensive error handling and reporting
"""

from pathlib import Path
from typing import NewType, TypedDict, Literal, Optional, List, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime


# Basic Git Type Aliases
GitBranchName = NewType('GitBranchName', str)
GitCommitHash = NewType('GitCommitHash', str)
GitRemoteName = NewType('GitRemoteName', str)
GitTagName = NewType('GitTagName', str)


# Git File Status Types
GitFileStatusType = Literal[
    "modified", "added", "deleted", "renamed", "copied", 
    "unmerged", "untracked", "ignored", "staged"
]


# Git Operation Status Types
GitOperationStatus = Literal["success", "failure", "timeout", "cancelled"]


class GitValidationError(Exception):
    """Exception raised when Git type validation fails."""
    
    def __init__(self, message: str, value: Any = None, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.value = value
        self.context = context or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        return f"Git validation error: {self.message}"


class GitOperationError(Exception):
    """Exception raised when Git operations fail."""
    
    def __init__(self, message: str, command: Optional[str] = None, exit_code: Optional[int] = None):
        self.message = message
        self.command = command
        self.exit_code = exit_code
        super().__init__(message)


@dataclass
class GitRepositoryPath:
    """
    Type-safe representation of a Git repository path.
    
    Validates that the path points to a valid Git repository and provides
    metadata about the repository structure and state.
    
    Attributes:
        path: The normalized absolute path to the repository
        is_bare: Whether this is a bare repository
        git_dir: Path to the .git directory
        work_tree: Path to the working tree (None for bare repos)
    """
    
    path: Path
    is_bare: bool = False
    git_dir: Optional[Path] = None
    work_tree: Optional[Path] = None
    current_branch: Optional[str] = None
    remotes: Optional[List[str]] = None
    is_clean: bool = True
    
    def __init__(self, path: Union[str, Path]):
        """
        Initialize GitRepositoryPath with validation.
        
        Args:
            path: Path to the Git repository (string or Path object)
            
        Raises:
            GitValidationError: If path is not a valid Git repository
        """
        # Convert to Path object and normalize
        if isinstance(path, str):
            path_obj = Path(path)
        else:
            path_obj = path
        
        # Resolve to absolute path
        try:
            normalized_path = path_obj.resolve()
        except (OSError, ValueError) as e:
            raise GitValidationError(
                f"Invalid path: {path}",
                value=path,
                context={"error": str(e)}
            )
        
        # Check if path exists
        if not normalized_path.exists():
            raise GitValidationError(
                f"Path does not exist: {normalized_path}",
                value=path,
                context={"resolved_path": str(normalized_path)}
            )
        
        # Validate Git repository
        git_dir, work_tree, is_bare = self._validate_git_repository(normalized_path)
        
        # Set attributes
        self.path = normalized_path
        self.git_dir = git_dir
        self.work_tree = work_tree
        self.is_bare = is_bare
        self.current_branch = None  # Will be populated on demand
        self.remotes = []  # Will be populated on demand
        self.is_clean = True  # Will be populated on demand
    
    def _validate_git_repository(self, path: Path) -> tuple[Path, Optional[Path], bool]:
        """
        Validate that the path is a valid Git repository.
        
        Returns:
            Tuple of (git_dir, work_tree, is_bare)
            
        Raises:
            GitValidationError: If not a valid Git repository
        """
        # Check for .git directory or bare repository
        if path.is_file():
            raise GitValidationError(
                f"Path is a file, not a directory: {path}",
                value=path
            )
        
        # Check for .git subdirectory (normal repository)
        git_subdir = path / ".git"
        if git_subdir.exists():
            if git_subdir.is_dir():
                return git_subdir, path, False
            elif git_subdir.is_file():
                # Git worktree - .git is a file containing path to real .git
                try:
                    git_file_content = git_subdir.read_text().strip()
                    if git_file_content.startswith("gitdir: "):
                        git_dir_path = Path(git_file_content[8:])
                        if not git_dir_path.is_absolute():
                            git_dir_path = path / git_dir_path
                        return git_dir_path.resolve(), path, False
                except (OSError, ValueError):
                    pass
        
        # Check if this is a bare repository
        if (path / "objects").exists() and (path / "refs").exists():
            # Additional checks for bare repository
            head_file = path / "HEAD"
            config_file = path / "config"
            if head_file.exists() and config_file.exists():
                return path, None, True
        
        # Check if we're inside a git repository (search parent directories)
        current = path
        while current != current.parent:
            git_dir = current / ".git"
            if git_dir.exists():
                if git_dir.is_dir():
                    return git_dir, current, False
                elif git_dir.is_file():
                    # Handle git worktree
                    try:
                        git_file_content = git_dir.read_text().strip()
                        if git_file_content.startswith("gitdir: "):
                            git_dir_path = Path(git_file_content[8:])
                            if not git_dir_path.is_absolute():
                                git_dir_path = current / git_dir_path
                            return git_dir_path.resolve(), current, False
                    except (OSError, ValueError):
                        pass
            current = current.parent
        
        raise GitValidationError(
            f"Path is not a Git repository: {path}",
            value=path,
            context={"searched_parents": True}
        )
    
    def __str__(self) -> str:
        return str(self.path)
    
    def __fspath__(self) -> str:
        """Support os.PathLike protocol."""
        return str(self.path)
    
    def is_valid(self) -> bool:
        """
        Check if this is a valid git repository path.
        
        Returns:
            True if the path is a valid git repository
        """
        return self.git_dir is not None and self.git_dir.exists()
    
    def exists(self) -> bool:
        """
        Check if the repository path exists.
        
        Returns:
            True if the path exists
        """
        return self.path.exists()
    
    def get_repository_info(self) -> Dict[str, Any]:
        """
        Get metadata about the repository.
        
        Returns:
            Dictionary containing repository information
        """
        return {
            "path": str(self.path),
            "git_dir": str(self.git_dir) if self.git_dir else None,
            "work_tree": str(self.work_tree) if self.work_tree else None,
            "is_bare": self.is_bare,
            "exists": self.path.exists(),
            "is_git_repository": True
        }


@dataclass
class GitBranch:
    """Type-safe representation of a Git branch."""
    
    name: GitBranchName
    is_current: bool = False
    is_remote: bool = False
    remote_name: Optional[GitRemoteName] = None
    upstream: Optional[str] = None
    commit_hash: Optional[GitCommitHash] = None
    
    def __init__(self, name: str, **kwargs):
        """Initialize GitBranch with validation."""
        if not self._is_valid_branch_name(name):
            raise GitValidationError(f"Invalid branch name: {name}", value=name)
        
        self.name = GitBranchName(name)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @staticmethod
    def _is_valid_branch_name(name: str) -> bool:
        """Validate Git branch name according to Git rules."""
        if not name:
            return False
        
        # Basic checks - can be extended
        invalid_chars = [' ', '~', '^', ':', '?', '*', '[', '\\']
        if any(char in name for char in invalid_chars):
            return False
        
        if name.startswith('-') or name.endswith('.') or '..' in name:
            return False
        
        return True
    
    def __str__(self) -> str:
        """Return the branch name as string."""
        return str(self.name)
    
    def is_valid(self) -> bool:
        """Check if this is a valid branch."""
        return self._is_valid_branch_name(str(self.name))
    
    def is_feature_branch(self) -> bool:
        """Check if this is a feature branch."""
        return str(self.name).startswith('feature/')
    
    def is_main_branch(self) -> bool:
        """Check if this is a main branch."""
        return str(self.name) in ['main', 'master', 'develop']
    
    def is_release_branch(self) -> bool:
        """Check if this is a release branch."""
        return str(self.name).startswith('release/')
    
    @property
    def parent_branch(self) -> Optional[str]:
        """Get the parent branch name."""
        if '/' in str(self.name):
            return str(self.name).split('/')[0]
        return None
    
    @property
    def namespace(self) -> Optional[str]:
        """Get the branch namespace."""
        if '/' in str(self.name):
            return str(self.name).split('/')[0]
        return None


@dataclass
class GitCommitHash:
    """Type-safe representation of a Git commit hash."""
    
    hash: str
    is_short: bool = False
    
    def __init__(self, hash_value: str):
        """Initialize GitCommitHash with validation."""
        if not self._is_valid_commit_hash(hash_value):
            raise GitValidationError(f"Invalid commit hash: {hash_value}", value=hash_value)
        
        self.hash = hash_value
        self.is_short = len(hash_value) < 40
    
    @staticmethod
    def _is_valid_commit_hash(hash_value: str) -> bool:
        """Validate Git commit hash format."""
        if not hash_value:
            return False
        
        # Check length (7-40 characters for Git hashes)
        if not (7 <= len(hash_value) <= 40):
            return False
        
        # Check if all characters are valid hex
        try:
            int(hash_value, 16)
            return True
        except ValueError:
            return False
    
    def __str__(self) -> str:
        return self.hash
    
    def is_valid(self) -> bool:
        """Check if this is a valid commit hash."""
        return self._is_valid_commit_hash(self.hash)
    
    def short(self) -> str:
        """Return short version of the hash (7 characters)."""
        return self.hash[:7]
    
    def full(self) -> str:
        """Return full version of the hash (40 characters)."""
        if len(self.hash) == 40:
            return self.hash
        # For short hashes, we can't expand to full, so return what we have
        return self.hash


@dataclass
class GitFileStatus:
    """Representation of Git file status."""
    
    path: str
    status: GitFileStatusType
    old_path: Optional[str] = None  # For renamed files
    
    def __init__(self, path: str, status: GitFileStatusType, old_path: Optional[str] = None):
        if status not in ["modified", "added", "deleted", "renamed", "copied", 
                         "unmerged", "untracked", "ignored", "staged"]:
            raise GitValidationError(f"Invalid file status: {status}", value=status)
        
        self.path = path
        self.status = status
        self.old_path = old_path
    
    def is_modified(self) -> bool:
        return self.status == "modified"
    
    def is_added(self) -> bool:
        return self.status == "added"
    
    def is_deleted(self) -> bool:
        return self.status == "deleted"
    
    def is_staged(self) -> bool:
        return self.status == "staged"


@dataclass
class GitOperationResult:
    """Result of a Git operation."""
    
    success: bool
    message: str
    command: Optional[str] = None
    exit_code: Optional[int] = None
    output: Optional[str] = None
    error: Optional[str] = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if not self.success and not self.error:
            raise GitValidationError("Failed operations must include error information")
    
    def raise_for_status(self) -> None:
        """Raise GitOperationError if operation failed."""
        if not self.success:
            raise GitOperationError(
                self.message, 
                command=self.command, 
                exit_code=self.exit_code
            )
    
    def chain(self, other: 'GitOperationResult') -> 'GitOperationResult':
        """Chain this result with another operation result."""
        if not self.success:
            return self
        return other


class GitStatusResult(TypedDict):
    """Result of git status operation."""
    branch: Optional[str]
    is_clean: bool
    staged_files: List[GitFileStatus]
    modified_files: List[GitFileStatus]
    untracked_files: List[str]
    deleted_files: List[str]
    renamed_files: List[GitFileStatus]


class GitDiffResult(TypedDict):
    """Result of git diff operation."""
    files_changed: int
    insertions: int
    deletions: int
    diff_content: str


class GitLogResult(TypedDict):
    """Result of git log operation."""
    commits: List['GitCommitInfo']
    total_count: int
    has_more: bool


class GitCommitInfo(TypedDict):
    """Complete commit information."""
    hash: GitCommitHash
    author: str
    author_email: str
    committer: str
    committer_email: str
    message: str
    timestamp: datetime
    parent_hashes: List[GitCommitHash]


class GitBranchInfo(TypedDict):
    """Complete branch information."""
    name: GitBranchName
    is_current: bool
    is_remote: bool
    commit_hash: GitCommitHash
    upstream: Optional[str]
    ahead: Optional[int]
    behind: Optional[int]


class GitRemoteInfo(TypedDict):
    """Complete remote information."""
    name: GitRemoteName
    url: str
    fetch_url: str
    push_url: str
    branches: List[GitBranchName]


# Type Integration Tests Support
class GitTypeIntegration:
    """Helper class for testing type integration."""
    
    @staticmethod
    def validate_repository_path(path: Union[str, Path]) -> GitRepositoryPath:
        """Validate and return GitRepositoryPath."""
        return GitRepositoryPath(path)
    
    @staticmethod
    def create_branch(name: str, **kwargs) -> GitBranch:
        """Create and validate GitBranch."""
        return GitBranch(name, **kwargs)
    
    @staticmethod
    def create_commit_hash(hash_value: str) -> GitCommitHash:
        """Create and validate GitCommitHash."""
        return GitCommitHash(hash_value)


# Export all public types
__all__ = [
    # Core types
    'GitRepositoryPath',
    'GitBranch', 
    'GitCommitHash',
    'GitRemoteName',
    'GitBranchName',
    'GitTagName',
    'GitFileStatus',
    'GitOperationResult',
    
    # Status and info types
    'GitStatusResult',
    'GitDiffResult',
    'GitLogResult',
    'GitCommitInfo', 
    'GitBranchInfo',
    'GitRemoteInfo',
    
    # Error types
    'GitValidationError',
    'GitOperationError',
    
    # Literal types
    'GitFileStatusType',
    'GitOperationStatus',
    
    # Helper classes
    'GitTypeIntegration',
]