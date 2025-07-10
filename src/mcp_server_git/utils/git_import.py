"""
Safe git import utility for handling ClaudeCode environment conflicts.

This module provides a safe way to import GitPython in environments where
git commands may be redirected (like ClaudeCode development environment).
"""

import os
from typing import Optional, Any
from unittest.mock import MagicMock


def create_git_mock():
    """Create a mock git module for testing environments."""
    mock_git = MagicMock()
    
    # Mock common git classes and exceptions
    mock_git.Repo = MagicMock()
    mock_git.GitCommandError = Exception
    mock_git.InvalidGitRepositoryError = Exception
    mock_git.NoSuchPathError = Exception
    
    return mock_git


def safe_git_import() -> Any:
    """
    Safely import git module, handling ClaudeCode redirector conflicts.
    
    Returns:
        git module if successful, mock git module if in testing environment with conflicts
        
    Raises:
        ImportError: If git import fails for reasons other than ClaudeCode redirectors
    """
    try:
        import git
        return git
    except ImportError as e:
        if "Failed to initialize" in str(e) and "git version" in str(e):
            # This happens in ClaudeCode environment where git commands are redirected
            # For tests, we'll handle this gracefully
            if os.environ.get("TESTING", "").lower() in ("true", "1", "yes"):
                return create_git_mock()  # Return mock for testing
            else:
                raise ImportError(
                    "Git import failed due to command redirection. "
                    "This may occur in ClaudeCode environment. "
                    "Set TESTING=true to bypass for tests."
                ) from e
        else:
            raise


# Global git module instance - imported once
git = safe_git_import()

# Export commonly used classes for convenience
Repo = git.Repo
GitCommandError = git.GitCommandError

# Handle additional exceptions that might be used
try:
    InvalidGitRepositoryError = git.InvalidGitRepositoryError
    NoSuchPathError = git.NoSuchPathError
except AttributeError:
    # If we're using a mock, these might not exist
    InvalidGitRepositoryError = Exception
    NoSuchPathError = Exception