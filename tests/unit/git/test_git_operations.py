"""
Unit tests for Git operations in mcp_server_git.git.operations module.

These tests verify the git operations functions that provide the core functionality
for the MCP git server, focusing on the git_add function and related operations.

Critical for TDD Compliance:
    These tests define the behavior that the implementation must satisfy.
    DO NOT modify these tests to match a broken implementation - the
    implementation must be fixed to pass these tests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from mcp_server_git.git.operations import git_add
from mcp_server_git.utils.git_import import GitCommandError, Repo


class TestGitAdd:
    """Test git_add operations with comprehensive validation."""

    @patch('mcp_server_git.git.operations.Path')
    def test_git_add_stages_existing_files_successfully(self, mock_path_class):
        """Should successfully stage existing files that exist on filesystem."""
        # Arrange
        mock_repo = Mock(spec=Repo)
        mock_repo.working_dir = "/test/repo"
        mock_repo.git.status.return_value = ""  # No files in status
        mock_repo.git.add = Mock()
        mock_repo.git.diff.return_value = "file1.py\nfile2.py"
        
        mock_path = Mock()
        mock_path.__truediv__ = Mock(return_value=mock_path)
        mock_path.exists.return_value = True
        mock_path.is_symlink.return_value = False
        mock_path_class.return_value = mock_path
        
        files = ["file1.py", "file2.py"]

        # Act
        result = git_add(mock_repo, files)

        # Assert
        assert "✅ Added 2 file(s) to staging area: file1.py, file2.py" in result
        mock_repo.git.add.assert_called_once_with(*files)
        mock_repo.git.diff.assert_called_once_with("--cached", "--name-only")

    @patch('mcp_server_git.git.operations.Path')
    def test_git_add_stages_deleted_files_successfully(self, mock_path_class):
        """Should successfully stage deleted files that appear in git status."""
        # Arrange
        mock_repo = Mock(spec=Repo)
        mock_repo.working_dir = "/test/repo"
        # Simulate git status showing deleted file
        mock_repo.git.status.return_value = " D deleted_file.py\n M modified_file.py"
        mock_repo.git.add = Mock()
        mock_repo.git.diff.return_value = "deleted_file.py"
        
        mock_path = Mock()
        mock_path.__truediv__ = Mock(return_value=mock_path)
        mock_path.exists.return_value = False  # File doesn't exist (deleted)
        mock_path.is_symlink.return_value = False
        mock_path_class.return_value = mock_path
        
        files = ["deleted_file.py"]

        # Act
        result = git_add(mock_repo, files)

        # Assert
        assert "✅ Added 1 file(s) to staging area: deleted_file.py" in result
        mock_repo.git.add.assert_called_once_with(*files)
        # Should call git status to check for deleted files
        mock_repo.git.status.assert_called_once_with("--porcelain")

    @patch('mcp_server_git.git.operations.Path')
    def test_git_add_rejects_truly_missing_files(self, mock_path_class):
        """Should fail for files that don't exist and aren't in git status."""
        # Arrange
        mock_repo = Mock(spec=Repo)
        mock_repo.working_dir = "/test/repo"
        # Empty git status - no files known to git
        mock_repo.git.status.return_value = ""
        
        mock_path = Mock()
        mock_path.__truediv__ = Mock(return_value=mock_path)
        mock_path.exists.return_value = False  # File doesn't exist
        mock_path.is_symlink.return_value = False
        mock_path_class.return_value = mock_path
        
        files = ["nonexistent_file.py"]

        # Act
        result = git_add(mock_repo, files)

        # Assert
        assert "❌ Files not found: nonexistent_file.py" in result
        # Should not call git add since file validation failed
        mock_repo.git.add.assert_not_called()

    @patch('mcp_server_git.git.operations.Path')
    def test_git_add_handles_mixed_scenarios(self, mock_path_class):
        """Should handle both existing and deleted files in one call."""
        # Arrange
        mock_repo = Mock(spec=Repo)
        mock_repo.working_dir = "/test/repo"
        # Git status shows one deleted file
        mock_repo.git.status.return_value = " D deleted.py"
        mock_repo.git.add = Mock()
        mock_repo.git.diff.return_value = "existing.py\ndeleted.py"
        
        # Mock path behavior: existing.py exists, deleted.py doesn't
        def mock_path_side_effect(path_str):
            mock_path = Mock()
            if "existing.py" in str(path_str):
                mock_path.exists.return_value = True
            else:  # deleted.py
                mock_path.exists.return_value = False
            mock_path.is_symlink.return_value = False
            return mock_path
        
        mock_path_class.side_effect = lambda x: mock_path_side_effect(x)
        mock_path_class.return_value.__truediv__.side_effect = mock_path_side_effect
        
        files = ["existing.py", "deleted.py"]

        # Act
        result = git_add(mock_repo, files)

        # Assert
        assert "✅ Added 2 file(s) to staging area: existing.py, deleted.py" in result
        mock_repo.git.add.assert_called_once_with(*files)

    def test_git_add_handles_git_command_error(self):
        """Should handle git command errors gracefully."""
        # Arrange
        mock_repo = Mock(spec=Repo)
        mock_repo.working_dir = "/test/repo"
        mock_repo.git.status.side_effect = GitCommandError("Git command failed")
        
        files = ["test.py"]

        # Act
        result = git_add(mock_repo, files)

        # Assert
        assert "❌ Git add failed: Git command failed" in result

    def test_git_add_handles_general_exception(self):
        """Should handle general exceptions gracefully."""
        # Arrange
        mock_repo = Mock(spec=Repo)
        mock_repo.working_dir = "/test/repo"
        mock_repo.git.status.side_effect = Exception("Unexpected error")
        
        files = ["test.py"]

        # Act
        result = git_add(mock_repo, files)

        # Assert
        assert "❌ Add error: Unexpected error" in result

    @patch('mcp_server_git.git.operations.Path')
    def test_git_add_parses_porcelain_format_correctly(self, mock_path_class):
        """Should correctly parse git status --porcelain format."""
        # Arrange
        mock_repo = Mock(spec=Repo)
        mock_repo.working_dir = "/test/repo"
        # Various porcelain format entries
        mock_repo.git.status.return_value = (
            " M modified_file.py\n"
            " D deleted_file.py\n"
            "?? untracked_file.py\n"
            "A  added_file.py\n"
            "MM conflict_file.py"
        )
        mock_repo.git.add = Mock()
        mock_repo.git.diff.return_value = "deleted_file.py"
        
        mock_path = Mock()
        mock_path.__truediv__ = Mock(return_value=mock_path)
        mock_path.exists.return_value = False  # File doesn't exist (deleted)
        mock_path.is_symlink.return_value = False
        mock_path_class.return_value = mock_path
        
        files = ["deleted_file.py"]

        # Act
        result = git_add(mock_repo, files)

        # Assert
        assert "✅ Added 1 file(s) to staging area: deleted_file.py" in result
        # Verify that status parsing correctly identified the file

    @patch('mcp_server_git.git.operations.Path')
    def test_git_add_avoids_false_positives_in_status_parsing(self, mock_path_class):
        """Should avoid false positives when parsing status lines."""
        # Arrange
        mock_repo = Mock(spec=Repo)
        mock_repo.working_dir = "/test/repo"
        # Status contains "file.py" but we're looking for "my_file.py"
        mock_repo.git.status.return_value = " D some_other_file.py"
        
        mock_path = Mock()
        mock_path.__truediv__ = Mock(return_value=mock_path)
        mock_path.exists.return_value = False  # File doesn't exist 
        mock_path.is_symlink.return_value = False
        mock_path_class.return_value = mock_path
        
        files = ["file.py"]  # This should NOT match "some_other_file.py"

        # Act
        result = git_add(mock_repo, files)

        # Assert
        assert "❌ Files not found: file.py" in result
        mock_repo.git.add.assert_not_called()

    @patch('mcp_server_git.git.operations.Path')
    def test_git_add_handles_symlinks_correctly(self, mock_path_class):
        """Should handle symlinks the same as regular files."""
        # Arrange
        mock_repo = Mock(spec=Repo)
        mock_repo.working_dir = "/test/repo"
        mock_repo.git.status.return_value = ""
        mock_repo.git.add = Mock()
        mock_repo.git.diff.return_value = "symlink_file.py"
        
        mock_path = Mock()
        mock_path.__truediv__ = Mock(return_value=mock_path)
        mock_path.exists.return_value = False  # Symlink doesn't exist as regular file
        mock_path.is_symlink.return_value = True  # But it is a symlink
        mock_path_class.return_value = mock_path
        
        files = ["symlink_file.py"]

        # Act
        result = git_add(mock_repo, files)

        # Assert
        assert "✅ Added 1 file(s) to staging area: symlink_file.py" in result
        mock_repo.git.add.assert_called_once_with(*files)

    @patch('mcp_server_git.git.operations.Path')
    def test_git_add_handles_verification_fallback(self, mock_path_class):
        """Should handle verification fallback when git diff fails."""
        # Arrange
        mock_repo = Mock(spec=Repo)
        mock_repo.working_dir = "/test/repo"
        mock_repo.git.status.return_value = ""
        mock_repo.git.add = Mock()
        # First diff call fails, should try fallback
        mock_repo.git.diff.side_effect = GitCommandError("diff failed")
        
        # Mock the index.diff fallback
        mock_item = Mock()
        mock_item.a_path = "test_file.py"
        mock_repo.index.diff.return_value = [mock_item]
        
        mock_path = Mock()
        mock_path.__truediv__ = Mock(return_value=mock_path)
        mock_path.exists.return_value = True
        mock_path.is_symlink.return_value = False
        mock_path_class.return_value = mock_path
        
        files = ["test_file.py"]

        # Act
        result = git_add(mock_repo, files)

        # Assert
        assert "✅ Added 1 file(s) to staging area: test_file.py" in result
        mock_repo.git.add.assert_called_once_with(*files)

    @patch('mcp_server_git.git.operations.Path')
    def test_git_add_handles_verification_double_fallback(self, mock_path_class):
        """Should handle when both verification methods fail."""
        # Arrange
        mock_repo = Mock(spec=Repo)
        mock_repo.working_dir = "/test/repo"
        mock_repo.git.status.return_value = ""
        mock_repo.git.add = Mock()
        # Both verification methods fail
        mock_repo.git.diff.side_effect = GitCommandError("diff failed")
        mock_repo.index.diff.side_effect = GitCommandError("index diff failed")
        
        mock_path = Mock()
        mock_path.__truediv__ = Mock(return_value=mock_path)
        mock_path.exists.return_value = True
        mock_path.is_symlink.return_value = False
        mock_path_class.return_value = mock_path
        
        files = ["test_file.py"]

        # Act
        result = git_add(mock_repo, files)

        # Assert
        assert "⚠️ No changes detected in specified files" in result
        mock_repo.git.add.assert_called_once_with(*files)


# Test fixtures for integration testing
@pytest.fixture
def temp_git_repo_with_files(tmp_path):
    """Create a temporary Git repository with sample files for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    import subprocess
    
    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo_path, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True
    )
    
    # Create some test files
    (repo_path / "existing_file.py").write_text("print('hello')")
    (repo_path / "to_be_deleted.py").write_text("print('goodbye')")
    
    # Add and commit initial files
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)
    
    # Delete a file to test deleted file staging
    os.unlink(repo_path / "to_be_deleted.py")
    
    # Create a new file
    (repo_path / "new_file.py").write_text("print('new')")
    
    return repo_path


# Mark for test organization
pytestmark = [pytest.mark.unit, pytest.mark.git_operations]