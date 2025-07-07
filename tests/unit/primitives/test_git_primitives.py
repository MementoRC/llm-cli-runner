"""
Test specifications for Git primitive operations.

These tests define the behavioral requirements for atomic Git operations
that serve as the foundation for higher-level Git functionality.

CRITICAL: These tests are IMMUTABLE once complete. They define the interface
and behavior that the git_primitives module MUST implement. 

DO NOT modify these tests to match implementation - implementation must 
satisfy these tests to prevent the LLM compliance issue.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

# Import the primitives we expect to be implemented  
# These imports will fail initially (RED phase) - that's expected!
try:
    from mcp_server_git.primitives.git_primitives import (
        execute_git_command,
        get_repository_status,
        get_staged_files,
        get_unstaged_files,
        get_untracked_files,
        is_git_repository,
        get_current_branch,
        get_commit_hash,
        validate_repository_path,
        parse_git_status_output,
        parse_git_log_output,
        format_git_error,
        GitCommandError,
        GitRepositoryError,
        GitValidationError,
    )
    PRIMITIVES_AVAILABLE = True
except ImportError:
    PRIMITIVES_AVAILABLE = False


class TestExecuteGitCommand:
    """Test specifications for execute_git_command primitive."""

    def test_should_execute_simple_git_commands(self):
        """execute_git_command should execute basic git commands."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock successful git command
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "On branch main\nnothing to commit, working tree clean"
            mock_result.stderr = ""
            
            # ACT & ASSERT: Should execute command successfully
            with patch('subprocess.run', return_value=mock_result):
                result = execute_git_command("/tmp/repo", ["status", "--porcelain"])
                
                assert result.success is True
                assert result.output == "On branch main\nnothing to commit, working tree clean"
                assert result.error is None
                assert result.return_code == 0
        else:
            pytest.fail("execute_git_command not implemented - RED phase expected")

    def test_should_handle_git_command_failures(self):
        """execute_git_command should handle git command failures properly."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock failed git command
            mock_result = MagicMock()
            mock_result.returncode = 128
            mock_result.stdout = ""
            mock_result.stderr = "fatal: not a git repository"
            
            # ACT & ASSERT: Should handle failure gracefully
            with patch('subprocess.run', return_value=mock_result):
                result = execute_git_command("/tmp/not-repo", ["status"])
                
                assert result.success is False
                assert result.error == "fatal: not a git repository"
                assert result.return_code == 128

    def test_should_validate_repository_path(self):
        """execute_git_command should validate repository path exists."""
        if PRIMITIVES_AVAILABLE:
            # ACT & ASSERT: Should raise error for non-existent path
            with pytest.raises(GitRepositoryError) as exc_info:
                execute_git_command("/does/not/exist", ["status"])
            
            assert "repository path does not exist" in str(exc_info.value).lower()

    def test_should_validate_git_command_format(self):
        """execute_git_command should validate git command format."""
        if PRIMITIVES_AVAILABLE:
            # ACT & ASSERT: Should raise error for invalid commands
            with pytest.raises(GitCommandError) as exc_info:
                execute_git_command("/tmp/repo", [])  # Empty command
            
            assert "invalid git command" in str(exc_info.value).lower()

    def test_should_handle_command_timeout(self):
        """execute_git_command should handle command timeouts."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock timeout
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("git", 30)):
                # ACT & ASSERT: Should handle timeout gracefully
                result = execute_git_command("/tmp/repo", ["log"], timeout=30)
                
                assert result.success is False
                assert "timeout" in result.error.lower()


class TestRepositoryValidation:
    """Test specifications for repository validation primitives."""

    def test_is_git_repository_should_detect_valid_repos(self):
        """is_git_repository should correctly identify git repositories."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock valid git repository
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.is_dir", return_value=True),
                patch("pathlib.Path.glob", return_value=[Path(".git")]),
            ):
                # ACT & ASSERT: Should identify as git repository
                assert is_git_repository("/tmp/valid-repo") is True

    def test_is_git_repository_should_reject_non_repos(self):
        """is_git_repository should reject non-git directories."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock non-git directory
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.is_dir", return_value=True),
                patch("pathlib.Path.glob", return_value=[]),  # No .git directory
            ):
                # ACT & ASSERT: Should reject as non-git repository
                assert is_git_repository("/tmp/not-repo") is False

    def test_validate_repository_path_should_accept_valid_paths(self):
        """validate_repository_path should accept valid repository paths."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock valid repository
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.is_dir", return_value=True),
                patch("pathlib.Path.glob", return_value=[Path(".git")]),
            ):
                # ACT & ASSERT: Should validate successfully
                result = validate_repository_path("/tmp/valid-repo")
                assert result.is_valid is True
                assert result.absolute_path.is_absolute()

    def test_validate_repository_path_should_reject_invalid_paths(self):
        """validate_repository_path should reject invalid paths."""
        if PRIMITIVES_AVAILABLE:
            # ACT & ASSERT: Should raise validation error
            with pytest.raises(GitValidationError) as exc_info:
                validate_repository_path("/does/not/exist")
            
            assert "invalid repository path" in str(exc_info.value).lower()


class TestRepositoryStatus:
    """Test specifications for repository status primitives."""

    def test_get_repository_status_should_parse_clean_repo(self):
        """get_repository_status should correctly parse clean repository status."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock clean repository status
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = ""  # Empty porcelain output = clean
            
            with patch('mcp_server_git.primitives.git_primitives.execute_git_command', return_value=mock_result):
                # ACT: Get repository status
                status = get_repository_status("/tmp/repo")
                
                # ASSERT: Should indicate clean repository
                assert status.is_clean is True
                assert len(status.modified_files) == 0
                assert len(status.untracked_files) == 0
                assert len(status.staged_files) == 0

    def test_get_repository_status_should_parse_dirty_repo(self):
        """get_repository_status should correctly parse dirty repository status."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock dirty repository status
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = " M file1.py\n?? file2.py\nA  file3.py"
            
            with patch('mcp_server_git.primitives.git_primitives.execute_git_command', return_value=mock_result):
                # ACT: Get repository status
                status = get_repository_status("/tmp/repo")
                
                # ASSERT: Should parse all file statuses correctly
                assert status.is_clean is False
                assert "file1.py" in status.modified_files
                assert "file2.py" in status.untracked_files
                assert "file3.py" in status.staged_files

    def test_get_staged_files_should_return_staged_files_only(self):
        """get_staged_files should return only staged files."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock git status output with staged files
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "A  staged1.py\nM  staged2.py\n M unstaged.py\n?? untracked.py"
            
            with patch('mcp_server_git.primitives.git_primitives.execute_git_command', return_value=mock_result):
                # ACT: Get staged files
                staged_files = get_staged_files("/tmp/repo")
                
                # ASSERT: Should return only staged files
                assert "staged1.py" in staged_files
                assert "staged2.py" in staged_files
                assert "unstaged.py" not in staged_files
                assert "untracked.py" not in staged_files

    def test_get_unstaged_files_should_return_unstaged_files_only(self):
        """get_unstaged_files should return only unstaged modified files."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock git status output with unstaged files
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "A  staged.py\n M unstaged1.py\n D unstaged2.py\n?? untracked.py"
            
            with patch('mcp_server_git.primitives.git_primitives.execute_git_command', return_value=mock_result):
                # ACT: Get unstaged files
                unstaged_files = get_unstaged_files("/tmp/repo")
                
                # ASSERT: Should return only unstaged modified files
                assert "unstaged1.py" in unstaged_files
                assert "unstaged2.py" in unstaged_files
                assert "staged.py" not in unstaged_files
                assert "untracked.py" not in unstaged_files

    def test_get_untracked_files_should_return_untracked_files_only(self):
        """get_untracked_files should return only untracked files."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock git status output with untracked files
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "A  staged.py\n M modified.py\n?? untracked1.py\n?? untracked2.py"
            
            with patch('mcp_server_git.primitives.git_primitives.execute_git_command', return_value=mock_result):
                # ACT: Get untracked files
                untracked_files = get_untracked_files("/tmp/repo")
                
                # ASSERT: Should return only untracked files
                assert "untracked1.py" in untracked_files
                assert "untracked2.py" in untracked_files
                assert "staged.py" not in untracked_files
                assert "modified.py" not in untracked_files


class TestBranchOperations:
    """Test specifications for branch operation primitives."""

    def test_get_current_branch_should_return_branch_name(self):
        """get_current_branch should return current branch name."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock git branch output
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "main"
            
            with patch('mcp_server_git.primitives.git_primitives.execute_git_command', return_value=mock_result):
                # ACT: Get current branch
                branch = get_current_branch("/tmp/repo")
                
                # ASSERT: Should return branch name
                assert branch == "main"

    def test_get_current_branch_should_handle_detached_head(self):
        """get_current_branch should handle detached HEAD state."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock detached HEAD state
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "HEAD detached at a1b2c3d"
            
            with patch('mcp_server_git.primitives.git_primitives.execute_git_command', return_value=mock_result):
                # ACT: Get current branch
                branch = get_current_branch("/tmp/repo")
                
                # ASSERT: Should return None or special indicator for detached HEAD
                assert branch is None or branch == "HEAD"


class TestCommitOperations:
    """Test specifications for commit operation primitives."""

    def test_get_commit_hash_should_return_current_commit(self):
        """get_commit_hash should return current commit hash."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock git rev-parse output
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "a1b2c3d4e5f6789012345678901234567890abcd"
            
            with patch('mcp_server_git.primitives.git_primitives.execute_git_command', return_value=mock_result):
                # ACT: Get commit hash
                commit_hash = get_commit_hash("/tmp/repo")
                
                # ASSERT: Should return full commit hash
                assert commit_hash == "a1b2c3d4e5f6789012345678901234567890abcd"
                assert len(commit_hash) == 40

    def test_get_commit_hash_should_support_short_format(self):
        """get_commit_hash should support short hash format."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock short hash output
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "a1b2c3d"
            
            with patch('mcp_server_git.primitives.git_primitives.execute_git_command', return_value=mock_result):
                # ACT: Get short commit hash
                commit_hash = get_commit_hash("/tmp/repo", short=True)
                
                # ASSERT: Should return short commit hash
                assert commit_hash == "a1b2c3d"
                assert len(commit_hash) == 7


class TestOutputParsing:
    """Test specifications for git output parsing primitives."""

    def test_parse_git_status_output_should_parse_porcelain_format(self):
        """parse_git_status_output should correctly parse git status --porcelain output."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Sample git status --porcelain output
            status_output = """ M file1.py
A  file2.py
D  file3.py
?? file4.py
R  file5.py -> file6.py
C  file7.py -> file8.py"""
            
            # ACT: Parse status output
            parsed = parse_git_status_output(status_output)
            
            # ASSERT: Should correctly categorize all files
            assert "file1.py" in parsed.modified_files
            assert "file2.py" in parsed.added_files
            assert "file3.py" in parsed.deleted_files
            assert "file4.py" in parsed.untracked_files
            assert any("file5.py" in rename for rename in parsed.renamed_files)
            assert any("file7.py" in copy for copy in parsed.copied_files)

    def test_parse_git_log_output_should_parse_commit_entries(self):
        """parse_git_log_output should correctly parse git log output."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Sample git log output
            log_output = """commit a1b2c3d4e5f6789012345678901234567890abcd
Author: John Doe <john@example.com>
Date:   Mon Dec 1 10:00:00 2023 +0000

    feat: add new authentication system
    
    This commit adds a comprehensive authentication system
    with support for multiple providers.

commit b2c3d4e5f6789012345678901234567890abcde1
Author: Jane Smith <jane@example.com>
Date:   Sun Nov 30 15:30:00 2023 +0000

    fix: resolve memory leak in cache manager"""
            
            # ACT: Parse log output
            commits = parse_git_log_output(log_output)
            
            # ASSERT: Should parse all commit information
            assert len(commits) == 2
            assert commits[0].hash == "a1b2c3d4e5f6789012345678901234567890abcd"
            assert commits[0].author_name == "John Doe"
            assert commits[0].author_email == "john@example.com"
            assert "feat: add new authentication system" in commits[0].message


class TestErrorHandling:
    """Test specifications for error handling primitives."""

    def test_format_git_error_should_create_readable_errors(self):
        """format_git_error should create human-readable error messages."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Raw git error
            raw_error = "fatal: not a git repository (or any of the parent directories): .git"
            command = ["git", "status"]
            repo_path = "/tmp/not-repo"
            
            # ACT: Format error
            formatted = format_git_error(raw_error, command, repo_path)
            
            # ASSERT: Should provide helpful error message
            assert "git repository" in formatted.message.lower()
            assert repo_path in formatted.context
            assert formatted.command == command
            assert formatted.suggestion is not None

    def test_git_command_error_should_provide_debug_info(self):
        """GitCommandError should provide comprehensive debug information."""
        if PRIMITIVES_AVAILABLE:
            # ACT & ASSERT: Should provide debug context
            try:
                raise GitCommandError("Command failed", command=["git", "status"], repo_path="/tmp/repo")
            except GitCommandError as e:
                assert hasattr(e, "command")
                assert hasattr(e, "repo_path")
                assert hasattr(e, "return_code")
                assert hasattr(e, "stderr")

    def test_git_repository_error_should_indicate_repo_issues(self):
        """GitRepositoryError should clearly indicate repository-related issues."""
        if PRIMITIVES_AVAILABLE:
            # ACT & ASSERT: Should provide repository context
            try:
                raise GitRepositoryError("Repository not found", repo_path="/tmp/missing")
            except GitRepositoryError as e:
                assert hasattr(e, "repo_path")
                assert hasattr(e, "suggested_action")

    def test_git_validation_error_should_provide_validation_context(self):
        """GitValidationError should provide detailed validation context."""
        if PRIMITIVES_AVAILABLE:
            # ACT & ASSERT: Should provide validation details
            try:
                raise GitValidationError("Invalid branch name", field="branch", value="invalid..name")
            except GitValidationError as e:
                assert hasattr(e, "field")
                assert hasattr(e, "value")
                assert hasattr(e, "validation_rule")


# Integration tests to ensure primitives work together
class TestPrimitiveIntegration:
    """Test specifications for primitive integration."""

    def test_primitives_should_compose_cleanly(self):
        """Git primitives should compose together without conflicts."""
        if PRIMITIVES_AVAILABLE:
            # ARRANGE: Mock repository
            repo_path = "/tmp/test-repo"
            
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.is_dir", return_value=True),
                patch("pathlib.Path.glob", return_value=[Path(".git")]),
            ):
                # ACT: Use multiple primitives together
                is_valid = is_git_repository(repo_path)
                validated_path = validate_repository_path(repo_path)
                
                # ASSERT: Should work together seamlessly
                assert is_valid is True
                assert validated_path.is_valid is True

    def test_error_handling_should_be_consistent(self):
        """All primitives should handle errors consistently."""
        if PRIMITIVES_AVAILABLE:
            # Test that all functions handle invalid repository paths consistently
            invalid_path = "/does/not/exist"
            
            # All these should raise appropriate errors
            with pytest.raises((GitRepositoryError, GitValidationError)):
                get_repository_status(invalid_path)
            
            with pytest.raises((GitRepositoryError, GitValidationError)):
                get_current_branch(invalid_path)
            
            with pytest.raises((GitRepositoryError, GitValidationError)):
                get_commit_hash(invalid_path)


# Mark all tests that will initially fail (RED phase)
pytestmark = [pytest.mark.unit, pytest.mark.primitives]