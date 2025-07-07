"""
Unit tests for Git operations module.

These tests verify the higher-level Git operations that build on primitives
to provide complex functionality. Tests focus on operation composition,
error handling, and business logic validation.

Critical for TDD Compliance:
    These tests define the behavior that the implementation must satisfy.
    DO NOT modify these tests to match a broken implementation - the
    implementation must be fixed to pass these tests.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, call

from mcp_server_git.operations.git_operations import (
    CommitRequest,
    BranchRequest,
    MergeRequest,
    commit_changes_with_validation,
    create_branch_with_checkout,
    merge_branches_with_conflict_detection,
    push_with_validation,
)

from mcp_server_git.primitives.git_primitives import (
    GitCommandError,
    GitRepositoryError,
)


class TestCommitChangesWithValidation:
    """Test commit operations with comprehensive validation."""

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_repository_status')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    @patch('mcp_server_git.operations.git_operations.get_commit_hash')
    def test_successful_commit_with_files(
        self, mock_get_commit_hash, mock_execute_git, mock_get_status, mock_validate_repo
    ):
        """Should successfully commit specified files with validation."""
        # Arrange
        repo_path = Path("/test/repo")
        commit_request = CommitRequest(
            message="feat: add new feature",
            files=["src/feature.py", "tests/test_feature.py"],
            author="Developer",
            email="dev@example.com"
        )
        
        mock_get_commit_hash.return_value = "abc123def456"
        
        # Act
        result = commit_changes_with_validation(repo_path, commit_request)
        
        # Assert
        assert result.success is True
        assert result.commit_hash == "abc123def456"
        assert result.message == "Successfully committed: feat: add new feature"
        assert result.files_committed == ["src/feature.py", "tests/test_feature.py"]
        assert result.error is None
        
        # Verify primitive calls
        mock_validate_repo.assert_called_once_with(str(repo_path))
        mock_get_status.assert_called_once_with(str(repo_path))
        
        # Verify file staging
        expected_add_calls = [
            call(["add", "src/feature.py"], str(repo_path)),
            call(["add", "tests/test_feature.py"], str(repo_path)),
        ]
        
        # Verify commit command
        expected_commit_call = call([
            "commit", "-m", "feat: add new feature", 
            "--author", "Developer <dev@example.com>"
        ], str(repo_path))
        
        # Check all execute_git_command calls
        actual_calls = mock_execute_git.call_args_list
        assert expected_add_calls[0] in actual_calls
        assert expected_add_calls[1] in actual_calls
        assert expected_commit_call in actual_calls

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_repository_status')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    @patch('mcp_server_git.operations.git_operations.get_commit_hash')
    def test_commit_with_gpg_signing(
        self, mock_get_commit_hash, mock_execute_git, mock_get_status, mock_validate_repo
    ):
        """Should create GPG-signed commit when requested."""
        # Arrange
        repo_path = Path("/test/repo")
        commit_request = CommitRequest(
            message="feat: secure feature",
            gpg_sign=True,
            gpg_key_id="ABC123"
        )
        
        mock_get_commit_hash.return_value = "def456abc789"
        
        # Act
        result = commit_changes_with_validation(repo_path, commit_request)
        
        # Assert
        assert result.success is True
        assert result.commit_hash == "def456abc789"
        
        # Verify GPG signing in commit command
        expected_commit_call = call([
            "commit", "-m", "feat: secure feature", 
            "--gpg-sign", "-S", "ABC123"
        ], str(repo_path))
        
        assert expected_commit_call in mock_execute_git.call_args_list

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_repository_status')
    def test_commit_fails_with_empty_message(self, mock_get_status, mock_validate_repo):
        """Should fail validation when commit message is empty."""
        # Arrange
        repo_path = Path("/test/repo")
        commit_request = CommitRequest(message="   ")  # Empty/whitespace message
        
        # Act
        result = commit_changes_with_validation(repo_path, commit_request)
        
        # Assert
        assert result.success is False
        assert "Commit message cannot be empty" in result.error
        assert result.commit_hash is None

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_repository_status')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    def test_commit_fails_when_file_staging_fails(
        self, mock_execute_git, mock_get_status, mock_validate_repo
    ):
        """Should fail gracefully when file staging fails."""
        # Arrange
        repo_path = Path("/test/repo")
        commit_request = CommitRequest(
            message="feat: add feature",
            files=["nonexistent.py"]
        )
        
        mock_execute_git.side_effect = GitCommandError("File not found: nonexistent.py")
        
        # Act
        result = commit_changes_with_validation(repo_path, commit_request)
        
        # Assert
        assert result.success is False
        assert "Failed to stage file nonexistent.py" in result.error
        assert result.commit_hash is None

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    def test_commit_fails_with_invalid_repository(self, mock_validate_repo):
        """Should fail when repository validation fails."""
        # Arrange
        repo_path = Path("/invalid/repo")
        commit_request = CommitRequest(message="feat: add feature")
        
        mock_validate_repo.side_effect = GitRepositoryError("Not a git repository")
        
        # Act
        result = commit_changes_with_validation(repo_path, commit_request)
        
        # Assert
        assert result.success is False
        assert "Not a git repository" in result.error
        assert result.commit_hash is None


class TestCreateBranchWithCheckout:
    """Test branch creation operations with checkout."""

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    def test_successful_branch_creation_with_checkout(
        self, mock_execute_git, mock_get_current_branch, mock_validate_repo
    ):
        """Should successfully create and checkout new branch."""
        # Arrange
        repo_path = Path("/test/repo")
        branch_request = BranchRequest(
            name="feature/new-feature",
            base_branch="main",
            checkout=True
        )
        
        mock_get_current_branch.return_value = "main"
        mock_execute_git.side_effect = [
            GitCommandError("Branch not found"),  # show-ref check (branch doesn't exist)
            None  # checkout -b command succeeds
        ]
        
        # Act
        result = create_branch_with_checkout(repo_path, branch_request)
        
        # Assert
        assert result.success is True
        assert result.branch_name == "feature/new-feature"
        assert result.previous_branch == "main"
        assert result.message == "Successfully created branch: feature/new-feature"
        assert result.error is None
        
        # Verify branch creation command
        expected_checkout_call = call([
            "checkout", "-b", "feature/new-feature", "main"
        ], str(repo_path))
        
        assert expected_checkout_call in mock_execute_git.call_args_list

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    def test_branch_creation_without_checkout(
        self, mock_execute_git, mock_get_current_branch, mock_validate_repo
    ):
        """Should create branch without checking out when requested."""
        # Arrange
        repo_path = Path("/test/repo")
        branch_request = BranchRequest(
            name="feature/background-task",
            base_branch="develop",
            checkout=False
        )
        
        mock_get_current_branch.return_value = "main"
        mock_execute_git.side_effect = [
            GitCommandError("Branch not found"),  # show-ref check
            None  # branch command succeeds
        ]
        
        # Act
        result = create_branch_with_checkout(repo_path, branch_request)
        
        # Assert
        assert result.success is True
        assert result.branch_name == "feature/background-task"
        
        # Verify branch creation command (not checkout)
        expected_branch_call = call([
            "branch", "feature/background-task", "develop"
        ], str(repo_path))
        
        assert expected_branch_call in mock_execute_git.call_args_list

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    def test_branch_creation_fails_when_branch_exists(
        self, mock_execute_git, mock_get_current_branch, mock_validate_repo
    ):
        """Should fail when branch already exists and force is not set."""
        # Arrange
        repo_path = Path("/test/repo")
        branch_request = BranchRequest(
            name="existing-branch",
            force=False
        )
        
        mock_get_current_branch.return_value = "main"
        mock_execute_git.return_value = None  # show-ref succeeds (branch exists)
        
        # Act
        result = create_branch_with_checkout(repo_path, branch_request)
        
        # Assert
        assert result.success is False
        assert "already exists" in result.error
        assert "force=True" in result.error

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    def test_branch_creation_fails_with_empty_name(self, mock_get_current_branch, mock_validate_repo):
        """Should fail validation when branch name is empty."""
        # Arrange
        repo_path = Path("/test/repo")
        branch_request = BranchRequest(name="  ")  # Empty/whitespace name
        
        mock_get_current_branch.return_value = "main"
        
        # Act
        result = create_branch_with_checkout(repo_path, branch_request)
        
        # Assert
        assert result.success is False
        assert "Branch name cannot be empty" in result.error


class TestMergeBranchesWithConflictDetection:
    """Test merge operations with conflict detection."""

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    @patch('mcp_server_git.operations.git_operations.get_commit_hash')
    def test_successful_merge_with_commit(
        self, mock_get_commit_hash, mock_execute_git, mock_get_current_branch, mock_validate_repo
    ):
        """Should successfully merge branches and create merge commit."""
        # Arrange
        repo_path = Path("/test/repo")
        merge_request = MergeRequest(
            source_branch="feature/new-feature",
            target_branch="main",
            message="Merge feature into main"
        )
        
        mock_get_current_branch.return_value = "main"
        mock_get_commit_hash.return_value = "merge123abc456"
        mock_execute_git.side_effect = [
            None,  # show-ref (source branch exists)
            None,  # merge command succeeds
        ]
        
        # Act
        result = merge_branches_with_conflict_detection(repo_path, merge_request)
        
        # Assert
        assert result.success is True
        assert result.merge_commit_hash == "merge123abc456"
        assert result.message == "Successfully merged 'feature/new-feature'"
        assert result.conflicts is None
        assert result.error is None
        
        # Verify merge command
        expected_merge_call = call([
            "merge", "-m", "Merge feature into main", "feature/new-feature"
        ], str(repo_path))
        
        assert expected_merge_call in mock_execute_git.call_args_list

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    @patch('mcp_server_git.operations.git_operations.get_repository_status')
    def test_merge_detects_conflicts(
        self, mock_get_status, mock_execute_git, mock_get_current_branch, mock_validate_repo
    ):
        """Should detect and report merge conflicts."""
        # Arrange
        repo_path = Path("/test/repo")
        merge_request = MergeRequest(
            source_branch="feature/conflicting-feature",
            target_branch="main"
        )
        
        mock_get_current_branch.return_value = "main"
        mock_execute_git.side_effect = [
            None,  # show-ref (source branch exists)
            GitCommandError("Merge conflict in src/file.py"),  # merge fails with conflict
        ]
        
        # Mock status to return conflicts
        mock_status = Mock()
        mock_status.conflicted_files = ["src/file.py", "src/other.py"]
        mock_get_status.return_value = mock_status
        
        # Act
        result = merge_branches_with_conflict_detection(repo_path, merge_request)
        
        # Assert
        assert result.success is False
        assert result.conflicts == ["src/file.py", "src/other.py"]
        assert "Merge conflicts detected" in result.error
        assert result.merge_commit_hash is None

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    def test_merge_fails_with_nonexistent_source_branch(
        self, mock_execute_git, mock_get_current_branch, mock_validate_repo
    ):
        """Should fail when source branch does not exist."""
        # Arrange
        repo_path = Path("/test/repo")
        merge_request = MergeRequest(
            source_branch="nonexistent-branch",
            target_branch="main"
        )
        
        mock_get_current_branch.return_value = "main"
        mock_execute_git.side_effect = GitCommandError("Branch not found")
        
        # Act
        result = merge_branches_with_conflict_detection(repo_path, merge_request)
        
        # Assert
        assert result.success is False
        assert "does not exist" in result.error

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    @patch('mcp_server_git.operations.git_operations.get_commit_hash')
    def test_merge_with_squash_option(
        self, mock_get_commit_hash, mock_execute_git, mock_get_current_branch, mock_validate_repo
    ):
        """Should perform squash merge when requested."""
        # Arrange
        repo_path = Path("/test/repo")
        merge_request = MergeRequest(
            source_branch="feature/small-feature",
            squash=True
        )
        
        mock_get_current_branch.return_value = "main"
        mock_execute_git.side_effect = [
            None,  # show-ref (source branch exists)
            None,  # merge command succeeds
        ]
        
        # Act
        result = merge_branches_with_conflict_detection(repo_path, merge_request)
        
        # Assert
        assert result.success is True
        # Squash merges don't create merge commits immediately
        assert result.merge_commit_hash is None
        
        # Verify squash option in merge command
        expected_merge_call = call([
            "merge", "--squash", "feature/small-feature"
        ], str(repo_path))
        
        assert expected_merge_call in mock_execute_git.call_args_list


class TestPushWithValidation:
    """Test push operations with comprehensive validation."""

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    @patch('mcp_server_git.operations.git_operations.get_repository_status')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    def test_successful_push_with_current_branch(
        self, mock_execute_git, mock_get_status, mock_get_current_branch, mock_validate_repo
    ):
        """Should successfully push current branch to remote."""
        # Arrange
        repo_path = Path("/test/repo")
        
        mock_get_current_branch.return_value = "main"
        mock_execute_git.side_effect = [
            "https://github.com/user/repo.git",  # remote get-url
            None,  # push command succeeds
        ]
        
        # Act
        result = push_with_validation(repo_path, remote="origin")
        
        # Assert
        assert result["success"] is True
        assert result["message"] == "Successfully pushed 'main' to 'origin'"
        assert result["remote"] == "origin"
        assert result["branch"] == "main"
        assert result["force"] is False
        assert result["set_upstream"] is False
        
        # Verify push command
        expected_push_call = call(["push", "origin", "main"], str(repo_path))
        assert expected_push_call in mock_execute_git.call_args_list

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    @patch('mcp_server_git.operations.git_operations.get_repository_status')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    def test_push_with_upstream_setting(
        self, mock_execute_git, mock_get_status, mock_get_current_branch, mock_validate_repo
    ):
        """Should set upstream tracking when requested."""
        # Arrange
        repo_path = Path("/test/repo")
        
        mock_get_current_branch.return_value = "feature/new-branch"
        mock_execute_git.side_effect = [
            "https://github.com/user/repo.git",  # remote get-url
            None,  # push command succeeds
        ]
        
        # Act
        result = push_with_validation(
            repo_path, 
            remote="origin", 
            branch="feature/new-branch",
            set_upstream=True
        )
        
        # Assert
        assert result["success"] is True
        assert result["set_upstream"] is True
        
        # Verify upstream setting in push command
        expected_push_call = call([
            "push", "--set-upstream", "origin", "feature/new-branch"
        ], str(repo_path))
        
        assert expected_push_call in mock_execute_git.call_args_list

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    def test_push_fails_with_nonexistent_remote(
        self, mock_execute_git, mock_get_current_branch, mock_validate_repo
    ):
        """Should fail when remote does not exist."""
        # Arrange
        repo_path = Path("/test/repo")
        
        mock_get_current_branch.return_value = "main"
        mock_execute_git.side_effect = GitCommandError("Remote 'invalid' not found")
        
        # Act
        result = push_with_validation(repo_path, remote="invalid")
        
        # Assert
        assert result["success"] is False
        assert "not found" in result["error"]
        assert result["remote"] == "invalid"

    @patch('mcp_server_git.operations.git_operations.validate_repository_path')
    @patch('mcp_server_git.operations.git_operations.get_current_branch')
    @patch('mcp_server_git.operations.git_operations.get_repository_status')
    @patch('mcp_server_git.operations.git_operations.execute_git_command')
    def test_force_push_when_requested(
        self, mock_execute_git, mock_get_status, mock_get_current_branch, mock_validate_repo
    ):
        """Should perform force push when explicitly requested."""
        # Arrange
        repo_path = Path("/test/repo")
        
        mock_get_current_branch.return_value = "main"
        mock_execute_git.side_effect = [
            "https://github.com/user/repo.git",  # remote get-url
            None,  # push command succeeds
        ]
        
        # Act
        result = push_with_validation(repo_path, force=True)
        
        # Assert
        assert result["success"] is True
        assert result["force"] is True
        
        # Verify force option in push command
        expected_push_call = call(["push", "--force", "origin", "main"], str(repo_path))
        assert expected_push_call in mock_execute_git.call_args_list


# Test fixtures and utilities for integration testing
@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary Git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    
    # Initialize git repo
    import subprocess
    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
    
    return repo_path


@pytest.fixture
def sample_commit_request():
    """Sample commit request for testing."""
    return CommitRequest(
        message="test: add sample feature",
        files=["test_file.py"],
        author="Test Author",
        email="test@example.com"
    )


@pytest.fixture
def sample_branch_request():
    """Sample branch request for testing."""
    return BranchRequest(
        name="test/sample-branch",
        base_branch="main",
        checkout=True
    )


@pytest.fixture
def sample_merge_request():
    """Sample merge request for testing."""
    return MergeRequest(
        source_branch="feature/test-feature",
        target_branch="main",
        message="Merge test feature"
    )


# Mark for test organization
pytestmark = [pytest.mark.unit, pytest.mark.operations]