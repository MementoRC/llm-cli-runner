"""
Unit tests for GitHub primitive operations.

This module provides comprehensive unit tests for the GitHub primitives module,
testing all public functions, error conditions, and integration points.

Critical for TDD Compliance:
    These tests define the interface that github_primitives must implement.
    DO NOT modify these tests to match implementation - the implementation
    must satisfy these test requirements to prevent LLM compliance issues.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from mcp_server_git.primitives.github_primitives import (
    GitHubAPIError,
    GitHubAuthenticationError,
    GitHubPrimitiveError,
    GitHubRateLimitError,
    build_github_headers,
    build_github_url,
    check_repository_access,
    format_github_error,
    get_authenticated_user,
    get_commit_info,
    get_file_content,
    get_github_token,
    get_pull_request_info,
    get_repository_info,
    list_repository_contents,
    make_github_request,
    parse_github_url,
    search_repositories,
    validate_github_token,
)


class TestGitHubExceptions:
    """Test GitHub primitive exception classes."""

    def test_github_primitive_error_base(self):
        """Test GitHubPrimitiveError base exception."""
        error = GitHubPrimitiveError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_github_authentication_error(self):
        """Test GitHubAuthenticationError exception."""
        error = GitHubAuthenticationError("Auth failed")
        assert str(error) == "Auth failed"
        assert isinstance(error, GitHubPrimitiveError)

    def test_github_rate_limit_error(self):
        """Test GitHubRateLimitError exception."""
        error = GitHubRateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, GitHubPrimitiveError)

    def test_github_api_error_basic(self):
        """Test GitHubAPIError without status code."""
        error = GitHubAPIError("API error")
        assert str(error) == "API error"
        assert error.status_code is None
        assert isinstance(error, GitHubPrimitiveError)

    def test_github_api_error_with_status(self):
        """Test GitHubAPIError with status code."""
        error = GitHubAPIError("Not found", status_code=404)
        assert str(error) == "Not found"
        assert error.status_code == 404


class TestTokenHandling:
    """Test GitHub token validation and retrieval."""

    def test_get_github_token_found(self):
        """Test getting GitHub token from environment."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
            token = get_github_token()
            assert token == "test_token"

    def test_get_github_token_not_found(self):
        """Test getting GitHub token when not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            token = get_github_token()
            assert token is None

    def test_validate_github_token_valid_classic(self):
        """Test validation of classic personal access token."""
        token = "ghp_" + "1" * 36
        assert validate_github_token(token) is True

    def test_validate_github_token_valid_fine_grained(self):
        """Test validation of fine-grained personal access token."""
        token = "github_pat_" + "a" * 82
        assert validate_github_token(token) is True

    def test_validate_github_token_valid_app_installation(self):
        """Test validation of GitHub App installation token."""
        token = "ghs_" + "1" * 36
        assert validate_github_token(token) is True

    def test_validate_github_token_valid_app_user(self):
        """Test validation of GitHub App user token."""
        token = "ghu_" + "1" * 36
        assert validate_github_token(token) is True

    def test_validate_github_token_invalid_format(self):
        """Test validation of invalid token format."""
        assert validate_github_token("invalid_token") is False
        assert validate_github_token("ghp_short") is False
        assert validate_github_token("") is False
        assert validate_github_token("   ") is False

    def test_validate_github_token_none(self):
        """Test validation with None token."""
        assert validate_github_token(None) is False


class TestRequestBuilding:
    """Test GitHub request building utilities."""

    def test_build_github_headers_valid_token(self):
        """Test building GitHub headers with valid token."""
        token = "ghp_" + "1" * 36
        headers = build_github_headers(token)

        assert headers["Authorization"] == f"Bearer {token}"
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["User-Agent"] == "MCP-Git-Server/1.1.0"

    def test_build_github_headers_invalid_token(self):
        """Test building GitHub headers with invalid token."""
        with pytest.raises(GitHubAuthenticationError):
            build_github_headers("invalid_token")

    def test_build_github_url_default_base(self):
        """Test building GitHub URL with default base."""
        url = build_github_url("/repos/owner/repo")
        assert url == "https://api.github.com/repos/owner/repo"

    def test_build_github_url_custom_base(self):
        """Test building GitHub URL with custom base."""
        url = build_github_url("/user", "https://api.github.example.com")
        assert url == "https://api.github.example.com/user"

    def test_build_github_url_strips_slashes(self):
        """Test URL building strips leading/trailing slashes properly."""
        url = build_github_url("repos/owner/repo/", "https://api.github.com/")
        assert url == "https://api.github.com/repos/owner/repo"


class TestGitHubRequests:
    """Test GitHub API request functionality."""

    @pytest.mark.asyncio
    async def test_make_github_request_success(self):
        """Test successful GitHub API request."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"login": "testuser"})
        mock_response.text = AsyncMock(return_value='{"login": "testuser"}')

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            # Mock the request method to return an async context manager
            mock_request_cm = AsyncMock()
            mock_request_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_request_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.request = MagicMock(return_value=mock_request_cm)

            mock_session_class.return_value = mock_session

            with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_" + "1" * 36}):
                result = await make_github_request("GET", "/user")

            assert result == {"login": "testuser"}

    @pytest.mark.asyncio
    async def test_make_github_request_no_token(self):
        """Test GitHub request without token."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(GitHubAuthenticationError):
                await make_github_request("GET", "/user")

    @pytest.mark.asyncio
    async def test_make_github_request_auth_error(self):
        """Test GitHub request with authentication error."""
        mock_response = MagicMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            # Mock the request method to return an async context manager
            mock_request_cm = AsyncMock()
            mock_request_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_request_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.request = MagicMock(return_value=mock_request_cm)

            mock_session_class.return_value = mock_session

            with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_" + "1" * 36}):
                with pytest.raises(GitHubAuthenticationError):
                    await make_github_request("GET", "/user")

    @pytest.mark.asyncio
    async def test_make_github_request_rate_limit(self):
        """Test GitHub request with rate limit error."""
        mock_response = MagicMock()
        mock_response.status = 403
        mock_response.text = AsyncMock(return_value="rate limit exceeded")
        mock_response.headers = {"X-RateLimit-Reset": "1234567890"}

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            # Mock the request method to return an async context manager
            mock_request_cm = AsyncMock()
            mock_request_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_request_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.request = MagicMock(return_value=mock_request_cm)

            mock_session_class.return_value = mock_session

            with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_" + "1" * 36}):
                with pytest.raises(GitHubRateLimitError):
                    await make_github_request("GET", "/user")

    @pytest.mark.asyncio
    async def test_make_github_request_api_error(self):
        """Test GitHub request with general API error."""
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not Found")

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            # Mock the request method to return an async context manager
            mock_request_cm = AsyncMock()
            mock_request_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_request_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.request = MagicMock(return_value=mock_request_cm)

            mock_session_class.return_value = mock_session

            with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_" + "1" * 36}):
                with pytest.raises(GitHubAPIError) as exc_info:
                    await make_github_request("GET", "/unknown")

                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_make_github_request_network_error(self):
        """Test GitHub request with network error."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.request = MagicMock(
                side_effect=aiohttp.ClientError("Network error")
            )
            mock_session_class.return_value = mock_session

            with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_" + "1" * 36}):
                with pytest.raises(GitHubAPIError):
                    await make_github_request("GET", "/user")

    @pytest.mark.asyncio
    async def test_make_github_request_with_params(self):
        """Test GitHub request with URL parameters."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"items": []})
        mock_response.text = AsyncMock(return_value='{"items": []}')

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            # Mock the request method to return an async context manager
            mock_request_cm = AsyncMock()
            mock_request_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_request_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.request = MagicMock(return_value=mock_request_cm)

            mock_session_class.return_value = mock_session

            with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_" + "1" * 36}):
                result = await make_github_request(
                    "GET", "/search/repositories", params={"q": "test"}
                )

            assert result == {"items": []}

    @pytest.mark.asyncio
    async def test_make_github_request_with_json_data(self):
        """Test GitHub request with JSON data."""
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={"id": 123, "title": "Test Issue"})
        mock_response.text = AsyncMock(
            return_value='{"id": 123, "title": "Test Issue"}'
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            # Mock the request method to return an async context manager
            mock_request_cm = AsyncMock()
            mock_request_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_request_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.request = MagicMock(return_value=mock_request_cm)

            mock_session_class.return_value = mock_session

            with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_" + "1" * 36}):
                result = await make_github_request(
                    "POST",
                    "/repos/owner/repo/issues",
                    json_data={"title": "Test Issue"},
                )

            assert result == {"id": 123, "title": "Test Issue"}


class TestHighLevelOperations:
    """Test high-level GitHub API operations."""

    @pytest.mark.asyncio
    async def test_get_authenticated_user(self):
        """Test getting authenticated user information."""
        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = {"login": "testuser", "id": 12345}

            result = await get_authenticated_user()

            assert result == {"login": "testuser", "id": 12345}
            mock_request.assert_called_once_with("GET", "/user")

    @pytest.mark.asyncio
    async def test_check_repository_access_success(self):
        """Test checking repository access when access is granted."""
        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = {"full_name": "owner/repo"}

            result = await check_repository_access("owner", "repo")

            assert result is True
            mock_request.assert_called_once_with("GET", "/repos/owner/repo")

    @pytest.mark.asyncio
    async def test_check_repository_access_denied(self):
        """Test checking repository access when access is denied."""
        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.side_effect = GitHubAPIError("Not found", status_code=404)

            result = await check_repository_access("owner", "repo")

            assert result is False

    @pytest.mark.asyncio
    async def test_get_repository_info(self):
        """Test getting repository information."""
        repo_data = {
            "full_name": "owner/repo",
            "default_branch": "main",
            "private": False,
        }

        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = repo_data

            result = await get_repository_info("owner", "repo")

            assert result == repo_data
            mock_request.assert_called_once_with("GET", "/repos/owner/repo")

    @pytest.mark.asyncio
    async def test_get_pull_request_info(self):
        """Test getting pull request information."""
        pr_data = {
            "number": 123,
            "title": "Test PR",
            "state": "open",
            "user": {"login": "author"},
        }

        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = pr_data

            result = await get_pull_request_info("owner", "repo", 123)

            assert result == pr_data
            mock_request.assert_called_once_with("GET", "/repos/owner/repo/pulls/123")

    @pytest.mark.asyncio
    async def test_get_commit_info(self):
        """Test getting commit information."""
        commit_data = {
            "sha": "abc123",
            "commit": {"author": {"name": "Test Author"}, "message": "Test commit"},
        }

        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = commit_data

            result = await get_commit_info("owner", "repo", "abc123")

            assert result == commit_data
            mock_request.assert_called_once_with(
                "GET", "/repos/owner/repo/commits/abc123"
            )

    @pytest.mark.asyncio
    async def test_list_repository_contents_directory(self):
        """Test listing repository contents for directory."""
        contents_data = [
            {"type": "file", "name": "README.md"},
            {"type": "dir", "name": "src"},
        ]

        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = contents_data

            result = await list_repository_contents("owner", "repo", "src")

            assert result == contents_data
            mock_request.assert_called_once_with(
                "GET", "/repos/owner/repo/contents/src", params={}
            )

    @pytest.mark.asyncio
    async def test_list_repository_contents_file(self):
        """Test listing repository contents for single file."""
        file_data = {"type": "file", "name": "README.md", "content": "..."}

        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = file_data

            result = await list_repository_contents("owner", "repo", "README.md")

            assert result == [file_data]  # Should be wrapped in list

    @pytest.mark.asyncio
    async def test_list_repository_contents_with_ref(self):
        """Test listing repository contents with specific ref."""
        contents_data = [{"type": "file", "name": "README.md"}]

        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = contents_data

            result = await list_repository_contents(
                "owner", "repo", "src", ref="develop"
            )

            assert result == contents_data
            mock_request.assert_called_once_with(
                "GET", "/repos/owner/repo/contents/src", params={"ref": "develop"}
            )

    @pytest.mark.asyncio
    async def test_get_file_content(self):
        """Test getting file content."""
        file_data = {
            "type": "file",
            "name": "README.md",
            "content": "SGVsbG8gV29ybGQ=",  # "Hello World" in base64
        }

        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = file_data

            result = await get_file_content("owner", "repo", "README.md")

            assert result == file_data
            mock_request.assert_called_once_with(
                "GET", "/repos/owner/repo/contents/README.md", params={}
            )

    @pytest.mark.asyncio
    async def test_search_repositories(self):
        """Test searching repositories."""
        search_data = {
            "total_count": 1,
            "items": [{"full_name": "owner/repo", "stargazers_count": 100}],
        }

        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = search_data

            result = await search_repositories("language:python")

            assert result == search_data
            mock_request.assert_called_once_with(
                "GET",
                "/search/repositories",
                params={
                    "q": "language:python",
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 30,
                    "page": 1,
                },
            )

    @pytest.mark.asyncio
    async def test_search_repositories_with_custom_params(self):
        """Test searching repositories with custom parameters."""
        search_data = {"total_count": 0, "items": []}

        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = search_data

            result = await search_repositories(
                "test", sort="stars", order="asc", per_page=10, page=2
            )

            assert result == search_data
            mock_request.assert_called_once_with(
                "GET",
                "/search/repositories",
                params={
                    "q": "test",
                    "sort": "stars",
                    "order": "asc",
                    "per_page": 10,
                    "page": 2,
                },
            )

    @pytest.mark.asyncio
    async def test_search_repositories_per_page_limit(self):
        """Test that search repositories respects GitHub's per_page limit."""
        search_data = {"total_count": 0, "items": []}

        with patch(
            "mcp_server_git.primitives.github_primitives.make_github_request"
        ) as mock_request:
            mock_request.return_value = search_data

            await search_repositories("test", per_page=150)  # Above GitHub limit

            # Should be capped at 100
            mock_request.assert_called_once_with(
                "GET",
                "/search/repositories",
                params={
                    "q": "test",
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 100,
                    "page": 1,
                },
            )


class TestUtilityFunctions:
    """Test utility functions."""

    def test_parse_github_url_https(self):
        """Test parsing HTTPS GitHub URL."""
        url = "https://github.com/owner/repo"
        result = parse_github_url(url)
        assert result == {"owner": "owner", "repo": "repo"}

    def test_parse_github_url_https_with_git(self):
        """Test parsing HTTPS GitHub URL with .git suffix."""
        url = "https://github.com/owner/repo.git"
        result = parse_github_url(url)
        assert result == {"owner": "owner", "repo": "repo"}

    def test_parse_github_url_https_with_slash(self):
        """Test parsing HTTPS GitHub URL with trailing slash."""
        url = "https://github.com/owner/repo/"
        result = parse_github_url(url)
        assert result == {"owner": "owner", "repo": "repo"}

    def test_parse_github_url_ssh(self):
        """Test parsing SSH GitHub URL."""
        url = "git@github.com:owner/repo"
        result = parse_github_url(url)
        assert result == {"owner": "owner", "repo": "repo"}

    def test_parse_github_url_ssh_with_git(self):
        """Test parsing SSH GitHub URL with .git suffix."""
        url = "git@github.com:owner/repo.git"
        result = parse_github_url(url)
        assert result == {"owner": "owner", "repo": "repo"}

    def test_parse_github_url_invalid(self):
        """Test parsing invalid GitHub URL."""
        invalid_urls = [
            "https://gitlab.com/owner/repo",
            "https://github.com/owner",
            "invalid-url",
            "",
        ]

        for url in invalid_urls:
            result = parse_github_url(url)
            assert result is None

    def test_format_github_error_authentication(self):
        """Test formatting authentication error."""
        error = GitHubAuthenticationError("Invalid token")
        result = format_github_error(error)
        assert "🔒 Authentication failed" in result
        assert "Invalid token" in result

    def test_format_github_error_rate_limit(self):
        """Test formatting rate limit error."""
        error = GitHubRateLimitError("Rate limit exceeded")
        result = format_github_error(error)
        assert "⏱️ Rate limit exceeded" in result

    def test_format_github_error_api_with_status(self):
        """Test formatting API error with status code."""
        error = GitHubAPIError("Not found", status_code=404)
        result = format_github_error(error)
        assert "❌ GitHub API error (HTTP 404)" in result
        assert "Not found" in result

    def test_format_github_error_api_without_status(self):
        """Test formatting API error without status code."""
        error = GitHubAPIError("General error")
        result = format_github_error(error)
        assert "❌ GitHub API error: General error" in result
        assert "(HTTP " not in result

    def test_format_github_error_unexpected(self):
        """Test formatting unexpected error."""
        error = ValueError("Unexpected error")
        result = format_github_error(error)
        assert "💥 Unexpected error" in result
        assert "Unexpected error" in result
