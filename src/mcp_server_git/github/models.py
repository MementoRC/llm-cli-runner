"""Pydantic models for GitHub API tools"""

from pydantic import BaseModel


class GitHubGetPRChecks(BaseModel):
    repo_owner: str
    repo_name: str
    pr_number: int
    status: str | None = None
    conclusion: str | None = None


class GitHubGetFailingJobs(BaseModel):
    repo_owner: str
    repo_name: str
    pr_number: int
    include_logs: bool = True
    include_annotations: bool = True


class GitHubGetWorkflowRun(BaseModel):
    repo_owner: str
    repo_name: str
    run_id: int
    include_logs: bool = False


class GitHubGetPRDetails(BaseModel):
    repo_owner: str
    repo_name: str
    pr_number: int
    include_files: bool = False
    include_reviews: bool = False


class GitHubListPullRequests(BaseModel):
    repo_owner: str
    repo_name: str
    state: str = "open"
    head: str | None = None
    base: str | None = None
    sort: str = "created"
    direction: str = "desc"
    per_page: int = 30
    page: int = 1


class GitHubGetPRStatus(BaseModel):
    repo_owner: str
    repo_name: str
    pr_number: int


class GitHubGetPRFiles(BaseModel):
    repo_owner: str
    repo_name: str
    pr_number: int
    per_page: int = 30
    page: int = 1
    include_patch: bool = False


# GitHub CLI Models
class GitHubCLICreatePR(BaseModel):
    repo_path: str
    title: str
    body: str | None = None
    base: str | None = None
    head: str | None = None
    draft: bool = False
    web: bool = False


class GitHubCLIEditPR(BaseModel):
    repo_path: str
    pr_number: int
    title: str | None = None
    body: str | None = None
    base: str | None = None
    add_assignee: list[str] | None = None
    remove_assignee: list[str] | None = None
    add_label: list[str] | None = None
    remove_label: list[str] | None = None
    add_reviewer: list[str] | None = None
    remove_reviewer: list[str] | None = None


class GitHubCLIMergePR(BaseModel):
    repo_path: str
    pr_number: int
    merge_method: str = "merge"  # merge, squash, rebase
    delete_branch: bool = False
    auto: bool = False


class GitHubCLIClosePR(BaseModel):
    repo_path: str
    pr_number: int
    comment: str | None = None


class GitHubCLIReopenPR(BaseModel):
    repo_path: str
    pr_number: int
    comment: str | None = None


class GitHubCLIReadyPR(BaseModel):
    repo_path: str
    pr_number: int


# GitHub Issues Models
class GitHubCreateIssue(BaseModel):
    repo_owner: str
    repo_name: str
    title: str
    body: str | None = None
    labels: list[str] | None = None
    assignees: list[str] | None = None
    milestone: int | None = None


class GitHubListIssues(BaseModel):
    repo_owner: str
    repo_name: str
    state: str = "open"  # open, closed, all
    labels: list[str] | None = None
    assignee: str | None = None
    creator: str | None = None
    mentioned: str | None = None
    milestone: str | None = None
    sort: str = "created"  # created, updated, comments
    direction: str = "desc"  # asc, desc
    since: str | None = None
    per_page: int = 30
    page: int = 1


class GitHubUpdateIssue(BaseModel):
    repo_owner: str
    repo_name: str
    issue_number: int
    title: str | None = None
    body: str | None = None
    state: str | None = None  # open, closed
    labels: list[str] | None = None
    assignees: list[str] | None = None
    milestone: int | None = None


class GitHubEditPRDescription(BaseModel):
    repo_owner: str
    repo_name: str
    pr_number: int
    description: str
