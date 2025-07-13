"""Pydantic models for Git operations"""

from typing import Union
from pydantic import BaseModel


class GitStatus(BaseModel):
    repo_path: str
    porcelain: bool = False


class GitDiffUnstaged(BaseModel):
    repo_path: str
    stat_only: Union[bool, None] = False
    max_lines: Union[int, None] = None


class GitDiffStaged(BaseModel):
    repo_path: str
    stat_only: Union[bool, None] = False
    max_lines: Union[int, None] = None


class GitDiff(BaseModel):
    repo_path: str
    target: str
    stat_only: Union[bool, None] = False
    max_lines: Union[int, None] = None


class GitCommit(BaseModel):
    repo_path: str
    message: str
    gpg_sign: bool = False
    gpg_key_id: Union[str, None] = None


class GitAdd(BaseModel):
    repo_path: str
    files: list[str]


class GitReset(BaseModel):
    repo_path: str
    mode: Union[str, None] = None  # --soft, --mixed, --hard
    target: Union[str, None] = None  # commit hash, branch, tag
    files: Union[list[str], None] = None  # specific files to reset


class GitLog(BaseModel):
    repo_path: str
    max_count: int = 10
    oneline: bool = False
    graph: bool = False
    format: Union[str, None] = None


class GitCreateBranch(BaseModel):
    repo_path: str
    branch_name: str
    base_branch: Union[str, None] = None


class GitCheckout(BaseModel):
    repo_path: str
    branch_name: str


class GitShow(BaseModel):
    repo_path: str
    revision: str
    stat_only: Union[bool, None] = False
    max_lines: Union[int, None] = None


class GitInit(BaseModel):
    repo_path: str


class GitPush(BaseModel):
    repo_path: str
    remote: str = "origin"
    branch: Union[str, None] = None
    set_upstream: bool = False
    force: bool = False


class GitPull(BaseModel):
    repo_path: str
    remote: str = "origin"
    branch: Union[str, None] = None


class GitDiffBranches(BaseModel):
    repo_path: str
    base_branch: str
    compare_branch: str
    stat_only: Union[bool, None] = False
    max_lines: Union[int, None] = None


class GitRebase(BaseModel):
    repo_path: str
    target_branch: str


class GitMerge(BaseModel):
    repo_path: str
    source_branch: str
    strategy: str = "merge"
    message: Union[str, None] = None


class GitCherryPick(BaseModel):
    repo_path: str
    commit_hash: str
    no_commit: bool = False


class GitAbort(BaseModel):
    repo_path: str
    operation: str


class GitContinue(BaseModel):
    repo_path: str
    operation: str


class GitSecurityValidate(BaseModel):
    repo_path: str


class GitSecurityEnforce(BaseModel):
    repo_path: str
    strict_mode: bool = True
