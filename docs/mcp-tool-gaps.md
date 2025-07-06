# MCP Git Server - Missing Tool Analysis

This document tracks missing MCP tools identified during real-world usage of the MCP Git Server during LLM compliance TDD framework development.

## Missing GitHub Issue Management Tools

### 1. github_create_issue
**Status**: MISSING  
**Priority**: HIGH  
**Use Case**: Create GitHub issues to track bugs, feature requests, and missing functionality

```python
# Needed tool
mcp__git__github_create_issue(
    repo_owner="owner",
    repo_name="repo",
    title="Issue title",
    body="Issue description",
    labels=["bug", "enhancement"],
    assignees=["username"]
)
```

### 2. github_edit_pr_description  
**Status**: MISSING  
**Priority**: MEDIUM  
**Use Case**: Update PR descriptions with progress, milestones, and status

```python
# Needed tool
mcp__git__github_edit_pr_description(
    repo_owner="owner", 
    repo_name="repo",
    pr_number=12,
    description="Updated PR description"
)
```

### 3. github_list_issues
**Status**: MISSING
**Priority**: MEDIUM
**Use Case**: List and filter GitHub issues

```python
# Needed tool
mcp__git__github_list_issues(
    repo_owner="owner",
    repo_name="repo", 
    state="open",
    labels=["bug"],
    assignee="username"
)
```

### 4. github_update_issue
**Status**: MISSING
**Priority**: MEDIUM  
**Use Case**: Update issue status, labels, assignees

```python
# Needed tool
mcp__git__github_update_issue(
    repo_owner="owner",
    repo_name="repo",
    issue_number=42,
    state="closed",
    labels=["resolved"]
)
```

## Tool Gaps Identified During Usage

### Real-World Usage Scenario
During Task 19 (Git Primitive Operations) development:

1. **Needed to update PR description** with progress
   - **Workaround Used**: `gh pr edit 12 --body`
   - **Should Use**: `mcp__git__github_edit_pr_description`

2. **Needed to create GitHub issues** for missing functionality  
   - **Workaround Used**: Manual GitHub web interface
   - **Should Use**: `mcp__git__github_create_issue`

3. **Needed to track missing tools** systematically
   - **Workaround Used**: This documentation file
   - **Should Use**: GitHub issues with proper labeling

## Impact Analysis

### Developer Experience Impact
- **Inconsistent tooling**: Mixing MCP tools with bash commands
- **Missing dogfooding**: Not using our own MCP server for all Git/GitHub operations
- **Reduced confidence**: Can't validate MCP server completeness through usage

### Quality Impact  
- **Missing test coverage**: Tools we don't have can't be tested
- **Incomplete API surface**: GitHub API not fully exposed through MCP
- **Workflow gaps**: Common GitHub workflows require bash fallbacks

## Recommended Implementation Priority

### High Priority
1. `github_create_issue` - Essential for tracking and collaboration
2. `github_edit_pr_description` - Common workflow need

### Medium Priority  
3. `github_list_issues` - Issue management workflows
4. `github_update_issue` - Issue lifecycle management
5. `github_add_issue_comment` - Issue collaboration

### Low Priority
6. `github_assign_issue` - Advanced issue management
7. `github_label_issue` - Issue organization
8. `github_milestone_operations` - Project management

## Next Steps

1. **Create GitHub issues** for each missing tool (using web interface as workaround)
2. **Implement high-priority tools** in upcoming tasks
3. **Establish dogfooding principle**: Use only MCP tools for Git/GitHub operations
4. **Update this document** as tools are implemented

## Dogfooding Commitment

Going forward, we commit to:
- ✅ Use MCP tools exclusively for Git/GitHub operations
- ✅ Document any missing functionality immediately  
- ✅ Create GitHub issues for missing tools
- ✅ Prioritize implementing tools we need for our own development

This ensures our MCP server is battle-tested and complete for real-world usage.