# Pre-Merge Checklist

## Development Files Cleanup

When merging feature branches to `development` or `main`, the following development files should be removed:

### Files to Remove
- `.taskmaster/` - TaskMaster AI project management
- `.claude/` - Claude Code IDE settings and state
- `.mcp.json` - MCP server configuration

### Automated Cleanup

1. **GitHub Action** (Automatic)
   - Triggers on PR merge to `development` or `main`
   - Automatically removes development files
   - Commits cleanup with descriptive message

2. **Manual Script** (Local)
   ```bash
   ./scripts/cleanup-dev-files.sh
   ```

### Manual Process

If needed, manually remove files:
```bash
# Remove development files
rm -rf .taskmaster/
rm -rf .claude/
rm -f .mcp.json

# Stage and commit
git add -A
git commit -m "chore: remove development files before merge"
```

### Preserving Files in Feature Branches

These files are intentionally tracked in feature branches to:
- Maintain TaskMaster project state
- Preserve Claude Code configurations
- Keep MCP server settings

### Backup Before Merge

To backup development files before merge:
```bash
# Create backup
tar -czf dev-files-backup-$(date +%Y%m%d-%H%M%S).tar.gz .taskmaster/ .claude/ .mcp.json

# Or copy to a safe location
cp -r .taskmaster/ ~/.taskmaster-backup/
cp -r .claude/ ~/.claude-backup/
cp .mcp.json ~/.mcp-backup.json
```

## Verification Steps

1. Run all tests: `pixi run test`
2. Check code quality: `pixi run ruff check --select=F,E9`
3. Run pre-commit hooks: `eval "$(pixi shell-hook -e dev)" && pre-commit run --all-files`
4. Verify CI passes
5. Remove development files (automated or manual)
6. Final review before merge
