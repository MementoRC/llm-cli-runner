#!/bin/bash
# Script to remove development files before merging to stable branches
# Usage: ./scripts/cleanup-dev-files.sh

echo "🧹 Cleaning up development files..."

# Store current branch
CURRENT_BRANCH=$(git branch --show-current)

# Check if we're on a protected branch
if [[ "$CURRENT_BRANCH" == "main" ]] || [[ "$CURRENT_BRANCH" == "development" ]]; then
    echo "⚠️  WARNING: You are on a protected branch ($CURRENT_BRANCH)"
    read -p "Are you sure you want to remove development files? (y/N): " confirm
    if [[ "$confirm" != "y" ]]; then
        echo "Cleanup cancelled."
        exit 0
    fi
fi

# Remove development files
echo "Removing .taskmaster/..."
rm -rf .taskmaster/

echo "Removing .claude/..."
rm -rf .claude/

echo "Removing .mcp.json..."
rm -f .mcp.json

# Check what was removed
CHANGES=$(git status --porcelain | grep -E "(\.taskmaster|\.claude|\.mcp\.json)")

if [[ -n "$CHANGES" ]]; then
    echo "✅ Development files removed:"
    echo "$CHANGES"
    echo ""
    echo "To commit these changes, run:"
    echo "  git add -A"
    echo "  git commit -m 'chore: remove development files before merge'"
else
    echo "ℹ️  No development files found to remove"
fi

echo "🎯 Cleanup complete!"
