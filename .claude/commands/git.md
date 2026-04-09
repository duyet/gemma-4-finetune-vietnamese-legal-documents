---
description: Git automation for commit and push
---

Git automation with semantic commit messages.

**Usage:**
```bash
./scripts/git_sync.sh [command] [options]
```

**Commands:**
```bash
# Interactive mode
./scripts/git_sync.sh

# Quick sync (commit + push)
./scripts/git_sync.sh sync

# Commit with scope
./scripts/git_sync.sh commit crawler "Add parallel crawling"

# Commit with custom message
./scripts/git_sync.sh commit "Update README"

# Push only
./scripts/git_sync.sh push

# Show status
./scripts/git_sync.sh status
```

**Commit Scopes:**
- `crawler` - Crawler changes
- `data` - Data processing
- `rag` - RAG pipeline
- `docs` - Documentation
- `chore` - Maintenance
- `feat` - New feature
- `fix` - Bug fix

**Examples:**
```bash
# After crawler changes
./scripts/git_sync.sh commit crawler "Add state persistence"

# After training
./scripts/git_sync.sh commit train "Update Colab notebooks"

# Full workflow
./scripts/git_sync.sh sync
```

**Environment:**
- Uses semantic commit format
- Auto-stages all changes
- Pushes to current branch
- Sets upstream if needed
