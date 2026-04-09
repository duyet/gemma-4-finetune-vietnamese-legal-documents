---
name: git-preferences
description: Git workflow preferences for this repository
type: feedback
---

# Git Workflow Preferences

## History Policy

**Keep detailed history** - Do NOT squash or rewrite commits.

**Why**: Maintains transparent development timeline, shows iterative fixes, preserves context for each change.

**How to apply**:
- Never use `git rebase` to squash commits
- Never use `git commit --amend` for public history
- Each logical change gets its own commit
- Fix commits are kept separate from feature commits

## Commit Authors

Use different commit authors based on who is making the change:

- **duyet** - Human commits (manual changes by user)
- **duyetbot** - Automated commits (CI/CD, scripts, automation)  
- **claude** - Claude Code commits (AI assistant changes)

**Why**: Clear attribution of commit provenance for project history

**How to apply**:
- When committing as Claude Code, use: `--author="claude <claude@anthropic.com>"`
- For automation scripts, use: `--author="duyetbot <bot@duyet.net>"`
- For human commits, use default git config (duyet)

## Commit Format

**Semantic commits with scope**:
```
scope: message

Scopes:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactoring
- chore: Maintenance
- test: Testing

Components:
- crawler: Crawler changes
- data: Data processing
- rag: RAG pipeline
- docs: Documentation
- train: Training
- config: Configuration
```
