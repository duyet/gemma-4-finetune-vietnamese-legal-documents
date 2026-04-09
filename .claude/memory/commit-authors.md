---
name: commit-authors
description: Git commit authors for this repository
type: feedback
---

# Git Commit Authors

Use different commit authors based on who is making the change:

- **duyet** - Human commits (manual changes by user)
- **duyetbot** - Automated commits (CI/CD, scripts, automation)
- **claude** - Claude Code commits (AI assistant changes)

**Why**: Clear attribution of commit provenance for project history

**How to apply**:
- When committing as Claude Code, use: `--author="claude <claude@anthropic.com>"`
- For automation scripts, use: `--author="duyetbot <bot@duyet.net>"`
- For human commits, use default git config (duyet)
