---
description: Maintain and monitor Gemma 4 Vietnamese Legal project health
---

# 🛠️ Repository Maintenance

Run regular maintenance tasks to keep the project healthy and up-to-date.

## Quick Health Check

```bash
# Check all systems
/maintenance check
```

**This will:**
- ✅ Check Python environment and dependencies
- ✅ Validate data directory structure
- ✅ Check crawler state database
- ✅ Verify git remotes configuration
- ✅ Check HuggingFace CLI status
- ✅ Validate notebook JSON format
- ✅ Show project statistics

---

## Maintenance Tasks

### 1. Environment Check

Check if all dependencies are properly installed:

```bash
# Check Python dependencies
uv sync --check

# Check Playwright (for crawler)
playwright install chromium --dry-run

# Check HuggingFace CLI
huggingface-cli whoami
```

**What it checks:**
- UV package dependencies
- Playwright browser installation
- HuggingFace authentication
- Git remote connectivity

### 2. Data Validation

Validate crawled and processed data:

```bash
# Check data directories
ls -lh data/*/ 2>/dev/null || echo "No data directories"

# Validate data if exists
if [ -f "data/raw/playwright_documents.jsonl" ]; then
    echo "Crawled documents: $(wc -l < data/raw/playwright_documents.jsonl)"
fi

if [ -f "data/processed/documents.parquet" ]; then
    echo "Processed documents: $(python -c "import pandas as pd; print(len(pd.read_parquet('data/processed/documents.parquet'))")"
fi
```

**What it validates:**
- Data directory structure exists
- Raw data files present
- Processed data available
- File sizes reasonable

### 3. Code Quality

Run code quality checks:

```bash
# Format code
uv run black scripts/ crawler/ rag/
uv run ruff check scripts/ crawler/ rag/

# Type checking (if mypy installed)
uv run mypy scripts/ 2>/dev/null || echo "mypy not installed"
```

**What it checks:**
- Python code formatting (Black)
- Linting (Ruff)
- Type hints (MyPy)

### 4. Documentation Updates

Update documentation timestamps and statistics:

```bash
# Update README statistics
python3 << 'EOF'
import json
from pathlib import Path

# Count documents
data_dir = Path("data/processed")
if data_dir.exists():
    parquet_files = list(data_dir.glob("**/*.parquet"))
    total_docs = sum(len(pd.read_parquet(f)) for f in parquet_files)
    print(f"Total documents: {total_docs:,}")
EOF
```

**What it updates:**
- README.md statistics
- CLAUDE.md timestamps
- Dataset counts

### 5. Git Health

Check git repository health:

```bash
# Check remotes
git remote -v

# Check for large files (should be in .gitignore)
find . -type f -size +50M ! -path "./.git/*" ! -path "./data/*" ! -path "./.venv/*" | head -10

# Check for uncommitted changes
git status --short
```

**What it checks:**
- Remote configuration
- Large files not ignored
- Uncommitted changes

### 6. Notebook Validation

Validate Colab notebooks:

```bash
# Validate notebook JSON format
for notebook in notebooks/*.ipynb; do
    if python3 -c "import json; json.load(open('$notebook'))" 2>/dev/null; then
        echo "✅ $(basename $notebook)"
    else
        echo "❌ $(basename $notebook) - INVALID JSON"
    fi
done
```

**What it validates:**
- JSON format (nbformat, nbformat_minor)
- Cell structure
- Required fields present

### 7. Crawler State

Check crawler state database:

```bash
# Check state database
if [ -f "data/raw/.crawler_state.db" ]; then
    python3 << 'EOF'
import sqlite3
import json
from pathlib import Path

db_path = Path("data/raw/.crawler_state.db")
conn = sqlite3.connect(db_path)

# Check tables
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", [t[0] for t in tables])

# Check stats
if "stats" in [t[0] for t in tables]:
    row = conn.execute("SELECT value FROM stats WHERE key='global'").fetchone()
    if row:
        stats = json.loads(row[0])
        print("Crawled documents:", stats.get("documents_extracted", 0))
        print("Pages crawled:", stats.get("pages_crawled", 0))

conn.close()
EOF
fi
```

**What it checks:**
- SQLite database exists
- Tables properly structured
- Statistics available
- Data integrity

### 8. HuggingFace Status

Check HuggingFace repositories:

```bash
# Check HF CLI
if command -v huggingface-cli &> /dev/null; then
    echo "✅ HF CLI installed"
    huggingface-cli whoami
else
    echo "⚠️ HF CLI not installed"
    echo "Install: pip install huggingface_hub"
fi
```

**What it checks:**
- HuggingFace CLI installed
- User authentication status
- Repository access

---

## Maintenance Schedule

### Daily
- Check git status
- Verify backups
- Monitor crawler if running

### Weekly
- Update dependencies
- Check disk space
- Run health check

### Monthly
- Update documentation
- Review code quality
- Clean up temporary files
- Archive old data

### As Needed
- After code changes
- After training runs
- When issues arise

---

## Status Reporting

Generate comprehensive status report:

```bash
/maintenance status
```

**Report includes:**
- Environment status
- Data statistics
- Git status
- Code quality metrics
- Recommendations

---

## Common Maintenance Tasks

### Update Dependencies

```bash
# Update UV dependencies
uv sync --upgrade

# Update Playwright
playwright install chromium --force
```

### Clean Up

```bash
# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Clean crawler state (reset)
rm data/raw/.crawler_state.db
```

### Fix Issues

```bash
# Fix git permissions
git config core.filemode false

# Fix line endings
git config core.autocrlf input

# Reinstall dependencies
uv sync --reinstall
```

---

## Monitoring

### Crawler Monitoring

```bash
# Check crawler progress
uv run python crawler/playwright_crawler.py --stats

# Monitor log files
tail -f crawler.log
```

### Training Monitoring

```bash
# Check training outputs
ls -lh outputs_*/

# Check GGUF export
ls -lh gemma4_e2b_tvpl_gguf/
```

---

## Health Indicators

| Indicator | Good | Warning | Critical |
|-----------|------|--------|----------|
| Dependencies | All satisfied | Some missing | Many missing |
| Data | >100K docs | <10K docs | No data |
| Git | Clean | Uncommitted | Diverged |
| Notebooks | Valid JSON | One invalid | Many invalid |
| Remotes | Both configured | One missing | Both missing |
| Disk Space | >50GB free | 10-50GB | <10GB |

---

## Troubleshooting

### Issues Found During Health Check

**Dependencies missing:**
```bash
uv sync
```

**Notebook format errors:**
```bash
# Notebook will auto-fix on next run
# Or recreate with proper JSON format
```

**Git remote issues:**
```bash
./scripts/git_sync.sh setup
```

**Data corruption:**
```bash
# Validate and reprocess
uv run python scripts/process_documents.py
```

---

## Automation

### Auto-Maintenance Script

Create `scripts/maintenance.sh` for automated checks:

```bash
#!/bin/bash
# Run health check and report issues
echo "Running maintenance check..."

# Add checks from above
```

---

## Quick Commands

```bash
# Full health check
/maintenance check

# Quick status
/maintenance status

# Fix common issues
/maintenance fix

# Clean up
/maintenance clean
```
