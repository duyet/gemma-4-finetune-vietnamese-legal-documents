---
description: Run TVPL crawler (polite mode with resume capability)
---

Run the TVPL crawler with state persistence and resume capability.

**Usage:**
```bash
uv run python crawler/parallel_crawler.py --workers 1 --delay 2.5
```

**Options:**
- `--workers/-w`: Number of parallel workers (default: 1)
- `--start-page/-s`: Starting page number
- `--max-pages/-m`: Maximum pages to crawl
- `--delay/-d`: Delay between requests in seconds (default: 2.5)
- `--resume`: Resume from last state
- `--stats`: Show statistics only
- `--export/-e`: Export to JSONL file

**Examples:**
```bash
# Single worker
uv run python crawler/parallel_crawler.py

# 4 parallel workers
uv run python crawler/parallel_crawler.py --workers 4

# Resume from last state
uv run python crawler/parallel_crawler.py --resume

# Show statistics
uv run python crawler/parallel_crawler.py --stats

# Export to file
uv run python crawler/parallel_crawler.py --export data/raw/export.jsonl
```

**State Persistence:**
- SQLite database: `data/raw/.crawler_state.db`
- Auto-export: `data/raw/documents_TIMESTAMP.jsonl`
- Tracks: seen URLs, extracted documents, worker status
