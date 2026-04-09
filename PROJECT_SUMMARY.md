# TVPL - Complete Project Summary

## 🎯 Overview

TVPL (Thư Viện Pháp Luạt) is a complete Vietnamese legal RAG system with:
- **Parallel crawler** with state persistence & resume
- **Data processing** pipeline (HTML → Markdown → Passages)
- **Two-stage training** (Pretrain + SFT for RAG)
- **Local RAG** pipeline (ChromaDB + GGUF model)
- **Automation scripts** (Git sync, dataset publishing)

## 📁 Project Structure

```
tvpl/
├── .claude/commands/      # Claude Code shortcuts
│   ├── crawl.md           # Crawler commands
│   ├── process.md         # Data processing
│   ├── train.md           # Training guide
│   ├── rag.md             # RAG pipeline
│   ├── publish.md         # HF publishing
│   ├── git.md             # Git automation
│   └── test.md            # Testing guide
│
├── crawler/
│   ├── parallel_crawler.py # ⭐ Single-file parallel crawler
│   ├── spiders/           # Legacy Scrapy spider
│   ├── items.py           # Data models (30+ fields)
│   ├── pipelines.py       # Processing pipelines
│   └── settings.py        # Scrapy config
│
├── scripts/
│   ├── publish_dataset.sh  # ⭐ Publish to HF
│   ├── git_sync.sh         # ⭐ Git automation
│   ├── test_crawler.py     # Test suite
│   ├── validate_data.py    # Data validation
│   ├── process_documents.py
│   ├── build_pretrain.py
│   ├── build_sft.py
│   ├── prepare_hf_dataset.py
│   └── upload_with_xet.py
│
├── rag/
│   ├── pipeline.py         # ⭐ Complete RAG pipeline
│   └── README.md
│
├── notebooks/
│   ├── 01_pretrain.ipynb   # Colab: Stage 1
│   └── 02_sft_rag.ipynb    # Colab: Stage 2
│
├── data/
│   ├── raw/                # Crawler output
│   ├── processed/          # Parquet files
│   ├── pretrain/           # Training corpus
│   ├── sft/                # Instruction pairs
│   └── .crawler_state.db   # ⭐ SQLite state DB
│
└── [config files: requirements.txt, pyproject.toml, etc.]
```

## 🚀 Quick Start

### 1. Setup
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Crawl (with state persistence)
```bash
# Test first
uv run tvpl-test

# Single worker
uv run python crawler/parallel_crawler.py

# 4 parallel workers
uv run python crawler/parallel_crawler.py --workers 4

# Resume after interruption
uv run python crawler/parallel_crawler.py --resume

# Check statistics
uv run python crawler/parallel_crawler.py --stats
```

### 3. Process Data
```bash
uv run tvpl-process
uv run tvpl-build-pretrain
uv run tvpl-build-sft
uv run tvpl-validate
```

### 4. Train on Colab
1. Upload `notebooks/01_pretrain.ipynb` to Colab
2. Upload `notebooks/02_sft_rag.ipynb` to Colab
3. Run both notebooks (T4 GPU, free tier)

### 5. Local RAG
```bash
# Install RAG deps
uv pip install chromadb langchain langchain-community

# Build index
uv run python rag/pipeline.py --rebuild

# Interactive Q&A
uv run python rag/pipeline.py --interactive --model models/gemma4-vi-legal.gguf
```

## 🔥 Key Features

### Parallel Crawler (`crawler/parallel_crawler.py`)

**Single file, powerful features:**
- ✅ **State persistence** (SQLite DB)
- ✅ **Stop & resume** (Ctrl+C safe)
- ✅ **Parallel workers** (multi-process)
- ✅ **Deduplication** (URL + doc_id)
- ✅ **Rate limiting** (polite crawling)
- ✅ **Progress tracking** (real-time stats)
- ✅ **Graceful shutdown** (saves state)

**Usage:**
```bash
# 4 parallel workers, each processing different pages
python crawler/parallel_crawler.py --workers 4 --delay 2.5

# Resume from exact state
python crawler/parallel_crawler.py --resume

# Export to JSONL
python crawler/parallel_crawler.py --export output.jsonl
```

**State Database Schema:**
```sql
-- Seen URLs (deduplication)
seen_urls (url, first_seen, worker_id, status)

-- Extracted documents
documents (doc_id, url, data JSON, crawled_at, worker_id)

-- Worker tracking
workers (worker_id, status, current_page, last_heartbeat, documents_fetched)

-- Statistics
stats (key, value JSON)
```

### Automation Scripts

**Git Automation (`scripts/git_sync.sh`):**
```bash
# Quick sync (commit + push)
./scripts/git_sync.sh sync

# Semantic commit
./scripts/git_sync.sh commit crawler "Add parallel crawling"

# Push only
./scripts/git_sync.sh push
```

**Dataset Publishing (`scripts/publish_dataset.sh`):**
```bash
# Full pipeline
./scripts/publish_dataset.sh

# Custom repo
./scripts/publish_dataset.sh -r username/tvpl-vi-legal
```

### Claude Code Commands

Located in `.claude/commands/`:
- `crawl.md` - Crawler commands
- `process.md` - Data processing
- `train.md` - Training guide
- `rag.md` - RAG pipeline
- `publish.md` - HF publishing
- `git.md` - Git automation
- `test.md` - Testing guide

Accessible via: `/crawl`, `/process`, `/train`, `/rag`, `/publish`, `/git`, `/test`

## 📊 Data Flow

```
thuvienphapluat.vn
    ↓
[Parallel Crawler] → SQLite state DB
    ↓
data/raw/documents.jsonl
    ↓
[Process] → Parquet + Passages
    ↓
data/processed/
    ↓
[Build Datasets] → Pretrain + SFT
    ↓
data/pretrain/ + data/sft/
    ↓
[Publish] → HuggingFace Hub
    ↓
[Train on Colab] → LoRA adapters + GGUF
    ↓
[RAG Pipeline] → Local Q&A system
```

## 🛠️ Troubleshooting

**Crawler stuck:**
```bash
# Check state
python crawler/parallel_crawler.py --stats

# Resume
python crawler/parallel_crawler.py --resume
```

**Reset crawler state:**
```bash
rm data/raw/.crawler_state.db
```

**Colab OOM:**
- Reduce `max_seq_length` to 2048
- Reduce batch size to 1
- Increase gradient accumulation to 8

**RAG slow:**
- Use smaller embedding: `intfloat/multilingual-e5-small`
- Reduce retriever `k` from 3 to 2

## 📈 Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Crawler | Pages/hour (single) | ~40 |
| Crawler | Pages/hour (4 workers) | ~150 |
| Processing | 10K docs | ~5 min |
| Embedding | 10K docs | ~10 min |
| RAG Query | Response time | ~2-5s |
| Memory | Total usage | ~8GB |

## 📝 Next Steps

1. **Start crawling** — 3-7 days at polite rate
2. **Process in parallel** — Don't wait for crawl
3. **Train on Colab** — Free T4 sufficient
4. **Deploy locally** — RAG on Mac with GGUF

## 📚 Resources

- [Unsloth Gemma 4](https://unsloth.ai/docs/models/gemma-4)
- [Vietnamese NLP](https://github.com/undertheseanlp)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [ChromaDB](https://www.trychroma.com/)
