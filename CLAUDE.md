# Claude Code Context - Gemma 4 Vietnamese Legal Documents

This file provides context for Claude Code to assist with development of this project.

## Project Overview

**Purpose**: End-to-end pipeline for Vietnamese legal RAG system using Gemma 4 E2B.

**Tech Stack**:
- **Crawler**: Scrapy + Playwright (Cloudflare bypass)
- **Processing**: Pandas, Markdownify, BeautifulSoup
- **Training**: Unsloth, Transformers, TRL
- **RAG**: ChromaDB, LangChain, llama.cpp (GGUF)
- **Automation**: Custom Bash scripts, Python configuration system

**Key Features**:
1. Stateful parallel crawler with resume capability
2. Multi-format content extraction (HTML → Markdown → Passages)
3. Two-stage training (Pretrain → SFT for RAG)
4. Local RAG with GGUF quantized models
5. Dual GitHub + HuggingFace sync

## Repository Structure

```
├── crawler/              # Web crawler components
│   ├── items.py          # Document schema (30+ fields)
│   ├── parallel_crawler.py    # Multi-process crawler with SQLite state
│   ├── playwright_crawler.py  # Cloudflare bypass
│   └── settings.py       # Scrapy configuration
│
├── scripts/              # Data processing & automation
│   ├── config.py         # Python config loader (loads .env)
│   ├── config.sh         # Bash config loader
│   ├── git_sync.sh       # Dual GitHub + HF sync
│   ├── download_hf_dataset.py
│   ├── merge_datasets.py
│   ├── process_documents.py
│   ├── build_pretrain.py
│   ├── build_sft.py
│   ├── prepare_hf_dataset.py
│   ├── upload_with_xet.py
│   └── setup_hf_repos.sh
│
├── notebooks/            # Training notebooks
│   └── Gemma4_Vietnamese_Legal_Train.ipynb  # Full pipeline with flags
│
├── rag/                  # RAG pipeline
│   └── pipeline.py       # Complete RAG with ChromaDB + GGUF
│
├── data/                 # Data storage (gitignored)
│   ├── raw/              # Crawler output
│   ├── processed/        # Parquet files
│   ├── pretrain/         # Training corpus
│   └── sft/              # Instruction pairs
│
└── docs/                 # Documentation
    ├── HF_STRUCTURE.md   # Repository structure
    ├── HF_SETUP.md       # Setup guide
    └── XET_UPLOAD.md     # Fast upload with XET
```

## Configuration System

**Environment Variables**: All customization via `.env` file (gitignored)

**Python Usage**:
```python
from scripts.config import git, crawler, training, rag

# Access configuration
print(f"HF Dataset: {git.hf_username}/{git.hf_dataset_name}")
print(f"Max pages: {crawler.max_pages}")
```

**Bash Usage**:
```bash
source scripts/config.sh
echo "HF Username: $HF_USERNAME"
```

**Key Configuration**:
- `GITHUB_USERNAME`, `HF_USERNAME` - Repository usernames
- `HF_DATASET_NAME`, `HF_MODEL_NAME` - Repository names
- `CRAWLER_MAX_PAGES`, `CRAWLER_DELAY` - Crawler settings
- `BASE_MODEL`, `BATCH_SIZE`, `LEARNING_RATE` - Training hyperparameters

## Training on Colab

### Full Pipeline Notebook (`Gemma4_Vietnamese_Legal_Train.ipynb`)

**Flag-controlled training notebook** - All features controlled by flags in first cell.

**Available Flags:**
```python
# Data Pipeline
CRAWL_ENABLED = False  # Enable/disable crawling
CRAWL_PAGES = 100  # Number of pages to crawl
DOWNLOAD_HF_DATASET = True  # Download base dataset
MERGE_DATASETS = True  # Merge crawled + HF data
UPLOAD_DATASET_TO_HF = False  # Upload merged dataset to HF

# Training
RUN_TRAINING = True  # Run fine-tuning
TRAINING_STAGE = "pretrain"  # "pretrain", "sft", "both"
MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

# Export & Upload
EXPORT_TO_GGUF = True  # Export to GGUF format
QUANTIZATION = "q4_k_m"  # Quantization method
UPLOAD_MODEL_TO_HF = False  # Upload model to HF

# Evaluation
GENERATE_SCORES = True  # Generate evaluation scores
RUN_BENCHMARKS = False  # Run detailed benchmarks

# Git
PUSH_RESULTS_TO_GITHUB = False  # Push training results
```

**Quick Presets:**
```python
# Fast test (1K docs, 30 min)
CRAWL_ENABLED = False; GENERATE_SCORES = True; RUN_BENCHMARKS = False

# Full pipeline
CRAWL_ENABLED = True; UPLOAD_DATASET_TO_HF = True; UPLOAD_MODEL_TO_HF = True

# Training only (skip crawling)
CRAWL_ENABLED = False; DOWNLOAD_HF_DATASET = True; RUN_TRAINING = True
```

**Workflow:**
1. Run `./scripts/colab.sh` to open Colab
2. Upload `Gemma4_Vietnamese_Legal_Train.ipynb`
3. Edit flags in first cell
4. Run all cells
5. Model downloads automatically when complete

## Development Workflow

### Git Commits

**Format**: Semantic commits with scope
```
scope: message

Options:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactoring
- chore: Maintenance
- test: Testing

Scopes:
- crawler: Crawler changes
- data: Data processing
- rag: RAG pipeline
- docs: Documentation
- train: Training
- config: Configuration
```

**Commit Authors**:
- `duyet` - Human commits
- `duyetbot` - Automated commits
- `claude` - Claude Code commits

**Git Sync Commands**:
```bash
# Setup dual remotes
./scripts/git_sync.sh setup

# Commit and push to both
./scripts/git_sync.sh sync

# Semantic commit
./scripts/git_sync.sh commit crawler "Add Cloudflare bypass"

# Push to specific target
./scripts/git_sync.sh push github
./scripts/git_sync.sh push hf
```

### Testing

**Crawler Test**:
```bash
# Quick test (1 page)
uv run python crawler/playwright_crawler.py --max-pages 1

# Check output
cat data/raw/playwright_documents.jsonl | jq
```

**Data Validation**:
```bash
# Validate processed data
uv run python scripts/validate_data.py

# Check statistics
uv run python crawler/playwright_crawler.py --stats
```

### Training Workflow

1. **Prepare Data**:
   ```bash
   uv run python scripts/download_hf_dataset.py
   uv run python scripts/process_documents.py
   uv run python scripts/build_pretrain.py
   ```

2. **Train on Colab**:
   - Upload `notebooks/Auto_Train.ipynb`
   - Configuration at top of notebook
   - No notebook updates needed - pulls from GitHub

3. **Export to GGUF**:
   - Done automatically in notebook
   - Outputs: `q4_k_m.gguf`, `q5_k_m.gguf`

### Deployment

**Local RAG**:
```bash
# Build vector store
uv run python rag/pipeline.py --rebuild

# Interactive Q&A
uv run python rag/pipeline.py --interactive \
    --model path/to/model.gguf
```

**Publish to HuggingFace**:
```bash
# Setup repositories
./scripts/setup_hf_repos.sh

# Upload dataset
uv run python scripts/upload_with_xet.py -r $HF_USERNAME/$HF_DATASET_NAME
```

## Key Implementation Details

### Crawler Architecture

**Parallel Crawler** (`crawler/parallel_crawler.py`):
- SQLite state DB for persistence
- Multi-process workers
- Deduplication by URL and doc_id
- Graceful shutdown handling
- Resume capability

**Playwright Crawler** (`crawler/playwright_crawler.py`):
- Headless Chromium for Cloudflare bypass
- Scrapy integration
- State persistence

### Data Processing

**Pipeline**:
1. Raw JSONL → Parquet (process_documents.py)
2. Extract passages (~512 tokens each)
3. Build pretrain corpus with document headers
4. Generate SFT Q&A pairs (rule-based or LLM-generated)

**Document Schema** (30+ fields in `crawler/items.py`):
```python
@dataclass
class LegalDocument:
    # Identification
    url: str
    doc_id: str
    title: str
    doc_number: str

    # Classification
    doc_type: str
    category: str
    sector: str
    field: str

    # Authority
    issuing_authority: str
    signatory: str
    signatory_title: str

    # Dates
    issue_date: str
    effective_date: str
    expiry_date: str | None

    # Status
    status: str
    effect_status: str

    # Content (multiple formats)
    content_html: str
    content_text: str
    content_markdown: str

    # Relationships
    amends_docs: list[str]
    repeals_docs: list[str]
    cites_docs: list[str]

    # And more...
```

### RAG Pipeline

**Components**:
1. **Embeddings**: `bkai-foundation-models/vietnamese-bi-encoder`
2. **Vector Store**: ChromaDB (persistent)
3. **LLM**: GGUF quantized Gemma 4 via llama.cpp
4. **Orchestration**: LangChain

**Usage**:
```python
from rag.pipeline import LegalRAGPipeline

# Initialize
rag = LegalRAGPipeline(
    embedding_model="bkai-foundation-models/vietnamese-bi-encoder",
    llm_path="path/to/model.gguf",
    persist_directory="./data/chroma_db"
)

# Query
answer = rag.query("Điều kiện chuyển nhượng đất nông nghiệp?")
```

## Troubleshooting

### Crawler Issues

**Stuck at "Just a moment"**:
- Cloudflare protection - use Playwright crawler
- Install: `playwright install chromium`

**Resume capability**:
- State saved to SQLite DB
- Run with `--resume` flag
- Check stats with `--stats`

### Training Issues

**Colab OOM**:
- Reduce `MAX_SEQ_LENGTH` to 2048
- Reduce `BATCH_SIZE` to 1
- Increase `GRADIENT_ACCUMULATION_STEPS` to 8

### RAG Issues

**Slow retrieval**:
- Use smaller embedding: `intfloat/multilingual-e5-small`
- Reduce `TOP_K_RETRIEVAL` to 2

**Poor answers**:
- Increase context length
- Use higher quality quantization (q5_k_m)
- Fine-tune with SFT data

## Performance Benchmarks

| Component | Metric | Value |
|-----------|--------|-------|
| Crawler | Pages/hour (single) | ~40 |
| Crawler | Pages/hour (4 workers) | ~150 |
| Processing | 10K docs | ~5 min |
| Embedding | 10K docs | ~10 min |
| RAG Query | Response time | ~2-5s |
| Memory | Total usage | ~8GB |

## Resources

- [Unsloth Gemma 4](https://unsloth.ai/docs/models/gemma-4)
- [Vietnamese NLP - underthesea](https://github.com/undertheseanlp)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [ChromaDB](https://www.trychroma.com/)
- [llama.cpp GGUF](https://github.com/ggerganov/llama.cpp)

## Dependencies

**Core**:
- `scrapy` - Web crawling framework
- `playwright` - Browser automation (Cloudflare bypass)
- `pandas` - Data processing
- `markdownify` - HTML to Markdown
- `beautifulsoup4` - HTML parsing

**Training**:
- `unsloth` - Fast fine-tuning
- `transformers` - HuggingFace transformers
- `trl` - Transformer Reinforcement Learning
- `datasets` - HuggingFace datasets

**RAG**:
- `chromadb` - Vector database
- `langchain` - LLM orchestration
- `llama-cpp-python` - GGUF inference

## License

- **Code**: CC BY 4.0
- **Legal Documents**: Public domain (Vietnamese law)
- **Base Model**: Apache 2.0 (Gemma 4)

---

**Last updated**: 2026-04-09
**Version**: 1.0.0
