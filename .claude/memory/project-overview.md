---
name: project-overview
description: Core project information for Gemma 4 Vietnamese Legal Documents
type: project
---

# Gemma 4 Vietnamese Legal Documents - Project Overview

## Purpose
End-to-end pipeline to:
1. Crawl Vietnamese legal documents from thuvienphapluat.vn
2. Prepare datasets for fine-tuning
3. Fine-tune Gemma 4 E2B (2.3B parameters) for Vietnamese legal RAG
4. Deploy local RAG system

## Key Components

### Crawler (`crawler/`)
- **parallel_crawler.py**: Multi-process crawler with SQLite state persistence
- **playwright_crawler.py**: Cloudflare bypass using headless Chromium
- **items.py**: Document schema with 30+ fields (HTML, Markdown, relationships)

### Scripts (`scripts/`)
- **config.py**: Python configuration loader (loads .env)
- **config.sh**: Bash configuration loader
- **git_sync.sh**: Dual GitHub + HuggingFace sync automation
- **download_hf_dataset.py**: Download base dataset from HF
- **merge_datasets.py**: Combine HF + crawled data
- **process_documents.py**: Raw JSONL → Parquet, extract passages
- **build_pretrain.py**: Build pretraining corpus with document headers
- **upload_with_xet.py**: Fast upload to HuggingFace
- **colab.sh**: Unified Colab helper script

### Notebooks (`notebooks/`)
- **Gemma4_Vietnamese_Legal_Train.ipynb**: Full pipeline notebook with flag controls
  - All features controlled by flags in first cell
  - Clone repo, crawl, train, evaluate, upload - all in one
  - Valid JSON format (nbformat: 4, nbformat_minor: 0)
- **01_pretrain.ipynb**: Manual Stage 1 training
- **02_sft_rag.ipynb**: Manual Stage 2 training
- **README.md**: Notebook documentation

### RAG Pipeline (`rag/`)
- **pipeline.py**: Complete RAG with ChromaDB + GGUF model support

## Training Strategy

### Two-Stage Training
1. **Stage 1 - Continued Pretraining**: Embed Vietnamese legal knowledge
2. **Stage 2 - SFT for RAG**: Fine-tune on instruction pairs with context

### Model
- **Base**: `unsloth/gemma-4-E2B-it` (2.3B parameters, Apache 2.0)
- **Training**: LoRA adapters (1% parameters, Q4 quantization)
- **Hardware**: Google Colab free T4 (16GB VRAM)
- **Export**: GGUF format (q4_k_m, q5_k_m)

### Datasets
- **Primary**: [th1nh0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nh0/vietnamese-legal-documents) (~150K docs)
- **Additional**: Crawled from thuvienphapluat.vn (optional)

## Repository Structure

```
gemma-4-finetune-vietnamese-legal-documents/
├── crawler/              # Web crawler
├── scripts/              # Data processing & automation
├── notebooks/            # Training notebooks (.ipynb gitignored)
├── rag/                  # RAG pipeline
├── data/                 # Data storage (gitignored)
├── docs/                 # Documentation
├── .env.example          # Configuration template
└── .env                  # User settings (gitignored)
```

## Key Files for Claude

### When User Asks About Training
- **Notebook**: `notebooks/Gemma4_Vietnamese_Legal_Train.ipynb`
- **Helper**: `./scripts/colab.sh`
- **Guide**: `notebooks/README.md`

### When User Asks About Crawling
- **Parallel**: `crawler/parallel_crawler.py`
- **Playwright**: `crawler/playwright_crawler.py`
- **Schema**: `crawler/items.py`

### When User Asks About Configuration
- **Python**: `from scripts.config import git, crawler, training, rag`
- **Bash**: `source scripts/config.sh`
- **Template**: `.env.example`

### When User Asks About Git
- **Sync**: `./scripts/git_sync.sh [setup|sync|commit|push]`
- **Config**: Loaded from `.env` file

## Important Notes

### Notebooks are GitIgnored
- `.ipynb` files are in `.gitignore`
- Users upload manually to Colab
- This prevents merge conflicts and large binary files in git

### Flag-Controlled Training
The main notebook uses flags to control everything:
```python
CRAWL_ENABLED = False
DOWNLOAD_HF_DATASET = True
RUN_TRAINING = True
EXPORT_TO_GGUF = True
GENERATE_SCORES = True
# ... etc
```

### Dual Repository Strategy
- **GitHub**: Code repository (all source code, scripts, docs)
- **HuggingFace**: Dataset + Model repositories (large files)

## How to Use This Context

When user asks about:
- **"Train model"** → Point to `Gemma4_Vietnamese_Legal_Train.ipynb` with flags
- **"Crawl data"** → Point to `crawler/parallel_crawler.py` or `playwright_crawler.py`
- **"Configure"** → Point to `.env.example` and config loaders
- **"Deploy RAG"** → Point to `rag/pipeline.py`
- **"Push to HF"** → Point to `scripts/git_sync.sh push hf`

**Why**: This context captures the project architecture and key workflows for efficient assistance.
