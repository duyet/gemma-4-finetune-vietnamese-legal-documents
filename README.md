# Gemma 4 Vietnamese Legal Documents - RAG Pipeline

End-to-end pipeline to crawl Vietnamese legal documents, prepare datasets, and fine-tune Gemma 4 E2B for Retrieval-Augmented Generation (RAG).

## 🚀 Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone git@github.com:duyet/gemma-4-finetune-vietnamese-legal-documents.git
cd gemma-4-finetune-vietnamese-legal-documents

# Install with UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or with pip
pip install -r requirements.txt
```

### 2. Configure Your Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
nano .env  # or your preferred editor
```

**Key settings to customize:**
- `GITHUB_USERNAME` - Your GitHub username
- `HF_USERNAME` - Your HuggingFace username
- `HF_DATASET_NAME` - Name for your HF dataset
- `HF_MODEL_NAME` - Name for your HF model
- `CRAWLER_MAX_PAGES` - Maximum pages to crawl
- `BASE_MODEL` - Base model for fine-tuning

### 3. Prepare Dataset

**Option A: Use existing HF dataset (fastest)**

```bash
# Download base dataset
uv run python scripts/download_hf_dataset.py

# Process into training formats
uv run python scripts/process_documents.py
uv run python scripts/build_pretrain.py
```

**Option B: Crawl additional data**

```bash
# Run Playwright crawler (bypasses Cloudflare)
uv run python crawler/playwright_crawler.py --max-pages 100

# Merge with existing dataset
uv run python scripts/merge_datasets.py
```

### 4. Fine-tune on Colab

Upload and run `notebooks/Auto_Train.ipynb` to Google Colab.

**The notebook will:**
- Auto-clone latest code from GitHub
- Install all dependencies
- Download dataset directly
- Fine-tune Gemma 4 E2B
- Export to GGUF format

### 5. Deploy RAG Pipeline

```bash
# Build vector store
uv run python rag/pipeline.py --rebuild

# Run interactive Q&A
uv run python rag/pipeline.py --interactive \
    --model path/to/model.gguf
```

## 🎯 Project Structure

```
gemma-4-finetune-vietnamese-legal-documents/
├── .env.example              # Configuration template
├── .env                      # Your custom settings (not in git)
│
├── crawler/                  # Web crawler
│   ├── spiders/              # Scrapy spiders
│   ├── items.py              # Document schema (30+ fields)
│   ├── parallel_crawler.py   # Multi-process crawler
│   └── playwright_crawler.py # Cloudflare bypass
│
├── scripts/                  # Data processing
│   ├── config.py             # Python config loader
│   ├── config.sh             # Bash config loader
│   ├── git_sync.sh           # Dual GitHub + HF sync
│   ├── download_hf_dataset.py
│   ├── merge_datasets.py
│   ├── process_documents.py
│   ├── build_pretrain.py
│   ├── upload_with_xet.py
│   └── setup_hf_repos.sh
│
├── notebooks/                # Training notebooks
│   └── Auto_Train.ipynb      # Auto-training (git clone)
│
├── rag/                      # RAG pipeline
│   └── pipeline.py           # Complete RAG implementation
│
├── data/                     # Data storage
│   ├── raw/                  # Raw crawled data
│   ├── processed/            # Cleaned documents
│   ├── pretrain/             # Pretraining corpus
│   └── sft/                  # SFT instruction pairs
│
└── docs/                     # Documentation
    ├── HF_STRUCTURE.md       # HF repository structure
    └── XET_UPLOAD.md         # Fast upload guide
```

## ⚙️ Customization

### Environment Variables

Create a `.env` file from `.env.example` and customize:

```bash
# ============================================
# Git Repository Configuration
# ============================================
GITHUB_REPO="git@github.com:YOUR_USERNAME/gemma-4-finetune-vietnamese-legal-documents.git"
GITHUB_USERNAME="YOUR_USERNAME"

# HuggingFace Configuration
HF_USERNAME="YOUR_USERNAME"
HF_REPO_NAME="gemma-4-finetune-vietnamese-legal-documents"
HF_DATASET_NAME="vietnamese-legal-documents"
HF_MODEL_NAME="gemma-4-vietnamese-legal-rag"

# ============================================
# Crawler Configuration
# ============================================
CRAWLER_MAX_PAGES=100          # Set 0 to skip crawling
CRAWLER_DELAY=2.5              # Seconds between requests
CRAWLER_CONCURRENT_REQUESTS=4  # Parallel workers

# ============================================
# Training Configuration
# ============================================
BASE_MODEL="unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH=4096
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-4
NUM_EPOCHS=1

# LoRA Settings
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05

# ============================================
# RAG Configuration
# ============================================
EMBEDDING_MODEL="bkai-foundation-models/vietnamese-bi-encoder"
VECTOR_STORE_PATH="./data/chroma_db"
TOP_K_RETRIEVAL=3
```

### Python Configuration

In Python scripts, configuration is loaded automatically:

```python
from scripts.config import git, crawler, training, rag

# Use configuration
print(f"HF Dataset: {git.hf_username}/{git.hf_dataset_name}")
print(f"Max pages: {crawler.max_pages}")
print(f"Learning rate: {training.learning_rate}")
```

### Bash Configuration

In bash scripts, source the config:

```bash
source scripts/config.sh

# Configuration variables are now available
echo "HF Username: $HF_USERNAME"
echo "Max pages: $CRAWLER_MAX_PAGES"
```

## 🔄 Git Sync - Dual GitHub + HuggingFace

The project supports syncing to both GitHub and HuggingFace:

```bash
# Setup remotes
./scripts/git_sync.sh setup

# Commit and push to both
./scripts/git_sync.sh sync

# Push to GitHub only
./scripts/git_sync.sh push github

# Push to HuggingFace only
./scripts/git_sync.sh push hf
```

Configuration via `.env`:
```bash
GITHUB_REPO="git@github.com:YOUR_USERNAME/repo.git"
HF_USERNAME="YOUR_USERNAME"
HF_REPO_NAME="your-repo-name"
```

## 📊 Dataset Details

### Source
- **Primary**: [th1nh0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nh0/vietnamese-legal-documents) (~150K docs from vbpl.vn)
- **Additional**: Crawled from thuvienphapluat.vn (optional, via Playwright)

### Document Types
- Luật (Laws)
- Nghị định (Decrees)
- Thông tư (Circulars)
- Quyết định (Decisions)
- Nghị quyết (Resolutions)
- And more...

### Metadata Fields (30+)
- Identification: `doc_id`, `title`, `url`, `doc_number`
- Classification: `doc_type`, `category`, `sector`, `field`
- Authority: `issuing_authority`, `signatory`, `signatory_title`
- Dates: `issue_date`, `effective_date`, `expiry_date`
- Status: `status`, `effect_status`
- Content: `content_html`, `content_text`, `content_markdown`
- Relationships: `amends_docs`, `repeals_docs`, `cites_docs`

## 🎓 Training

### Stage 1: Continued Pretraining
Embed Vietnamese legal knowledge into Gemma 4 E2B.

```python
# In Colab notebook
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-4-E2B-it",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

### Stage 2: SFT for RAG
Fine-tune on instruction pairs with retrieved context.

### Export to GGUF
```python
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

## 🤖 HuggingFace Repositories

### Dataset Repository
```
YOUR_USERNAME/vietnamese-legal-documents
├── documents/    # Full documents with metadata
├── passages/     # Chunked passages for RAG
├── pretrain/     # Pretraining corpus
└── sft/          # Instruction pairs
```

### Model Repository
```
YOUR_USERNAME/gemma-4-vietnamese-legal-rag
├── adapter_model.safetensors  # LoRA weights
├── tokenizer files
├── config.json
└── gguf/
    ├── q4_k_m.gguf  # Fast inference
    └── q5_k_m.gguf  # Higher quality
```

**Create repositories:**
```bash
./scripts/setup_hf_repos.sh
```

## 📜 License

Vietnamese legal documents are **public domain** under:
- Law on Access to Information (No. 104/2016/QH13)
- Law on Promulgation of Legal Documents (No. 64/2025/QH15)

The compiled dataset and code are released under **CC BY 4.0**.

## 🙏 Acknowledgments

- Base dataset: [th1nh0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nh0/vietnamese-legal-documents)
- Source: [vbpl.vn](https://vbpl.vn) - Official Government Legal Document Portal
- Additional source: [thuvienphapluat.vn](https://thuvienphapluat.vn)
- Base model: [Google Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma/)
- Training framework: [Unsloth](https://unsloth.ai/)

## 📧 Contact

For questions or collaboration:
- GitHub: [duyet](https://github.com/duyet)
- HuggingFace: [duyet](https://huggingface.co/duyet)

---

**Last updated**: 2026-04-09
**Version**: 1.0.0
