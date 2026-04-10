# Gemma 4 Vietnamese Legal Documents - Fine-tuning Pipeline

Fine-tune Gemma 4 E2B on Vietnamese legal documents for Retrieval-Augmented Generation (RAG).

## 🚀 Quick Start

### Option A: Train on Google Colab (Fastest)

**One notebook, everything automated!** 🚀

```bash
# Clone the repository
git clone git@github.com:duyet/gemma-4-finetune-vietnamese-legal-documents.git
cd gemma-4-finetune-vietnamese-legal-documents

# Open Colab helper
./scripts/colab.sh
```

**Then in Colab:**
1. Upload `notebooks/Gemma4_Vietnamese_Legal_Finetune.ipynb`
2. Set flags in first cell (control what runs)
3. Run all cells

**Control flags:**
- `CRAWL_ENABLED` - Enable/disable crawling (optional)
- `DOWNLOAD_HF_DATASET` - Download base dataset
- `RUN_TRAINING` - Run fine-tuning
- `TRAINING_STAGE` - Training stage (pretrain, sft, both)
- `MAX_SEQ_LENGTH` - Max sequence length
- `BATCH_SIZE` - Training batch size
- `EXPORT_TO_GGUF` - Export quantized model
- `GENERATE_SCORES` - Generate evaluation scores
- `DOWNLOAD_HF_DATASET` - Download base dataset
- `RUN_TRAINING` - Run fine-tuning
- `EXPORT_TO_GGUF` - Export quantized model
- `GENERATE_SCORES` - Generate evaluation scores
- And more...

**The notebook handles:**
- ✅ Clone latest code from GitHub
- ✅ Crawl fresh data (optional)
- ✅ Download/merge datasets
- ✅ Upload to HuggingFace (optional)
- ✅ Fine-tune Gemma 4 E2B
- ✅ Export to GGUF format
- ✅ Generate scores & metrics
- ✅ Download trained model

---

### Option B: Local Setup

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

# Install Playwright for Cloudflare bypass
playwright install chromium
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
- `CRAWLER_MAX_PAGES` - Maximum pages to crawl (set 0 to skip)
- `BASE_MODEL` - Base model for fine-tuning

> **📖 See [docs/HF_SETUP.md](docs/HF_SETUP.md) for complete HuggingFace setup guide**

### 3. Prepare Dataset

**Option A: Use existing HF dataset (fastest)**

```bash
# Download base dataset (~150K documents)
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

**Crawler Features:**
- ✅ State persistence (SQLite DB) - stop & resume anytime
- ✅ Parallel workers (multi-process)
- ✅ Deduplication (URL + doc_id)
- ✅ Rate limiting (polite: 2.5s delay)
- ✅ Graceful shutdown (Ctrl+C safe)

```bash
# Resume after interruption
uv run python crawler/playwright_crawler.py --resume

# Check statistics
uv run python crawler/playwright_crawler.py --stats

# 4 parallel workers
uv run python crawler/playwright_crawler.py --workers 4
```

### 4. Fine-tune (Colab, Local, or HuggingFace Jobs)

**Option A: On Colab (Free T4 GPU, 4-8 hrs)**

- Upload `notebooks/Gemma4_Vietnamese_Legal_Finetune.ipynb`
- Set configuration flags in first cell
- Run all cells
- Notebook automatically clones latest code from GitHub

**Option B: Locally (Mac M1/M2/M3 or NVIDIA GPU)**

```bash
# Install training dependencies
uv sync --all-extras

# Mac M1/M2/M3 (MPS acceleration)
uv run python scripts/local_train.py \
    --stage pretrain \
    --max-seq-length 2048 \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --epochs 1

# NVIDIA CUDA GPU (with 4-bit quantization)
uv run python scripts/local_train.py \
    --stage pretrain \
    --use-4bit \
    --max-seq-length 4096 \
    --batch-size 2 \
    --epochs 1
```

**Option C: HuggingFace Jobs (Scalable, Production-Ready)** ⭐

```bash
# Submit training job to HuggingFace infrastructure (two methods)

# Method 1: Simplified uv run (recommended for Llama models)
bash scripts/submit_hf_job.sh

# Method 2: Direct environment config (for Gemma 4 or custom configs)
bash scripts/submit_gemma4_job.sh

# Monitor job
hf jobs ps
hf jobs logs <job-id> --follow

# Results auto-upload to your HF repo
# https://huggingface.co/duyet/gemma-4-E2B-vietnamese-legal
```

**HF Jobs Benefits:**
- ✅ Free T4 tier available
- ✅ Scalable to A100/H100 (faster training)
- ✅ Auto-upload to HuggingFace Hub
- ✅ Run multiple jobs in parallel
- ✅ No Colab session timeout
- ✅ Managed dependencies via uv or Docker

**Submission Methods:**
- **`submit_hf_job.sh`**: Uses `hf jobs uv run` with command-line arguments (simpler, follows Unsloth best practices)
- **`submit_gemma4_job.sh`**: Uses environment variables for full control (better for Gemma 4 or custom configs)

**See `hf_jobs/README.md` for detailed configuration**

**Expected performance:**
| Platform | Hardware | VRAM | Cost | Time (324K) | Setup |
|----------|----------|------|------|-------------|-------|
| Colab | T4 (free) | 16GB | Free | ~5 hrs | Notebook |
| HF Jobs | T4 (free) | 16GB | Free | ~5 hrs | `bash scripts/submit_hf_job.sh` |
| HF Jobs | A100 | 40GB | $4/hr | ~2 hrs | `HARDWARE=a100.large bash scripts/submit_hf_job.sh` |
| HF Jobs | H100 | 80GB | $7/hr | ~1 hr | `HARDWARE=h100 bash scripts/submit_hf_job.sh` |
| Local | Mac M1/M2/M3 | Shared | Free | ~16 hrs | `scripts/local_train.py` |
| Local | NVIDIA 8GB | 8GB | Free | ~6 hrs | `--use-4bit` |

**HF Jobs will:**
- Auto-clone latest code from GitHub
- Install all dependencies via uv or Docker
- Download dataset directly from HuggingFace
- Fine-tune model with LoRA
- Export to GGUF format
- Auto-upload to Hub

**Note:** For Colab, push code changes to GitHub and re-run. For HF Jobs, just resubmit the job.

### 4.5. HuggingFace Jobs - Quick Reference

**Two submission methods:**

| Method | Use Case | Command |
|--------|----------|---------|
| **UV Run** | Llama models, standard configs | `bash scripts/submit_hf_job.sh` |
| **Environment** | Gemma 4, custom configs | `bash scripts/submit_gemma4_job.sh` |

**Method 1 (UV Run)** - Recommended for most use cases:
- ✅ Uses `hf jobs uv run` (faster, no git clone)
- ✅ Works with: `unsloth/Llama-3.2-3B-Instruct`, `Qwen2.5-7B-Instruct`, `LFM2.5-1.2B-Instruct`
- ✅ Follows Unsloth best practices

**Method 2 (Environment)** - For Gemma 4 or custom configs:
- ⚠️ Requires compatible Docker image for Gemma 4 (transformers >= 4.57.2)
- ✅ More flexible via environment variables
- ✅ Better for custom hyperparameters

> **Gemma 4 Note:** As of 2026-04-10, the Unsloth Docker image (2026.4.4) has transformers 4.57.1 which doesn't support Gemma 4. Use Llama models with Method 1 for now.

**Configuration:** Edit `scripts/submit_hf_job.sh` or `scripts/submit_gemma4_job.sh`:
```bash
DATASET_NAME="duyet/vietnamese-legal-instruct"
BASE_MODEL="unsloth/Llama-3.2-3B-Instruct"
OUTPUT_REPO="duyet/gemma-4-E2B-vietnamese-legal"
HARDWARE="t4-medium"  # or a100.small, a100.large, h100
```

**VRAM Optimization:**
| VRAM | Max Seq Length | Batch Size | Grad Accum |
|------|----------------|------------|------------|
| 16GB | 4096 | 2 | 4 |
| 12GB | 4096 | 1 | 8 |
| 8GB | 2048 | 1 | 8 |

**Monitoring:**
```bash
hf jobs ps                              # List jobs
hf jobs logs <job-id> --follow          # View logs
hf jobs inspect <job-id>                # Job details
bash scripts/hf_jobs_monitor.sh <id>    # Interactive monitor
```

### 5. Deploy RAG Pipeline

```bash
# Install RAG dependencies
uv pip install chromadb langchain langchain-community

# Build vector store
uv run python rag/pipeline.py --rebuild

# Run interactive Q&A
uv run python rag/pipeline.py --interactive \
    --model path/to/model.gguf
```

### HuggingFace Jobs Quick Reference

**Submit a job:**
```bash
# Method 1: UV Run (recommended - faster, simpler)
bash scripts/submit_hf_job.sh

# Method 2: Environment (for Gemma 4 or advanced configs)
bash scripts/submit_gemma4_job.sh

# Paid tier (A100, faster)
HARDWARE=a100.large bash scripts/submit_hf_job.sh
```

**Monitor job:**
```bash
# List all jobs
hf jobs ps

# Stream logs
hf jobs logs <job-id> --follow

# View job details
hf jobs inspect <job-id>
```

**Job results:**
- Auto-uploaded to: `https://huggingface.co/duyet/gemma-4-E2B-vietnamese-legal`
- Includes: LoRA adapters, GGUF export, training metrics
- Can download via: `git clone https://huggingface.co/duyet/gemma-4-E2B-vietnamese-legal`

**Configuration:** Edit the submission script to customize:
- `DATASET_NAME`, `BASE_MODEL`, `OUTPUT_REPO`, `HARDWARE`
- Training params: `MAX_SEQ_LENGTH`, `BATCH_SIZE`, `LEARNING_RATE`, `LORA_R`, `QUANTIZATION`

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
├── scripts/                  # Data processing & automation
│   ├── config.py             # Python config loader
│   ├── config.sh             # Bash config loader
│   ├── git_sync.sh           # Dual GitHub + HF sync
│   ├── colab_train.py        # Colab training script
│   ├── local_train.py        # Local training script
│   ├── submit_hf_job.sh      # Submit HF Job (uv run)
│   ├── submit_gemma4_job.sh  # Submit HF Job (Gemms 4)
│   ├── hf_jobs_monitor.sh    # Monitor HF Jobs
│   ├── setup_training_env.py # Setup training dependencies
│   ├── download_hf_dataset.py
│   ├── merge_datasets.py
│   ├── process_documents.py
│   ├── build_pretrain.py
│   ├── upload_with_xet.py
│   └── setup_hf_repos.sh
│
├── hf_jobs/                  # HuggingFace Jobs training
│   ├── train.py              # Training script (env-based)
│   ├── uv_train.py           # Training script (uv run)
│   └── README.md             # HF Jobs documentation
│
├── notebooks/                # Training notebooks
│   ├── Gemma4_Vietnamese_Legal_Finetune.ipynb  # Main Colab notebook
│   └── README.md            # Notebook documentation
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
    ├── HF_SETUP.md           # HuggingFace setup guide
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
- **Training Dataset**: [duyet/vietnamese-legal-instruct](https://huggingface.co/datasets/duyet/vietnamese-legal-instruct) - Instruction-tuning dataset for Vietnamese legal Q&A
- **Reference Dataset**: [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) (~150K docs from vbpl.vn) - for reference only

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

## 📈 Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Crawler | Pages/hour (single) | ~40 |
| Crawler | Pages/hour (4 workers) | ~150 |
| Processing | 10K docs | ~5 min |
| Embedding | 10K docs | ~10 min |
| RAG Query | Response time | ~2-5s |
| Memory | Total usage | ~8GB |

## 🛠️ Troubleshooting

**Crawler stuck:**
```bash
# Check state
python crawler/playwright_crawler.py --stats

# Resume
python crawler/playwright_crawler.py --resume
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

## 📜 License

Vietnamese legal documents are **public domain** under:
- Law on Access to Information (No. 104/2016/QH13)
- Law on Promulgation of Legal Documents (No. 64/2025/QH15)

The compiled dataset and code are released under **CC BY 4.0**.

## 🙏 Acknowledgments

- Training dataset: [duyet/vietnamese-legal-instruct](https://huggingface.co/datasets/duyet/vietnamese-legal-instruct) - Instruction-tuning dataset for Vietnamese legal Q&A
- Reference dataset: [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) (~150K docs from vbpl.vn) - for reference only
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
