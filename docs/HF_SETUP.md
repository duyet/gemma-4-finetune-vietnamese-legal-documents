# HuggingFace Setup Guide

This guide walks you through setting up HuggingFace for the Gemma 4 Vietnamese Legal Documents project.

## Prerequisites

1. A HuggingFace account (free at https://huggingface.co/join)
2. Your HuggingFace username

## Step 1: Install HuggingFace CLI

```bash
# Using pip
pip install huggingface_hub

# Or using UV
uv pip install huggingface_hub
```

## Step 2: Login to HuggingFace

```bash
huggingface-cli login
```

You'll be prompted for:
1. Your HuggingFace access token (get from https://huggingface.co/settings/tokens)
2. Token privileges (select "write" for creating repositories)

## Step 3: Configure Environment

Edit `.env` file with your HuggingFace username:

```bash
# Edit .env
nano .env  # or your preferred editor

# Set your HuggingFace username
HF_USERNAME="your-username"
HF_DATASET_NAME="vietnamese-legal-documents"
HF_MODEL_NAME="gemma-4-vietnamese-legal-rag"
```

## Step 4: Create HuggingFace Repositories

The project provides a script to create both dataset and model repositories:

```bash
# Run the setup script
./scripts/setup_hf_repos.sh
```

This will:
1. Create dataset repository: `your-username/vietnamese-legal-documents`
2. Create model repository: `your-username/gemma-4-vietnamese-legal-rag`
3. Generate proper README files for both

### Manual Repository Creation

Alternatively, create repositories manually:

**Dataset Repository:**
```bash
huggingface-cli repo create \
  --type dataset \
  --yes \
  "your-username/vietnamese-legal-documents"
```

**Model Repository:**
```bash
huggingface-cli repo create \
  --type model \
  --yes \
  "your-username/gemma-4-vietune-legal-rag"
```

## Step 5: Setup Git Remotes

After creating repositories, set up dual-sync:

```bash
# Configure remotes
./scripts/git_sync.sh setup

# Verify remotes
git remote -v
```

Expected output:
```
hf    https://huggingface.co/your-username/gemma-4-finetune-vietnamese-legal-documents (fetch)
hf    https://huggingface.co/your-username/gemma-4-finetune-vietnamese-legal-documents (push)
origin    git@github.com:duyet/gemma-4-finetune-vietnamese-legal-documents.git (fetch)
origin    git@github.com:duyet/gemma-4-finetune-vietnamese-legal-documents.git (push)
```

## Step 6: Push to Both GitHub and HuggingFace

```bash
# Push to both
./scripts/git_sync.sh sync

# Or push individually
./scripts/git_sync.sh push github
./scripts/git_sync.sh push hf
```

## Uploading Dataset

After processing your data, upload to HuggingFace:

```bash
# Prepare dataset
uv run python scripts/prepare_hf_dataset.py

# Upload with XET (faster for large files)
uv run python scripts/upload_with_xet.py -r $HF_USERNAME/$HF_DATASET_NAME
```

## Uploading Model

After training, upload your model:

```python
# In Python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id="your-username/gemma-4-vietnamese-legal-rag",
    repo_type="model",
    exist_ok=True
)

# Push model
model.push_to_hub("your-username/gemma-4-vietnamese-legal-rag")
tokenizer.push_to_hub("your-username/gemma-4-vietnamese-legal-rag")
```

## Troubleshooting

### "Not logged in" error
```bash
huggingface-cli login
```

### "Repository already exists" warning
This is normal if you've created the repository before. The script will continue.

### "Permission denied" error
Make sure your access token has "write" permissions. Get a new token at:
https://huggingface.co/settings/tokens

### XET upload not working
XET is optional. The script will fall back to regular upload if XET is not available.

## Next Steps

- **Crawl data**: `uv run python crawler/playwright_crawler.py --max-pages 100`
- **Process data**: `uv run python scripts/process_documents.py`
- **Train model**: Upload `notebooks/Auto_Train.ipynb` to Google Colab
- **Build RAG**: `uv run python rag/pipeline.py --rebuild`

---

For more details, see:
- [HF_STRUCTURE.md](HF_STRUCTURE.md) - Repository structure details
- [XET_UPLOAD.md](XET_UPLOAD.md) - Fast upload guide with XET
- [README.md](../README.md) - Project documentation
