---
description: Publish dataset to HuggingFace Hub
---

Prepare and upload dataset to HuggingFace Hub.

**Quick Publish:**
```bash
# Prepare and upload
./scripts/publish_dataset.sh

# Custom repo
./scripts/publish_dataset.sh -r your-username/tvpl-vi-legal

# Prepare only (no upload)
./scripts/publish_dataset.sh -p

# Upload only (skip preparation)
./scripts/publish_dataset.sh -u
```

**Manual Upload:**
```bash
# 1. Prepare dataset
uv run tvpl-prepare-hf -r your-username/tvpl-vi-legal

# 2. Upload with XET (fast)
uv run python scripts/upload_with_xet.py \
    -r your-username/tvpl-vi-legal \
    -d data/hf_dataset

# Or use HF CLI
huggingface-cli upload data/hf_dataset your-username/tvpl-vi-legal --repo-type dataset
```

**XET Upload (for large datasets):**
```bash
# Install XET
uv pip install "huggingface_hub[xet]"

# Upload
uv run python scripts/upload_with_xet.py -r your-username/tvpl-vi-legal
```

**Dataset Structure:**
```
your-username/tvpl-vi-legal
├── documents/       # Full documents with metadata
├── passages/        # Chunked passages for retrieval
├── pretrain/        # Pretraining corpus
└── sft/             # SFT instruction pairs
```

**After Upload:**
```python
from datasets import load_dataset

docs = load_dataset("your-username/tvpl-vi-legal", "documents")
passages = load_dataset("your-username/tvpl-vi-legal", "passages")
sft = load_dataset("your-username/tvpl-vi-legal", "sft")
```
