# HuggingFace Hub Repository Structure for TVPL

## Recommended Repositories

### 1. Dataset Repository (Primary)
**Repo:** `YOUR_USERNAME/tvpl-vi-legal`

**Purpose:** Store all training data (documents, passages, pretrain corpus, SFT data)

**Structure:**
```
YOUR_USERNAME/tvpl-vi-legal
├── README.md                    # Dataset card
├── data/
│   ├── documents/
│   │   ├── train-00000-of-00001.parquet
│   │   └── dataset_info.json
│   ├── passages/
│   │   ├── train-00000-of-00001.parquet
│   │   └── dataset_info.json
│   ├── pretrain/
│   │   ├── train-00000-of-00001.txt
│   │   └── dataset_info.json
│   └── sft/
│       ├── train-00000-of-00001.jsonl
│       └── dataset_info.json
└── repoCARD.md  # Auto-generated
```

**Configs:**
- `documents` - Full legal documents with metadata
- `passages` - Chunked passages for RAG retrieval
- `pretrain` - Text corpus for continued pretraining
- `sft` - Q&A pairs for supervised fine-tuning

**Usage:**
```python
from datasets import load_dataset

# Load documents
docs = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "documents", split="train")

# Load passages
passages = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "passages", split="train")

# Load pretrain corpus
pretrain = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "pretrain", split="train")

# Load SFT data
sft = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "sft", split="train")
```

---

### 2. Model Repository (Primary)
**Repo:** `YOUR_USERNAME/gemma4-tvpl-legal`

**Purpose:** Store fine-tuned Gemma 4 models for Vietnamese legal RAG

**Structure:**
```
YOUR_USERNAME/gemma4-tvpl-legal
├── README.md                    # Model card
├── config.json                 # Model configuration
├── pytorch_model.bin            # PyTorch weights (optional)
├── model.safetensors            # SafeTensors format (recommended)
├── tokenizer.json
├── tokenizer_config.json
├── adapter_config.json          # LoRA config
├── adapter_model.safetensors   # LoRA weights
├── README.md                    # Model card with usage examples
└── repoCARD.md
```

**Versions:**
```
gemma4-tvpl-legal
├── v1.0.0-pretrain   # Stage 1: Continued pretraining LoRA
├── v1.0.0-sft         # Stage 2: SFT for RAG
├── v1.0.0-merged      # Fully merged model (optional)
└── gguf/
    ├── q4_k_m.gguf    # Quantized for local use
    └── q5_k_m.gguf    # Higher quality
```

**Model Card Contents:**
- Model description
- Training data (tvpl-vi-legal dataset)
- Training procedure
- Usage examples
- Performance metrics
- License information
- Citation

---

### 3. Code Repository (Optional - Alternative to GitHub)
**Repo:** `YOUR_USERNAME/tvpl-code`

**Purpose:** Store project code on HuggingFace (HF supports git repos)

**Structure:**
```
YOUR_USERNAME/tvpl-code
├── README.md
├── crawler/
├── scripts/
├── rag/
├── notebooks/
└── pyproject.toml
```

**Note:** You can keep code on GitHub and just reference it from model/dataset cards.

---

## Setup Script

Here's a script to create and populate these repositories: