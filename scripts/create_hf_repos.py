#!/usr/bin/env python3
"""
Setup HuggingFace repositories for TVPL project.

Creates:
1. Dataset repository: YOUR_USERNAME/tvpl-vi-legal
2. Model repository: YOUR_USERNAME/gemma4-tvpl-legal
3. Generates proper README cards
"""

import json
from pathlib import Path
from datetime import datetime


def create_dataset_readme() -> str:
    """Create dataset README.md."""
    return """---
language:
- vi
- en
license: cc-by-4.0
pretty_name: TVPL - Vietnamese Legal Documents for RAG
tags:
- legal
- vietnamese
- law
- legislation
- documents
- retrieval-augmented-generation
- RAG
task_categories:
- text-generation
- question-answering
task_ids:
- retrieval-augmented-generation
- question-answering
size_categories:
- 100K<n<1M
---

# TVPL - Vietnamese Legal Documents Dataset for RAG

A comprehensive Vietnamese legal document dataset for training and evaluating RAG systems with Gemma 4 and other language models.

## 📚 Dataset Overview

| Config | Examples | Description |
|--------|----------|-------------|
| `documents` | ~150,000+ | Full legal documents with rich metadata |
| `passages` | ~500,000+ | Chunked passages (~512 tokens) for retrieval |
| `pretrain` | ~150,000+ | Text corpus for continued pretraining |
| `sft` | ~50,000+ | Question-answer pairs for SFT |

## 🎯 Use Cases

- **RAG Systems**: Build Vietnamese legal Q&A assistants
- **Legal Research**: Semantic search over laws and regulations
- **Document Analysis**: Extract legal relationships
- **Model Training**: Fine-tune LLMs for Vietnamese legal domain

## 📖 Dataset Details

### Documents (`documents`)
Complete Vietnamese legal documents including:
- **Luật** (Laws)
- **Nghị quyết** (Resolutions)
- **Nghị định** (Decrees)
- **Thông tư** (Circulars)
- **Quyết định** (Decisions)
- And more...

**Metadata fields:**
- Identification: `doc_id`, `title`, `url`, `doc_number`
- Classification: `doc_type`, `category`, `sub_category`, `sector`, `field`
- Authority: `issuing_authority`, `signatory`, `signatory_title`
- Dates: `issue_date`, `effective_date`, `expiry_date`, `published_date`
- Status: `status`, `effect_status`
- Content: `content_html`, `content_text`, `content_markdown`
- Relationships: `amends_docs`, `repeals_docs`, `cites_docs`

### Passages (`passages`)
Documents split into ~512 token passages for retrieval, preserving context boundaries.

### Pretrain (`pretrain`)
Continued pretraining corpus with document headers:
```
<bos>LUẬT SỐ 13/2023/QH15 - LUẬT ĐẤT ĐAI

Type: Luật
Authority: Quốc hội
Issue Date: 2023-01-01

Chương I: QUY ĐỊNH CHUNG...
<eos>
```

### SFT (`sft`)
Question-answer pairs in ShareGPT format:
```json
{
  "conversations": [
    {"role": "system", "content": "Bạn là trợ lý pháp luật Việt Nam..."},
    {"role": "user", "content": "Dựa vào văn bản sau...\\n\\nCâu hỏi: ..."},
    {"role": "assistant", "content": "Answer..."}
  ]
}
```

## 🚀 Usage

### Load Dataset
```python
from datasets import load_dataset

# Load documents
docs = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "documents", split="train")
print(f"Loaded {len(docs)} documents")

# Load passages for RAG
passages = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "passages", split="train")
print(f"Loaded {len(passages)} passages")

# Load pretrain corpus
pretrain = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "pretrain", split="train")
print(f"Loaded {len(pretrain)} training documents")

# Load SFT data
sft = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "sft", split="train")
print(f"Loaded {len(sft)} SFT examples")
```

### Use with Transformers
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model trained on this dataset
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/gemma4-tvpl-legal")
model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/gemma4-tvpl-legal")
```

## 📊 Statistics

- **Total Documents**: ~150,000+
- **Total Passages**: ~500,000+
- **Document Types**: 15+ types
- **Issuing Authorities**: 500+ organizations
- **Time Period**: 1945 – present
- **Languages**: Vietnamese (primary), English (some translations)

## 🏗️ Data Sources

### Primary Source
- **Base Dataset**: [th1nh0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nh0/vietnamese-legal-documents)
  - Source: vbpl.vn (official government portal)
  - Documents: ~150,000
  - License: CC BY 4.0

### Additional Source (if crawled)
- **Crawled Dataset**: thuvienphapluat.vn (via Playwright Cloudflare bypass)
  - Additional documents not in vbpl.vn
  - Includes: Circulars, Q&A, Case law
  - License: Public domain (Vietnamese legal documents)

## 📜 License

Vietnamese legal documents are **public domain** under:
- Law on Access to Information (No. 104/2016/QH13)
- Law on Promulgation of Legal Documents (No. 64/2025/QH15)

The compiled dataset (schema, processing, curation) is released under **CC BY 4.0**.

## 📝 Citation

```bibtex
@dataset{tvpl2024,
  title={TVPL - Vietnamese Legal Documents Dataset for RAG},
  author={Your Name},
  year={2024},
  publisher={HuggingFace},
  howpublished={\\url{https://huggingface.co/datasets/YOUR_USERNAME/tvpl-vi-legal}}
}
```

## 🙏 Acknowledgments

- Base dataset from [th1nh0](https://huggingface.co/datasets/th1nh0/vietnamese-legal-documents)
- Source: [vbpl.vn](https://vbpl.vn) - Official Government Legal Document Portal
- Additional source: [thuvienphapluat.vn](https://thuvienphapluat.vn)

## 📧 Training Gemma 4

See model repository: [YOUR_USERNAME/gemma4-tvpl-legal](https://huggingface.co/models/YOUR_USERNAME/gemma4-tvpl-legal)

### Training Script
```python
# Use with Unsloth for fast training
from unsloth import FastModel
from datasets import load_dataset
from trl import SFTTrainer

# Load dataset
dataset = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "pretrain")

# Load model
model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-4-E2B-it",
    max_seq_length = 4096,
    load_in_4bit = True,
)

# Configure LoRA
model = FastModel.get_peft_model(
    model,
    r = 16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
)
trainer.train()
```

## 📧 RAG Pipeline Example

```python
from datasets import load_dataset
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

# Load passages
passages = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "passages", split="train")

# Setup embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

# Create vector store
vectorstore = Chroma.from_texts(
    texts=[p["text"] for p in passages],
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Setup LLM (your fine-tuned model)
llm = LlamaCpp(
    model_path="path/to/gemma4-tvpl-legal.gguf",
    n_ctx=4096,
)

# Query
question = "Điều kiện chuyển nhượng đất nông nghiệp?"
docs = vectorstore.similarity_search(question, k=3)
# Generate answer...
```

## 🔗 Links

- **Model**: [YOUR_USERNAME/gemma4-tvpl-legal](https://huggingface.co/models/YOUR_USERNAME/gemma4-tvpl-legal)
- **Code**: [GitHub repo](https://github.com/YOUR_USERNAME/tvpl)
- **Original Source**: [thuvienphapluat.vn](https://thuvienphapluat.vn)

---

**Last updated**: {datetime.now().strftime("%Y-%m-%d")}
**Version**: 1.0.0
**Maintainer**: Your Name
"""


def create_model_readme() -> str:
    """Create model README.md."""
    return """---
language:
- vi
- en
license: apache-2.0
tags:
- gemma
- gemma-4
- legal
- vietnamese
- law
- RAG
- retrieval-augmented-generation
- gguf
library_name: transformers
license: apache-2.0
---

# Gemma 4 TVPL - Vietnamese Legal RAG Model

Gemma 4 E2B fine-tuned on Vietnamese legal documents for Retrieval-Augmented Generation (RAG) applications.

## 🎯 Model Overview

- **Base Model**: google/gemma-4-E2B-it (2.3B parameters)
- **Training Dataset**: [YOUR_USERNAME/tvpl-vi-legal](https://huggingface.co/datasets/YOUR_USERNAME/tvpl-vi-legal)
- **License**: Apache 2.0
- **Context Length**: 4096 tokens
- **Quantization**: 4-bit (Q4_K_M) available

## 🚀 Usage

### Load with Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "YOUR_USERNAME/gemma4-tvpl-legal",
    load_in_4bit=True  # For memory efficiency
)
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/gemma4-tvpl-legal")

# Generate
prompt = "Theo Luật Đất đai 2023, điều kiện chuyển nhượng quyền sử dụng đất là gì?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Load GGUF (for local inference)
```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download GGUF
huggingface-cli download YOUR_USERNAME/gemma4-tvpl-legal \
    --include "gguf/*.gguf" \
    --local-dir gemma4-tvpl-legal-gguf

# Run
./main -m gemma4-tvpl-legal-gguf/ggml-model-q4_k_m.gguf \
    -p "Theo Luật Đất đai, điều kiện chuyển nhượng đất là gì?" \
    -n 512
```

### Use with Ollama
```bash
# Create Modelfile
echo "FROM gemma4-tvpl-legal-gguf/ggml-model-q4_k_m.gguf" > Modelfile

# Build model
ollama create gemma4-tvpl-legal -f Modelfile

# Run
ollama run gemma4-tvpl-legal "Theo Luật Đất đai, điều kiện chuyển nhượng đất là gì?"
```

## 📊 Training Details

### Stage 1: Continued Pretraining
- **Dataset**: [YOUR_USERNAME/tvpl-vi-legal](https://huggingface.co/datasets/YOUR_USERNAME/tvpl-vi-legal) (pretrain)
- **Method**: LoRA (Low-Rank Adaptation)
- **Rank**: 16
- **Alpha**: 16
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Epochs**: 1
- **Batch Size**: 2 (per device)
- **Gradient Accumulation**: 4
- **Learning Rate**: 2e-4
- **Hardware**: 1x T4 GPU (16GB VRAM)

### Stage 2: SFT for RAG
- **Dataset**: [YOUR_USERNAME/tvpl-vi-legal](https://huggingface.co/datasets/YOUR_USERNAME/tvpl-vi-legal) (sft)
- **Format**: ShareGPT (chat format)
- **Method**: LoRA
- **Epochs**: 2
- **Learning Rate**: 1e-4

## 📈 Performance

| Metric | Value |
|--------|-------|
| Parameters | 2.3B (Gemma 4 E2B) |
| Trainable Parameters | ~1% (LoRA) |
| VRAM (4-bit) | ~4GB |
| VRAM (8-bit) | ~8GB |
| Context Length | 4096 tokens |
| Vietnamese Support | ✅ Native |
| Multilingual | ✅ 140+ languages |

## 🎯 Use Cases

### 1. Legal Q&A
```python
prompt = \"Dựa vào văn bản sau, trả lời câu hỏi:\\n\\n{context}\\n\\nCâu hỏi: {question}\"
answer = model.generate(prompt)
```

### 2. Document Summarization
```python
from datasets import load_dataset
docs = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "documents")
for doc in docs:
    summary = model.generate(f"Tóm tắt: {doc['title']}\\n\\n{doc['content_text'][:1000]}")
```

### 3. Legal Research
```python
# Semantic search + generation
# See RAG pipeline example below
```

## 🔧 RAG Pipeline

### Complete Setup
```python
from datasets import load_dataset
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

# Load passages
passages = load_dataset("YOUR_USERNAME/tvpl-vi-legal", "passages", split="train")

# Vietnamese embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

# Vector store
vectorstore = Chroma.from_texts(
    texts=[p["text"] for p in passages[:10000]],  # Start with subset
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# LLM (your fine-tuned model)
llm = LlamaCpp(
    model_path="path/to/gemma4-tvpl-legal.gguf",
    n_ctx=4096,
    temperature=0.7,
)

# Query
question = "Phạt tiền bao nhiêu khi không đăng ký doanh nghiệp?"
docs = vectorstore.similarity_search(question, k=3)
context = "\\n\\n".join([d.page_content for d in docs])

answer = llm(f"Dựa vào các văn bản pháp luật sau:\\n\\n{context}\\n\\nCâu hỏi: {question}")
print(answer)
```

## 📦 Model Files

### File Structure
```
gemma4-tvpl-legal
├── adapter_config.json           # LoRA configuration
├── adapter_model.safetensors    # LoRA weights (~50MB)
├── README.md                     # This file
├── config.json                   # Model config
├── merges.txt                    # Tokenizer merges
├── special_tokens_map.json       # Special tokens
├── tokenizer.json                # Tokenizer config
├── tokenizer_config.json         # Tokenizer configuration
└── gguf/
    ├── ggml-model-f16.gguf      # Full precision (9GB)
    ├── ggml-model-q4_k_m.gguf   # 4-bit quantized (2.5GB) ⭐
    └── ggml-model-q5_k_m.gguf   # 5-bit quantized (3GB)
```

## 🏆 Benchmarks

| Benchmark | Score |
|-----------|-------|
| Vietnamese Legal QA | TBD |
| Legal Document Retrieval | TBD |
| Instruction Following | TBD |

## 📜 License

This model is licensed under **Apache 2.0**.

The base Gemma 4 model is licensed by Google DeepMind under Apache 2.0.
The training dataset (TVPL) is CC BY 4.0 (Vietnamese legal documents are public domain).

## 📝 Citation

```bibtex
@model{gemma4_tvpl_legal,
  title={Gemma 4 TVPL - Vietnamese Legal RAG Model},
  author={Your Name},
  year={2024},
  publisher={HuggingFace},
  howpublished={\\url{https://huggingface.co/models/YOUR_USERNAME/gemma4-tvpl-legal}}
}
```

## 🔗 Links

- **Dataset**: [YOUR_USERNAME/tvpl-vi-legal](https://huggingface.co/datasets/YOUR_USERNAME/tvpl-vi-legal)
- **Code**: [GitHub](https://github.com/YOUR_USERNAME/tvpl)
- **Base Model**: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)

## 🙏 Acknowledgments

- **Base Model**: Google DeepMind ([Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma/))
- **Training Framework**: [Unsloth](https://unsloth.ai/)
- **Dataset**: [YOUR_USERNAME/tvpl-vi-legal](https://huggingface.co/datasets/YOUR_USERNAME/tvpl-vi-legal)

---

**Last updated**: {datetime.now().strftime("%Y-%m-%d")}
**Version**: 1.0.0
**Maintainer**: Your Name
"""


def create_model_card() -> str:
    """Create complete model card."""
    return create_model_readme()


if __name__ == "__main__":
    print(create_model_card())
