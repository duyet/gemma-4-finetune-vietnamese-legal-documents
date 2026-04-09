---
description: Process crawled documents and build datasets
---

Process crawled legal documents and prepare datasets for training.

**Workflow:**
1. Process → Parquet → Passages
2. Build pretraining corpus
3. Build SFT dataset
4. Validate data

**Commands:**
```bash
# Process documents
uv run tvpl-process

# Build pretraining corpus
uv run tvpl-build-pretrain

# Build SFT dataset
uv run tvpl-build-sft

# Validate all data
uv run tvpl-validate
```

**Pipeline:**
```bash
# Full pipeline
uv run tvpl-process && \
uv run tvpl-build-pretrain && \
uv run tvpl-build-sft && \
uv run tvpl-validate
```

**Outputs:**
- `data/processed/documents.parquet` - Full documents with metadata
- `data/processed/passages.parquet` - Chunked passages for RAG
- `data/pretrain/corpus.txt` - Pretraining text corpus
- `data/sft/train.jsonl` - SFT instruction pairs

**Validation:**
- Checks required fields
- Validates Vietnamese content
- Analyzes passage distribution
- Reports document type distribution
- Detects duplicates
