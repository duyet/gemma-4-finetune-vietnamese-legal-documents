# TVPL Local RAG Pipeline

Complete RAG setup for running Vietnamese legal Q&A locally on your Mac.

## Architecture

```
User Question (Vietnamese)
    ↓
Embedding (Vietnamese bi-encoder)
    ↓
Vector Store (ChromaDB) ←→ 100K+ legal passages
    ↓
Retriever (Top-K passages)
    ↓
LLM (Gemma 4 E2B GGUF)
    ↓
Answer + Citations
```

## Setup

### 1. Install Dependencies

```bash
uv pip install chromadb langchain langchain-community sentence-transformers
```

### 2. Download Vietnamese Embeddings

```bash
# Automatic download on first use
# Model: bkai-foundation-models/vietnamese-bi-encoder
# Or use: intfloat/multilingual-e5-small
```

### 3. Download Fine-tuned Model (GGUF)

After training on Colab, download the GGUF model:

```bash
# Create model directory
mkdir -p models
cd models

# Download from HuggingFace
huggingface-cli download YOUR_USERNAME/gemma4-e2b-vi-legal-rag \
    --repo-type model \
    --local-dir gemma4-vi-legal \
    --include "gemma4_e2b_vi_law_rag_gguf/*"
```

Or use Ollama:
```bash
echo "FROM ./gemma4-vi-legal.gguf" > Modelfile
ollama create vi-legal -f Modelfile
ollama run vi-legal
```

## Usage

### Build Vector Store

```bash
uv run python rag/pipeline.py --rebuild
```

### Interactive Q&A

```bash
uv run python rag/pipeline.py --interactive --model models/gemma4-vi-legal.gguf
```

### Single Query

```bash
uv run python rag/pipeline.py \
    --query "Điều kiện chuyển nhượng đất nông nghiệp?" \
    --model models/gemma4-vi-legal.gguf
```

## Configuration

Options:
- `--data`: Path to processed documents (default: data/processed/documents.parquet)
- `--model`: Path to GGUF model file
- `--persist-dir`: Vector store directory (default: ./chroma_db)
- `--rebuild`: Rebuild vector store
- `-i, --interactive`: Interactive mode
- `-q, --query`: Single query

## Performance

| Component | Time | Memory |
|-----------|------|--------|
| Embedding (10K docs) | ~10 min | ~2GB |
| Vector store query | ~0.1s | ~200MB |
| LLM generation (Q4) | ~2-5s | ~4GB |

Total: ~8GB (fits 16GB Mac)

## Troubleshooting

**Vector store errors:**
```bash
rm -rf ./chroma_db
uv run python rag/pipeline.py --rebuild
```

**LLM loading errors:**
- Check GGUF: `ls -lh models/`
- Check llama.cpp: `llama-cli --help`
