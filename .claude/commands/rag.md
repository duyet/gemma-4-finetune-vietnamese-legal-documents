---
description: Run local RAG pipeline with fine-tuned model
---

Run RAG pipeline locally on Mac with Vietnamese legal documents.

**Setup:**
```bash
# Install RAG dependencies
uv pip install chromadb langchain langchain-community

# Download model (after training)
# From HuggingFace or Colab download
```

**Usage:**
```bash
# Build vector store
uv run python rag/pipeline.py --rebuild

# Interactive Q&A
uv run python rag/pipeline.py --interactive \
    --model models/gemma4-vi-legal.gguf

# Single query
uv run python rag/pipeline.py \
    --query "Điều kiện chuyển nhượng đất?" \
    --model models/gemma4-vi-legal.gguf
```

**Configuration:**
- Embeddings: `bkai-foundation-models/vietnamese-bi-encoder`
- Vector store: ChromaDB (persistent in `./chroma_db`)
- LLM: Gemma 4 E2B GGUF (via llama.cpp)

**Performance:**
- Indexing 10K docs: ~10 min
- Query: ~0.1s retrieval + ~2-5s generation
- Memory: ~8GB total (fits 16GB Mac)

**Examples:**
```bash
# Rebuild index
uv run python rag/pipeline.py --rebuild --data data/processed/documents.parquet

# Interactive mode
uv run python rag/pipeline.py -i -m models/ggml-model-q4_k_m.gguf

# Single question
uv run python rag/pipeline.py -q "Phạt tiền bao nhiêu khi không đăng ký doanh nghiệp?"
```
