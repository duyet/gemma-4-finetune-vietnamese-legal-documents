---
description: Fine-tune Gemma 4 E2B on Colab
---

Fine-tune Gemma 4 E2B model on Vietnamese legal documents.

**Two-Stage Training:**

**Stage 1: Continued Pretraining** (`notebooks/01_pretrain.ipynb`)
- Embed Vietnamese legal knowledge
- 4-8 hours on Colab T4
- Output: LoRA adapters + GGUF

**Stage 2: SFT for RAG** (`notebooks/02_sft_rag.ipynb`)
- Train on Q&A pairs
- 2-4 hours on Colab T4
- Output: RAG-ready model

**Quick Start:**
```bash
# Upload notebooks to Colab
# 1. Go to https://colab.research.google.com
# 2. Upload notebooks/01_pretrain.ipynb
# 3. Select T4 GPU runtime
# 4. Update YOUR_USERNAME in notebook
# 5. Run all cells
```

**Requirements:**
- Google Colab with T4 GPU (free tier)
- HuggingFace account (for saving model)
- Dataset uploaded to HF or loaded via code

**Model Output:**
- LoRA adapters (~50MB)
- GGUF Q4 quantized (~2GB)
- Ready for local RAG inference

**After Training:**
1. Download GGUF model from Colab
2. Save to `models/` directory
3. Test with RAG pipeline
