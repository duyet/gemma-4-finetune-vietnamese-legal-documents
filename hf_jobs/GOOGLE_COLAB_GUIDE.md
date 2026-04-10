# Google Colab Training Guide

## Why Colab Instead of HF Jobs?

**HF Jobs Issues:**
- ❌ Unreliable connectivity to HuggingFace/ModelScope
- ❌ 120s timeout on model downloads
- ❌ Limited troubleshooting visibility
- ❌ 10+ consecutive failures

**Colab Advantages:**
- ✅ Direct internet access to HF hub
- ✅ Free T4 GPU (16GB VRAM) - same as HF Jobs t4-medium
- ✅ Jupyter notebooks for interactive debugging
- ✅ Officially supported by Unsloth
- ✅ Better error visibility and troubleshooting

## Quick Start

### 1. Open Colab Notebook
```bash
# From project directory
./scripts/colab.sh  # Opens Colab with the training notebook
```

### 2. Upload Notebook
- File to upload: `notebooks/Gemma4_Vietnamese_Legal_Train.ipynb`
- Or create new notebook with the training code

### 3. Run Training in Colab

```python
# Cell 1: Install Unsloth
!pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# Cell 2: Training code
# (Copy from hf_jobs/train_unsloth_native.py)
```

## Using Existing Training Script

Our optimized `train_unsloth_native.py` works in Colab with minimal changes:

```python
# In Colab cell:
import sys
sys.path.append('/content/drive/MyDrive/')

# Clone repo
!git clone https://github.com/duyet/gemma-4-finetune-vietnamese-legal-documents.git
%cd gemma-4-finetune-vietnamese-legal-documents

# Run training
!python hf_jobs/train_unsloth_native.py
```

## Environment Variables (Set in Colab)

```python
import os
os.environ["BASE_MODEL"] = "unsloth/Llama-3.2-3B-Instruct"
os.environ["DATASET_NAME"] = "duyet/vietnamese-legal-instruct"
os.environ["BATCH_SIZE"] = "2"
os.environ["GRADIENT_ACCUMULATION"] = "4"
os.environ["EPOCHS"] = "1"
os.environ["LEARNING_RATE"] = "2e-4"
os.environ["LORA_R"] = "16"
os.environ["LORA_ALPHA"] = "16"
os.environ["MAX_SEQ_LENGTH"] = "4096"
os.environ["HF_USERNAME"] = "duyet"
os.environ["HF_REPO_NAME"] = "llama-3-2-3b-vietnamese-legal"
os.environ["PUSH_TO_HUB"] = "true"
os.environ["EXPORT_GGUF"] = "true"
os.environ["QUANTIZATION"] = "q4_k_m"
os.environ["HF_TOKEN"] = "your_hf_token_here"  # Set in Colab secrets
```

## Expected Runtime (T4 Free Tier)

- Model loading: ~2-5 minutes
- Dataset loading: ~2 minutes
- Tokenization: ~5 minutes (222K examples)
- Training (1 epoch): ~2-4 hours
- GGUF export: ~20-40 minutes
- **Total: ~3-5 hours**

## After Training

1. **Download models** - Automatically saved to Colab session
2. **Upload to HF Hub** - Script auto-uploads if PUSH_TO_HUB=true
3. **Test inference** - Use provided inference examples

## Troubleshooting in Colab

### OOM Error
```python
# Reduce batch size
os.environ["BATCH_SIZE"] = "1"

# Reduce max length
os.environ["MAX_SEQ_LENGTH"] = "2048"
```

### Slow Training
```python
# Use gradient checkpointing (already enabled in our script)
# Reduce dataset size for testing
```

## Cost Comparison

| Platform | Cost | Time | Reliability |
|----------|------|------|-------------|
| HF Jobs (free) | $0 | 3-5h | ❌ Poor connectivity |
| Colab (free) | $0 | 3-5h | ✅ Excellent |
| Colab Pro | $10/month | 2-3h | ✅ Best performance |
| RunPod A40 | ~$0.79/hr | 1-2h | ✅ Professional |
