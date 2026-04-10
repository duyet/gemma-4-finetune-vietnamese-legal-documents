# HF Jobs Submission Guide

This guide explains the two HF Jobs submission methods and when to use each.

## Two Submission Methods

### Method 1: UV Run (`submit_hf_job.sh`)

**Recommended for:** Llama models, standard configurations

```bash
bash scripts/submit_hf_job.sh
```

**How it works:**
- Uses `hf jobs uv run` command
- Passes configuration via command-line arguments
- Follows Unsloth's official best practices
- Simpler, cleaner approach

**Configuration:** Edit `scripts/submit_hf_job.sh`:
```bash
DATASET_NAME="duyet/vietnamese-legal-instruct"
BASE_MODEL="unsloth/Llama-3.2-3B-Instruct"
OUTPUT_REPO="duyet/gemma-4-E2B-vietnamese-legal"
HARDWARE="t4-medium"  # Free tier
```

**Advantages:**
- ✅ Follows Unsloth blog post examples
- ✅ Cleaner argument-based configuration
- ✅ Works with models in Docker image
- ✅ Recommended by Unsloth team

**Limitations:**
- ❌ Requires models compatible with Docker image transformers (4.57.1)
- ❌ Gemma 4 may not work (requires transformers >= 4.57.2)

---

### Method 2: Environment Variables (`submit_gemma4_job.sh`)

**Recommended for:** Gemma 4, custom configurations, full control

```bash
bash scripts/submit_gemma4_job.sh
```

**How it works:**
- Uses `hf jobs run` with Docker
- Passes configuration via environment variables
- Clones repository and runs `hf_jobs/train.py`
- More flexible, can handle custom setups

**Configuration:** Edit `scripts/submit_gemma4_job.sh`:
```bash
DATASET_NAME="duyet/vietnamese-legal-instruct"
BASE_MODEL="unsloth/gemma-4-E2B-it"
OUTPUT_REPO="duyet/gemma-4-E2B-vietnamese-legal"
HARDWARE="t4-medium"  # Free tier
```

**Or set environment variables:**
```bash
export BASE_MODEL="unsloth/gemma-4-E2B-it"
export BATCH_SIZE=2
export LEARNING_RATE=2e-4
# ... more variables
```

**Advantages:**
- ✅ Full control via environment variables
- ✅ Can handle Gemma 4 (with proper Unsloth image)
- ✅ More flexible for custom configs
- ✅ Works with original `train.py`

**Limitations:**
- ❌ More complex setup
- ❌ Requires compatible Docker image for Gemma 4

---

## Choosing the Right Method

| Use Case | Recommended Method |
|----------|-------------------|
| Training Llama 3.2 models | **Method 1: UV Run** |
| Training Gemma 4 | **Method 2: Env** (when compatible image available) |
| Standard configs | **Method 1: UV Run** |
| Custom hyperparameters | **Method 2: Env** |
| Following Unsloth examples | **Method 1: UV Run** |
| Maximum flexibility | **Method 2: Env** |

---

## Model Compatibility

### Models That Work with Method 1 (UV Run)

These models work with the Docker image's transformers version (4.57.1):

✅ `unsloth/Llama-3.2-3B-Instruct`
✅ `unsloth/Llama-3.2-1B-Instruct`
✅ `unsloth/Llama-3.1-8B-Instruct`
✅ `unsloth/Qwen2.5-7B-Instruct`
✅ `LiquidAI/LFM2.5-1.2B-Instruct`

### Models That May Require Method 2 (Env)

⚠️ `unsloth/gemma-4-E2B-it` (requires transformers >= 4.57.2)
⚠️ `unsloth/gemma-4-9B-it` (requires transformers >= 4.57.2)

> **Note:** As of 2025-01-10, the Unsloth Docker image (2026.4.4) has transformers 4.57.1 which doesn't support Gemma 4. Use Llama models for now, or wait for Unsloth to update the Docker image.

---

## Hardware Options

### Free Tier
```bash
HARDWARE=t4-medium  # 16GB VRAM, free
```

### Paid Tiers
```bash
HARDWARE=a100.small      # 40GB VRAM, ~$4/hr
HARDWARE=a100.large      # 80GB VRAM, ~$7/hr
HARDWARE=h100            # 80GB VRAM, ~$7/hr
```

---

## Monitoring Jobs

### List Jobs
```bash
hf jobs ps
```

### View Logs
```bash
hf jobs logs <job-id> --follow
```

### Monitor Script
```bash
bash scripts/hf_jobs_monitor.sh <job-id>
```

### Job Details
```bash
hf jobs inspect <job-id>
```

---

## Training Parameters

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_SEQ_LENGTH` | 4096 | Max sequence length |
| `BATCH_SIZE` | 2 | Batch size per device |
| `GRADIENT_ACCUMULATION` | 4 | Gradient accumulation steps |
| `EPOCHS` | 1 | Number of epochs |
| `LEARNING_RATE` | 2e-4 | Learning rate |
| `LORA_R` | 16 | LoRA rank |
| `LORA_ALPHA` | 16 | LoRA alpha |
| `QUANTIZATION` | q4_k_m | GGUF quantization |

### VRAM Optimization

**For 16GB VRAM (T4, free tier):**
```bash
MAX_SEQ_LENGTH=4096
BATCH_SIZE=2
GRADIENT_ACCUMULATION=4
```

**For 8GB VRAM (local NVIDIA):**
```bash
MAX_SEQ_LENGTH=2048
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
```

---

## Output and Results

### Auto-Uploaded to HuggingFace Hub
- LoRA adapters
- GGUF quantized models
- Training metrics
- Configuration files

### Download Results
```bash
git clone https://huggingface.co/duyet/gemma-4-E2B-vietnamese-legal
```

### Use Model
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "duyet/gemma-4-E2B-vietnamese-legal"
)
```

---

## Troubleshooting

### Job Not Starting
1. Check authentication: `hf auth whoami`
2. Check job logs: `hf jobs logs <job-id>`
3. Verify repository is public or you have access

### OOM (Out of Memory)
1. Reduce `MAX_SEQ_LENGTH` to 2048
2. Reduce `BATCH_SIZE` to 1
3. Increase `GRADIENT_ACCUMULATION` to 8

### Slow Training
1. Upgrade hardware: `HARDWARE=a100.small`
2. Reduce dataset size for testing
3. Enable gradient checkpointing

### Transformers Compatibility
For Gemma 4, you may need to wait for Unsloth Docker image update. Use Llama models for now.

---

## References

- [Unsloth Jobs Blog Post](https://huggingface.co/blog/unsloth-jobs)
- [HuggingFace Jobs Docs](https://huggingface.co/docs/huggingface/jobs)
- [Unsloth Documentation](https://unsloth.ai/docs/)
