# HuggingFace Jobs for Gemma 4 Vietnamese Legal Training

## Quick Start

### 1. Create a New Job Repository

```bash
# Create a model repo for the job output
huggingface-cli repo create duyet/gemma-4-vi-legal-job-1 --type model

# Or create via web interface:
# https://huggingface.co/new
```

### 2. Submit Training Job

```bash
# From project root
bash scripts/hf_jobs_submit.sh
```

This will:
- Package training code
- Upload to HuggingFace repo
- Submit job to HF Jobs
- Monitor training progress

### 3. Monitor Job

```bash
# Check job status
huggingface-cli jobs list

# View job logs
huggingface-cli jobs logs <job-id>

# Stream logs in real-time
huggingface-cli jobs logs <job-id> --follow
```

### 4. Download Results

```bash
# Results are automatically uploaded to your HF repo
# https://huggingface.co/duyet/gemma-4-vi-legal-job-1

# Or download via CLI
git clone https://huggingface.co/duyet/gemma-4-vi-legal-job-1
```

## Configuration

Environment variables control training behavior:

```bash
# Model
BASE_MODEL=unsloth/gemma-4-E2B-it
MAX_SEQ_LENGTH=4096
LOAD_IN_4BIT=true

# LoRA
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Training
BATCH_SIZE=2
GRADIENT_ACCUMULATION=4
EPOCHS=1
LEARNING_RATE=2e-4

# Output
HF_USERNAME=duyet
HF_REPO_NAME=gemma-4-vi-legal-rag
PUSH_TO_HUB=true
EXPORT_GGUF=true
QUANTIZATION=q4_k_m
```

## Hardware Options

| Hardware | VRAM | Cost | Batch Size |
|----------|------|------|------------|
| Nvidia T4 | 16GB | Free tier | 2-4 |
| Nvidia A100 | 40GB | $4/hr | 8-16 |
| Nvidia H100 | 80GB | $7/hr | 16-32 |

## Tips

**Free tier training:**
- Use T4 (16GB VRAM)
- Set `BATCH_SIZE=2`, `MAX_SEQ_LENGTH=4096`
- Good for small datasets (<100K examples)

**Production training:**
- Use A100 or H100
- Set `BATCH_SIZE=8+`
- Can process full dataset (324K examples)

**Cost optimization:**
- Use gradient accumulation to simulate larger batches
- Reduce `MAX_SEQ_LENGTH` if OOM occurs
- Start with 1 epoch to test, then scale up
